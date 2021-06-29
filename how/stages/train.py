"""Implements training new models"""

import time
import copy
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as transforms

from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.datasets.datahelpers import collate_tuples
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.genericdataset import ImagesFromList

from ..networks import how_net
from ..utils import data_helpers, io_helpers, logging, plots
from . import evaluate


def train(demo_train, training, validation, model, globals):
    """Demo training a network

    :param dict demo_train: Demo-related options
    :param dict training: Training options
    :param dict validation: Validation options
    :param dict model: Model options
    :param dict globals: Global options
    """
    logger = globals["logger"]
    (globals["exp_path"] / "epochs").mkdir(exist_ok=True)
    if (globals["exp_path"] / f"epochs/model_epoch{training['epochs']}.pth").exists():
        logger.info("Skipping network training, already trained")
        return

    # Global setup
    set_seed(0)
    globals["device"] = torch.device("cpu")
    if demo_train['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_train['gpu_id']))

    # Initialize network
    net = how_net.init_network(**model).to(globals["device"])
    globals["transform"] = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])
    with logging.LoggingStopwatch("initializing network whitening", logger.info, logger.debug):
        initialize_dim_reduction(net, globals, **training['initialize_dim_reduction'])

    # Initialize training
    optimizer, scheduler, criterion, train_loader = \
            initialize_training(net.parameter_groups(training["optimizer"]), training, globals)
    validation = Validation(validation, globals)

    for epoch in range(training['epochs']):
        epoch1 = epoch + 1
        set_seed(epoch1)

        time0 = time.time()
        train_loss = train_epoch(train_loader, net, globals, criterion, optimizer, epoch1)

        validation.add_train_loss(train_loss, epoch1)
        validation.validate(net, epoch1)

        scheduler.step()

        io_helpers.save_checkpoint({
            'epoch': epoch1, 'meta': net.meta, 'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(), 'best_score': validation.best_score[1],
            'scores': validation.scores, 'net_params': model, '_version': 'how/2020',
        }, validation.best_score[0] == epoch1, epoch1 == training['epochs'], globals["exp_path"] / "epochs")

        logger.info(f"Epoch {epoch1} finished in {time.time() - time0:.1f}s")


def train_epoch(train_loader, net, globals, criterion, optimizer, epoch1):
    """Train for one epoch"""
    logger = globals['logger']
    batch_time = data_helpers.AverageMeter()
    data_time = data_helpers.AverageMeter()
    losses = data_helpers.AverageMeter()

    # Prepare epoch
    train_loader.dataset.create_epoch_tuples(net)
    net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        num_images = len(input[0]) # number of images per tuple
        for inp, trg in zip(input, target):
            output = torch.zeros(net.meta['outputdim'], num_images).to(globals["device"])
            for imi in range(num_images):
                output[:, imi] = net(inp[imi].to(globals["device"])).squeeze()
            loss = criterion(output, trg.to(globals["device"]))
            loss.backward()
            losses.update(loss.item())

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 20 == 0 or i == 0 or (i+1) == len(train_loader):
            logger.info(f'>> Train: [{epoch1}][{i+1}/{len(train_loader)}]\t' \
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})')

    return losses.avg


def set_seed(seed):
    """Sets given seed globally in used libraries"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def initialize_training(net_parameters, training, globals):
    """Initialize classes necessary for training"""
    # Need to check for keys because of defaults
    assert training['optimizer'].keys() == {"lr", "weight_decay"}
    assert training['lr_scheduler'].keys() == {"gamma"}
    assert training['loss'].keys() == {"margin"}
    assert training['dataset'].keys() == {"name", "mode", "imsize", "nnum", "qsize", "poolsize"}
    assert training['loader'].keys() == {"batch_size"}

    optimizer = torch.optim.Adam(net_parameters, **training["optimizer"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **training["lr_scheduler"])
    criterion = ContrastiveLoss(**training["loss"]).to(globals["device"])
    train_dataset = TuplesDataset(**training['dataset'], transform=globals["transform"])
    train_loader = torch.utils.data.DataLoader(train_dataset, **training['loader'], \
            pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_tuples, \
            num_workers=how_net.NUM_WORKERS)
    return optimizer, scheduler, criterion, train_loader



def extract_train_descriptors(net, globals, *, images, features_num):
    """Extract descriptors for a given number of images from the train set"""
    if features_num is None:
        features_num = net.runtime['features_num']

    images = data_helpers.load_dataset('train', data_root=globals['root_path'])[0][:images]
    dataset = ImagesFromList(root='', images=images, imsize=net.runtime['image_size'], bbxs=None,
                             transform=globals["transform"])
    des_train = how_net.extract_vectors_local(net, dataset, globals["device"],
                                              scales=net.runtime['training_scales'],
                                              features_num=features_num)[0]
    return des_train


def initialize_dim_reduction(net, globals, **kwargs):
    """Initialize dimensionality reduction by PCA whitening from 'images' number of descriptors"""
    if not net.dim_reduction:
        return

    print(">> Initializing dim reduction")
    des_train = extract_train_descriptors(net.copy_excluding_dim_reduction(), globals, **kwargs)
    net.dim_reduction.initialize_pca_whitening(des_train)


class Validation:
    """A convenient interface to validation, keeping historical values and plotting continuously

    :param dict validations: Options for each validation type (e.g. local_descriptor)
    :param dict globals: Global options
    """

    methods = {
        "global_descriptor": evaluate.eval_global,
        "local_descriptor": evaluate.eval_asmk,
    }

    def __init__(self, validations, globals):
        validations = copy.deepcopy(validations)
        self.frequencies = {x: y.pop("frequency") for x, y in validations.items()}
        self.validations = validations
        self.globals = globals
        self.scores = {x: defaultdict(list) for x in validations}
        self.scores["train_loss"] = []

    def add_train_loss(self, loss, epoch):
        """Store training loss for given epoch"""
        self.scores['train_loss'].append((epoch, loss))

        fig = plots.EpochFigure("train set", ylabel="loss")
        fig.plot(*list(zip(*self.scores["train_loss"])), 'o-', label='train')
        fig.save(self.globals['exp_path'] / "fig_train.jpg")

    def validate(self, net, epoch):
        """Perform validation of the network and store the resulting score for given epoch"""
        for name, frequency in self.frequencies.items():
            if frequency and epoch % frequency == 0:
                scores = self.methods[name](net, net.runtime, self.globals, **self.validations[name])
                for dataset, values in scores.items():
                    value = values['map_medium'] if "map_medium" in values else values['map']
                    self.scores[name][dataset].append((epoch, value))

                if "val_eccv20" in scores:
                    fig = plots.EpochFigure(f"val set - {name}", ylabel="mAP")
                    fig.plot(*list(zip(*self.scores[name]['val_eccv20'])), 'o-', label='val')
                    fig.save(self.globals['exp_path'] / f"fig_val_{name}.jpg")

                if scores.keys() - {"val_eccv20"}:
                    fig = plots.EpochFigure(f"test set - {name}", ylabel="mAP")
                    for dataset, value in self.scores[name].items():
                        if dataset != "val_eccv20":
                            fig.plot(*list(zip(*value)), 'o-', label=dataset)
                    fig.save(self.globals['exp_path'] / f"fig_test_{name}.jpg")

    @property
    def decisive_scores(self):
        """List of pairs (epoch, score) where score is decisive for comparing epochs"""
        for name in ["local_descriptor", "global_descriptor"]:
            if self.frequencies[name] and "val_eccv20" in self.scores[name]:
                return self.scores[name]['val_eccv20']
        return self.scores["train_loss"]

    @property
    def last_epoch(self):
        """Tuple (last epoch, last score) or (None, None) before decisive score is computed"""
        decisive_scores = self.decisive_scores
        if not decisive_scores:
            return None, None

        return decisive_scores[-1]

    @property
    def best_score(self):
        """Tuple (best epoch, best score) or (None, None) before decisive score is computed"""
        decisive_scores = self.decisive_scores
        if not decisive_scores:
            return None, None

        aggr = min
        for name in ["local_descriptor", "global_descriptor"]:
            if self.frequencies[name] and "val_eccv20" in self.scores[name]:
                aggr = max
        return aggr(decisive_scores, key=lambda x: x[1])
