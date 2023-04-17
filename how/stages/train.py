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
    globals["transform"] = how_net.build_transforms(net.runtime)
    if training['initialize_dim_reduction'] is not False:
        with logging.LoggingStopwatch("initializing network whitening", logger.info, logger.debug):
            initialize_dim_reduction(net, globals, **training['initialize_dim_reduction'])

    # Initialize training
    optimizer, scheduler, criterion, train_loader = \
            initialize_training(net.parameter_groups(training["optimizer"]), training, globals)
    validation = Validation(validation, globals)

    if not training['epochs']:
        # Save offtheshelf
        io_helpers.save_checkpoint({
            'epoch': 0, 'meta': net.meta, 'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_score': validation.best_score[1], 'scores': validation.scores,
            'net_params': model, '_version': 'how/2020',
        }, False, True, globals["exp_path"] / "epochs")
        (globals["exp_path"] / "epochs/model_best.pth").symlink_to("model_epoch0.pth")

    for epoch in range(training['epochs']):
        epoch1 = epoch + 1
        set_seed(epoch1)

        time0 = time.time()
        train_stats = train_epoch(train_loader, net, globals, criterion, optimizer, epoch1)

        validated = validation.validate(net, epoch1)
        train_stats['epoch_durations'] = time.time() - time0
        validation.add_train_stats(train_stats, epoch1)

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
    avg_neg_dist = train_loader.dataset.create_epoch_tuples(net)
    net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        num_images = len(input[0]) # number of images per tuple
        for inp, trg in zip(input, target):
            loss = process_batch(net, criterion, inp, trg, globals['device'])
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

    return {"train_loss": losses.avg, "avg_neg_dist": avg_neg_dist}

def process_batch(net, criterion, input, target, device):
    num_images = len(input)
    output = torch.zeros(net.meta['outputdim'], num_images).to(device)
    for imi in range(num_images):
        output[:, imi] = net(input[imi].to(device)).squeeze()
    return criterion(output, target.to(device))


def set_seed(seed):
    """Sets given seed globally in used libraries"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def initialize_loss(loss, device):
    """Initialize loss Module"""
    assert training['loss'].keys() == {"margin"}
    return ContrastiveLoss(**loss).to(device)


def initialize_training(net_parameters, training, globals):
    """Initialize classes necessary for training"""
    # Need to check for keys because of defaults
    assert training['optimizer'].keys() == {"lr", "weight_decay"}
    assert training['lr_scheduler'].keys() == {"gamma"}
    assert training['dataset'].keys() == {"name", "mode", "imsize", "nnum", "qsize", "poolsize"}
    assert training['loader'].keys() == {"batch_size"}

    optimizer = torch.optim.Adam(net_parameters, **training["optimizer"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **training["lr_scheduler"])
    criterion = initialize_loss(training['loss'], globals['device'])
    train_dataset = TuplesDataset(**training['dataset'], transform=globals["transform"])
    train_loader = torch.utils.data.DataLoader(train_dataset, **training['loader'], \
            pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_tuples, \
            num_workers=how_net.NUM_WORKERS)
    return optimizer, scheduler, criterion, train_loader


def load_checkpoint(state, net, optimizer, scheduler, validation, net_params, scheduler_params):
    """Load state for all objects, return the loaded epoch index starting from 1 (0 if not loaded)"""
    if not state:
        return 0

    io_helpers.assert_equal_filtered_keys(net_params, state['net_params'], ['architecture'])
    io_helpers.assert_equal_filtered_keys(net.meta, state['meta'], ['architecture'])
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    if 'scheduler' not in state:
        # Backwards compatibility
        state['scheduler'] = scheduler.__class__(optimizer, **scheduler_params,
                                                 last_epoch=state['epoch']).state_dict()
    scheduler.load_state_dict(state['scheduler'])
    validation.scores.update(state['scores'])
    assert validation.best_score[1] == state['best_score']
    print(f">> Loaded epoch {state['epoch']}")
    return state['epoch']


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
        self.scores = defaultdict(list)
        self.scores.update({x: defaultdict(list) for x in validations})

    def add_train_loss(self, train_loss, epoch):
        """Store training loss only for given epoch"""
        self.add_train_stats({"train_loss": train_loss}, epoch)

    def add_train_stats(self, stats, epoch):
        """Store training stats (e.g. train_loss) for given epoch"""
        for stat, val in stats.items():
            self.scores[stat].append((epoch, val))

        if "train_loss" in self.scores:
            fig = plots.EpochFigure("train set", ylabel="loss")
            fig.plot(*list(zip(*self.scores["train_loss"])), 'o-', label='train')
            fig.save(self.globals['exp_path'] / "fig_train.jpg")

    def validate(self, net, epoch):
        """Perform validation of the network and store the resulting score for given epoch"""
        net.eval()
        # Free torch cached gpu mem for faiss
        torch.cuda.empty_cache()
        validated = False
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
                validated = True
        return validated

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
        return aggr(decisive_scores, key=lambda x: float(x[1]))
