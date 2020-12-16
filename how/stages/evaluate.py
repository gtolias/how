"""Implements evaluation of trained models"""

import time
import warnings
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

from cirtorch.datasets.genericdataset import ImagesFromList

from asmk.asmk_method import ASMKMethod
from how.networks import how_net
from how.utils import score_helpers, data_helpers, logging

warnings.filterwarnings("ignore", r"^Possibly corrupt EXIF data", category=UserWarning)


def evaluate_demo(demo_eval, evaluation, globals):
    """Demo evaluating a trained network

    :param dict demo_eval: Demo-related options
    :param dict evaluation: Evaluation-related options
    :param dict globals: Global options
    """
    globals["device"] = torch.device("cpu")
    if demo_eval['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    # Handle net_path when directory
    net_path = Path(demo_eval['exp_folder']) / demo_eval['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"

    # Load net
    state = _convert_checkpoint(torch.load(net_path, map_location='cpu'))
    net = how_net.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])
    globals["transform"] = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    # Eval
    if evaluation['global_descriptor']['datasets']:
        eval_global(net, evaluation['inference'], globals, **evaluation['global_descriptor'])

    if evaluation['local_descriptor']['datasets']:
        eval_asmk(net, evaluation['inference'], globals, **evaluation['local_descriptor'])


def eval_global(net, inference, globals, *, datasets):
    """Evaluate global descriptors"""
    net.eval()
    time0 = time.time()
    logger = globals["logger"]
    logger.info("Starting global evaluation")

    scores = {}
    for dataset in datasets:
        images, qimages, bbxs, gnd = data_helpers.load_dataset(dataset, data_root=globals['root_path'])
        logger.info(f"Evaluating {dataset}")

        with logging.LoggingStopwatch("extracting database images", logger.info, logger.debug):
            dset = ImagesFromList(root='', images=images, imsize=inference['image_size'], bbxs=None,
                                  transform=globals['transform'])
            vecs = how_net.extract_vectors(net, dset, globals["device"], scales=inference['scales'])
        with logging.LoggingStopwatch("extracting query images", logger.info, logger.debug):
            qdset = ImagesFromList(root='', images=qimages, imsize=inference['image_size'], bbxs=bbxs,
                                   transform=globals['transform'])
            qvecs = how_net.extract_vectors(net, qdset, globals["device"], scales=inference['scales'])

        vecs, qvecs = vecs.numpy(), qvecs.numpy()
        ranks = np.argsort(-np.dot(vecs, qvecs.T), axis=0)
        scores[dataset] = score_helpers.compute_map_and_log(dataset, ranks, gnd, logger=logger)

    logger.info(f"Finished global evaluation in {int(time.time()-time0) // 60} min")
    return scores


def eval_asmk(net, inference, globals, *, datasets, codebook_training, asmk):
    """Evaluate local descriptors with ASMK"""
    net.eval()
    time0 = time.time()
    logger = globals["logger"]
    logger.info("Starting asmk evaluation")

    # Train codebook
    images = data_helpers.load_dataset('train', data_root=globals['root_path'])[0]
    images = images[:codebook_training['images']]
    dset = ImagesFromList(root='', images=images, imsize=inference['image_size'], bbxs=None,
                          transform=globals['transform'])
    infer_opts = {"scales": codebook_training['scales'], "features_num": inference['features_num']}
    des_train = how_net.extract_vectors_local(net, dset, globals["device"], **infer_opts)[0]
    asmk = ASMKMethod.initialize_untrained(asmk).train_codebook(des_train)
    logger.info(f"Codebook trained in {asmk.metadata['train_codebook']['train_time']:.1f}s")

    scores = {}
    for dataset in datasets:
        images, qimages, bbxs, gnd = data_helpers.load_dataset(dataset, data_root=globals['root_path'])
        data_opts = {"imsize": inference['image_size'], "transform": globals['transform']}
        infer_opts = {"scales": inference['scales'], "features_num": inference['features_num']}
        logger.info(f"Evaluating {dataset}")

        # Database vectors
        dset = ImagesFromList(root='', images=images, bbxs=None, **data_opts)
        des = how_net.extract_vectors_local(net, dset, globals["device"], **infer_opts)
        asmk_dataset = asmk.build_ivf(des[0], des[1])
        logger.info(f"Indexed images in {asmk_dataset.metadata['build_ivf']['index_time']:.2f}s")
        logger.debug(f"IVF stats: {asmk_dataset.metadata['build_ivf']['ivf_stats']}")

        # Query vectors
        qdset = ImagesFromList(root='', images=qimages, bbxs=bbxs, **data_opts)
        qdes = how_net.extract_vectors_local(net, qdset, globals["device"], **infer_opts)
        metadata, _images, ranks, _scores = asmk_dataset.query_ivf(qdes[0], qdes[1])
        logger.debug(f"Average query time (quant+aggr+search) is {metadata['query_avg_time']:.3f}s")
        scores[dataset] = score_helpers.compute_map_and_log(dataset, ranks.T, gnd, logger=logger)

    logger.info(f"Finished asmk evaluation in {int(time.time()-time0) // 60} min")
    return scores


#
# Helpers
#

def _convert_checkpoint(state):
    """Enable loading checkpoints in the old format"""
    if "_version" not in state:
        # Old checkpoint format
        meta = state['meta']
        state['net_params'] = {
            "architecture": meta['architecture'],
            "pretrained": True,
            "skip_layer": meta['skip_layer'],
            "dim_reduction": {"dim": meta["dim"]},
            "smoothing": {"kernel_size": meta["feat_pool_k"]},
            "runtime": {
                "mean_std": [meta['mean'], meta['std']],
                "image_size": 1024,
                "features_num": 1000,
                "scales": [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25],
                "training_scales": [1],
            },
        }

        state_dict = state['state_dict']
        state_dict['dim_reduction.weight'] = state_dict.pop("whiten.weight")
        state_dict['dim_reduction.bias'] = state_dict.pop("whiten.bias")

        state['_version'] = "how/2020"

    return state
