"""Functions for downloading files necessary for training and evaluation"""

import os.path
from cirtorch.utils.download import download_train, download_test
from . import io_helpers


def download_for_eval(evaluation, demo_eval, dataset_url, globals):
    """Download datasets for evaluation and network if given by url"""
    # Datasets
    datasets = evaluation['global_descriptor']['datasets'] \
                + evaluation['local_descriptor']['datasets']
    download_datasets(datasets, dataset_url, globals)
    # Network
    if demo_eval and (demo_eval['net_path'].startswith("http://") \
                        or demo_eval['net_path'].startswith("https://")):
        net_name = os.path.basename(demo_eval['net_path'])
        io_helpers.download_files([net_name], globals['root_path'] / "models",
                                  os.path.dirname(demo_eval['net_path']) + "/",
                                  logfunc=globals["logger"].info)
        demo_eval['net_path'] = globals['root_path'] / "models" / net_name


def download_for_train(validation, dataset_url, globals):
    """Download datasets for training"""

    datasets = ["train"] + validation['global_descriptor']['datasets'] \
                + validation['local_descriptor']['datasets']
    download_datasets(datasets, dataset_url, globals)


def download_datasets(datasets, dataset_url, globals):
    """Download data associated with each required dataset"""

    if "val_eccv20" in datasets:
        download_train(globals['root_path'])
        io_helpers.download_files(["retrieval-SfM-120k-val-eccv2020.pkl"],
                                  globals['root_path'] / "train/retrieval-SfM-120k",
                                  dataset_url, logfunc=globals["logger"].info)
    elif "train" in datasets:
        download_train(globals['root_path'])

    if "roxford5k" in datasets or "rparis6k" in datasets:
        download_test(globals['root_path'])
