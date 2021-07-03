#!/usr/bin/env python3
import os.path
import sys
import argparse
import json
import ast
from pathlib import Path
import yaml

# Add package root to pythonpath
sys.path.append(os.path.realpath(f"{__file__}/../../"))

from how.utils import io_helpers, logging, download
from how.stages import evaluate, train

DATASET_URL = "http://ptak.felk.cvut.cz/personal/toliageo/share/how/dataset/"


def add_parameter_arguments(parser_train, parser_eval):
    """Add arguments to given parsers, enabling to overwrite chosen yaml parameters keys"""

    # Dest parameter must be equal to a key in the yaml parameters, separate nested keys by a dot
    # Type parameter can be used for data conversion
    # Metavar parameter for pretty help printing

    # Train
    parser_train.add_argument("--experiment", "-e", metavar="NAME", dest="experiment")
    parser_train.add_argument("--epochs", metavar="EPOCHS", dest="training.epochs", type=int)
    parser_train.add_argument("--architecture", metavar="ARCH", dest="model.architecture")
    parser_train.add_argument("--skip-layer", metavar="ARCH", dest="model.skip_layer", type=int)
    parser_train.add_argument("--loss-margin", "-lm", metavar="MARGIN", dest="training.loss.margin",
                              type=float)

    # Eval
    parser_eval.add_argument("--experiment", "-e", metavar="NAME", dest="experiment")
    parser_eval.add_argument("--model-load", "-ml", metavar="PATH", dest="demo_eval.net_path")
    parser_eval.add_argument("--features-num", metavar="NUM",
                             dest="evaluation.inference.features_num", type=int)
    parser_eval.add_argument("--scales", metavar="SCALES", dest="evaluation.inference.scales",
                             type=ast.literal_eval)
    parser_eval.add_argument("--datasets-local", metavar="JSON", dest="evaluation.local_descriptor.datasets",
                             type=json.loads)
    parser_eval.add_argument("--step", metavar="STEP", dest="evaluation.multistep.step")
    parser_eval.add_argument("--partition", metavar="PARTITION", dest="evaluation.multistep.partition",
                             type=lambda x: tuple(int(y) for y in x.split("_")))
    parser_eval.add_argument("--distractors", metavar="STEP", dest="evaluation.multistep.distractors")


def main(args):
    """Argument parsing and parameter preparation for the demo"""
    # Arguments
    parser = argparse.ArgumentParser(description="HOW demo replicating results from ECCV 2020.")
    subparsers = parser.add_subparsers(title="command", dest="command")
    parser_train = subparsers.add_parser("train", help="Train demo")
    parser_train.add_argument('parameters', type=str,
                              help="Relative path to a yaml file that contains parameters.")
    parser_eval = subparsers.add_parser("eval", help="Eval demo")
    parser_eval.add_argument('parameters', type=str,
                             help="Relative path to a yaml file that contains parameters.")
    add_parameter_arguments(parser_train, parser_eval)
    args = parser.parse_args(args)

    # Load yaml params
    package_root = Path(__file__).resolve().parent.parent
    parameters_path = args.parameters
    if not parameters_path.endswith(".yml"):
        *folders, fname = parameters_path.split(".")
        fname = f"{args.command}_{fname}.yml"
        parameters_path = package_root / "examples/params" / "/".join(folders) / fname
    params = io_helpers.load_params(parameters_path)
    # Overlay with command-line arguments
    for arg, val in vars(args).items():
        if arg not in {"command", "parameters"} and val is not None:
            io_helpers.dict_deep_set(params, arg.split("."), val)

    # Resolve experiment name
    exp_name = params.pop("experiment")
    if not exp_name:
        exp_name = Path(parameters_path).name[:-len(".yml")]

    # Resolve data folders
    globals = {}
    globals["root_path"] = (package_root / params['demo_%s' % args.command]['data_folder'])
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    _overwrite_cirtorch_path(str(globals['root_path']))
    globals["exp_path"] = (package_root / params['demo_%s' % args.command]['exp_folder']) / exp_name
    globals["exp_path"].mkdir(parents=True, exist_ok=True)
    # Setup logging
    globals["logger"] = logging.init_logger(globals["exp_path"] / f"{args.command}.log")

    # Run demo
    io_helpers.save_params(globals["exp_path"] / f"{args.command}_params.yml", params)
    if args.command == "eval":
        download.download_for_eval(params['evaluation'], params['demo_eval'], DATASET_URL, globals)
        evaluate.evaluate_demo(**params, globals=globals)
    elif args.command == "train":
        download.download_for_train(params['validation'], DATASET_URL, globals)
        train.train(**params, globals=globals)


def _overwrite_cirtorch_path(root_path):
    """Hack to fix cirtorch paths"""
    from cirtorch.datasets import traindataset
    from cirtorch.networks import imageretrievalnet

    traindataset.get_data_root = lambda: root_path
    imageretrievalnet.get_data_root = lambda: root_path


if __name__ == "__main__":
    main(sys.argv[1:])
