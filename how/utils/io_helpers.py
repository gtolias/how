"""Helper functions related to io"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path
import yaml
import torch


def progress(iterable, *, size=None, print_freq=1, handle=sys.stdout):
    """Generator wrapping an iterable to print progress"""
    for i, element in enumerate(iterable):
        yield element

        if i == 0 or (i+1) % print_freq == 0 or (i+1) == size:
            if size:
                handle.write(f'\r>>>> {i+1}/{size} done...')
            else:
                handle.write(f'\r>>>> {i+1} done...')

    handle.write("\n")


# Params

def load_params(path):
    """Return loaded parameters from a yaml file"""
    with open(path, "r") as handle:
        content = yaml.safe_load(handle)
    return load_nested_templates(content, os.path.dirname(path))

def save_params(path, params):
    """Save given parameters to a yaml file"""
    with open(path, "w") as handle:
        yaml.safe_dump(params, handle, default_flow_style=False)

def load_nested_templates(params, root_path):
    """Find keys '__template__' in nested dictionary and replace corresponding value with loaded
        yaml file"""
    if not isinstance(params, dict):
        return params

    for key, value in params.items():
        params[key] = load_nested_templates(value, root_path)

    if "__template__" in params:
        template_path = os.path.expanduser(params.pop("__template__"))
        path = os.path.join(root_path, template_path)
        root_path = os.path.dirname(path)
        # Treat template as defaults
        params = dict_deep_overlay(load_params(path), params)

    return params

def dict_deep_overlay(defaults, params):
    """If defaults and params are both dictionaries, perform deep overlay (use params value for
        keys defined in params), otherwise use defaults value"""
    if isinstance(defaults, dict) and isinstance(params, dict):
        for key in params:
            defaults[key] = dict_deep_overlay(defaults.get(key, None), params[key])
        return defaults

    return params

def dict_deep_set(dct, key, value):
    """Set key to value for a nested dictionary where the key is a sequence (e.g. list)"""
    if len(key) == 1:
        dct[key[0]] = value
        return

    if not isinstance(dct[key[0]], dict) or key[0] not in dct:
        dct[key[0]] = {}
    dict_deep_set(dct[key[0]], key[1:], value)


# Download

def download_files(names, root_path, base_url, logfunc=None):
    """Download file names from given url to given directory path. If logfunc given, use it to log
        status."""
    root_path = Path(root_path)
    for name in names:
        path = root_path / name
        if path.exists():
            continue
        if logfunc:
            logfunc(f"Downloading file '{name}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_url + name, path)


# Checkpoints

def save_checkpoint(state, is_best, keep_epoch, directory):
    """Save state dictionary to the directory providing whether the corresponding epoch is the best
        and whether to keep it anyway"""
    path_epoch = os.path.join(directory, 'model_epoch%d.pth' % state['epoch'])
    path_best = os.path.join(directory, 'model_best.pth')
    if is_best and keep_epoch:
        torch.save(state, path_epoch)
        shutil.copyfile(path_epoch, path_best)
    elif is_best or keep_epoch:
        torch.save(state, path_best if is_best else path_epoch)

def last_checkpoint_state_dict(directory):
    """Load the last checkpoint in a directory"""
    epochs = [int(x[len("model_epoch"):-len(".pth")]) for x in os.listdir(directory)
                if x.startswith("model_epoch") and x.endswith(".pth")]
    if not epochs:
        return None
    return torch.load(f"{directory}/model_epoch{sorted(epochs)[-1]}.pth")

def assert_equal_filtered_keys(col1, col2, excluded_keys):
    """Compare two dictionaries while excluding defined keys from the comparison"""
    col1 = {x: y for x, y in col1.items() if x not in excluded_keys}
    col2 = {x: y for x, y in col2.items() if x not in excluded_keys}
    assert col1 == col2, f"{col1} != {col2}"
