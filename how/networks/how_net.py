"""Module of the HOW method"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

from cirtorch.networks import imageretrievalnet

from .. import layers
from ..layers import functional as HF
from ..utils import io_helpers

NUM_WORKERS = 6

CORERCF_SIZE = {
    'resnet18': 32,
    'resnet50': 32,
    'resnet101': 32,
}


class HOWNet(nn.Module):
    """Network for the HOW method

    :param list features: A list of torch.nn.Module which act as feature extractor
    :param torch.nn.Module attention: Attention layer
    :param torch.nn.Module smoothing: Smoothing layer
    :param torch.nn.Module dim_reduction: Dimensionality reduction layer
    :param dict meta: Metadata that are stored with the network
    :param dict runtime: Runtime options that can be used as default for e.g. inference
    """

    def __init__(self, features, attention, smoothing, dim_reduction, meta, runtime):
        super().__init__()

        self.features = features
        self.attention = attention
        self.smoothing = smoothing
        self.dim_reduction = dim_reduction

        self.meta = meta
        self.runtime = runtime


    def copy_excluding_dim_reduction(self):
        """Return a copy of this network without the dim_reduction layer"""
        meta = {**self.meta, "outputdim": self.meta['backbone_dim']}
        return self.__class__(self.features, self.attention, self.smoothing, None, meta, self.runtime)

    def copy_with_runtime(self, runtime):
        """Return a copy of this network with a different runtime dict"""
        return self.__class__(self.features, self.attention, self.smoothing, self.dim_reduction, self.meta, runtime)


    # Methods of nn.Module

    @staticmethod
    def _set_batchnorm_eval(mod):
        if mod.__class__.__name__.find('BatchNorm') != -1:
            # freeze running mean and std
            mod.eval()

    def train(self, mode=True):
        res = super().train(mode)
        if mode:
            self.apply(HOWNet._set_batchnorm_eval)
        return res

    def parameter_groups(self, optimizer_opts):
        """Return torch parameter groups"""
        layers = [self.features, self.attention, self.smoothing]
        parameters = [{'params': x.parameters()} for x in layers if x is not None]
        if self.dim_reduction:
            # Do not update dimensionality reduction layer
            parameters.append({'params': self.dim_reduction.parameters(), 'lr': 0.0})
        return parameters


    # Forward

    def features_attentions(self, x, *, scales):
        """Return a tuple (features, attentions) where each is a list containing requested scales"""
        feats = []
        masks = []
        for s in scales:
            xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
            o = self.features(xs)
            m = self.attention(o)
            if self.smoothing:
                o = self.smoothing(o)
            if self.dim_reduction:
                o = self.dim_reduction(o)
            feats.append(o)
            masks.append(m)

        # Normalize max weight to 1
        mx = max(x.max() for x in masks)
        masks = [x/mx for x in masks]

        return feats, masks

    def forward(self, x):
        return self.forward_global(x, scales=self.runtime['training_scales'])

    def forward_global(self, x, *, scales):
        """Return global descriptor"""
        feats, masks = self.features_attentions(x, scales=scales)
        return HF.weighted_spoc(feats, masks)

    def forward_local(self, x, *, features_num, scales):
        """Return local descriptors"""
        feats, masks = self.features_attentions(x, scales=scales)
        return HF.how_select_local(feats, masks, scales=scales, features_num=features_num)


    # String conversion

    def __repr__(self):
        meta_str = "\n".join("    %s: %s" % x for x in self.meta.items())
        return "%s(meta={\n%s\n})" % (self.__class__.__name__, meta_str)

    def meta_repr(self):
        """Return meta representation"""
        return str(self)


def init_network(architecture, pretrained, skip_layer, dim_reduction, smoothing, runtime):
    """Initialize HOW network

    :param str architecture: Network backbone architecture (e.g. resnet18)
    :param bool pretrained: Whether to start with a network pretrained on ImageNet
    :param int skip_layer: How many layers of blocks should be skipped (from the end)
    :param dict dim_reduction: Options for the dimensionality reduction layer
    :param dict smoothing: Options for the smoothing layer
    :param dict runtime: Runtime options to be stored in the network
    :return HOWNet: Initialized network
    """
    # Take convolutional layers as features, always ends with ReLU to make last activations non-negative
    net_in = getattr(torchvision.models, architecture)(pretrained=pretrained)
    if architecture.startswith('alexnet') or architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children()) + [nn.ReLU(inplace=True)]
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    if skip_layer > 0:
        features = features[:-skip_layer]
    backbone_dim = imageretrievalnet.OUTPUT_DIM[architecture] // (2 ** skip_layer)

    att_layer = layers.attention.L2Attention()
    smooth_layer = None
    if smoothing:
        smooth_layer = layers.pooling.SmoothingAvgPooling(**smoothing)
    reduction_layer = None
    if dim_reduction:
        reduction_layer = layers.dim_reduction.ConvDimReduction(**dim_reduction, input_dim=backbone_dim)

    meta = {
        "architecture": architecture,
        "backbone_dim": backbone_dim,
        "outputdim": reduction_layer.out_channels if dim_reduction else backbone_dim,
        "corercf_size": CORERCF_SIZE[architecture] // (2 ** skip_layer),
    }
    return HOWNet(nn.Sequential(*features), att_layer, smooth_layer, reduction_layer, meta, runtime)


def extract_vectors(net, dataset, device, *, scales):
    """Return global descriptors in torch.Tensor"""
    net.eval()
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    with torch.no_grad():
        vecs = torch.zeros(len(loader), net.meta['outputdim'])
        for i, inp in io_helpers.progress(enumerate(loader), size=len(loader), print_freq=100):
            vecs[i] = net.forward_global(inp.to(device), scales=scales).cpu().squeeze()

    return vecs


def extract_vectors_local(net, dataset, device, *, features_num, scales):
    """Return tuple (local descriptors, image ids, strenghts, locations and scales) where locations
        consists of (coor_x, coor_y, scale) and elements of each list correspond to each other"""
    net.eval()
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    with torch.no_grad():
        vecs, strengths, locs, scls, imids = [], [], [], [], []
        for imid, inp in io_helpers.progress(enumerate(loader), size=len(loader), print_freq=100):
            output = net.forward_local(inp.to(device), features_num=features_num, scales=scales)

            vecs.append(output[0].cpu().numpy())
            strengths.append(output[1].cpu().numpy())
            locs.append(output[2].cpu().numpy())
            scls.append(output[3].cpu().numpy())
            imids.append(np.full((output[0].shape[0],), imid))

    return np.vstack(vecs), np.hstack(imids), np.hstack(strengths), np.vstack(locs), np.hstack(scls)
