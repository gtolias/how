"""Layer functions"""

import torch
import torch.nn.functional as F

import cirtorch.layers.functional as CF


def smoothing_avg_pooling(feats, kernel_size):
    """Smoothing average pooling

    :param torch.Tensor feats: Feature map
    :param int kernel_size: kernel size of pooling
    :return torch.Tensor: Smoothend feature map
    """
    pad = kernel_size // 2
    return F.avg_pool2d(feats, (kernel_size, kernel_size), stride=1, padding=pad,
                        count_include_pad=False)


def weighted_spoc(ms_feats, ms_weights):
    """Weighted SPoC pooling, summed over scales.

    :param list ms_feats: A list of feature maps, each at a different scale
    :param list ms_weights: A list of weights, each at a different scale
    :return torch.Tensor: L2-normalized global descriptor
    """
    desc = torch.zeros((1, ms_feats[0].shape[1]), dtype=torch.float32, device=ms_feats[0].device)
    for feats, weights in zip(ms_feats, ms_weights):
        desc += (feats * weights).sum((-2, -1)).squeeze()
    return CF.l2n(desc)


def how_select_local(ms_feats, ms_masks, *, scales, features_num):
    """Convert multi-scale feature maps with attentions to a list of local descriptors

    :param list ms_feats: A list of feature maps, each at a different scale
    :param list ms_masks: A list of attentions, each at a different scale
    :param list scales: A list of scales (floats)
    :param int features_num: Number of features to be returned (sorted by attenions)
    :return tuple: A list of descriptors, attentions, locations (x_coor, y_coor) and scales where
            elements from each list correspond to each other
    """
    device = ms_feats[0].device
    size = sum(x.shape[0] * x.shape[1] for x in ms_masks)

    desc = torch.zeros(size, ms_feats[0].shape[1], dtype=torch.float32, device=device)
    atts = torch.zeros(size, dtype=torch.float32, device=device)
    locs = torch.zeros(size, 2, dtype=torch.int16, device=device)
    scls = torch.zeros(size, dtype=torch.float16, device=device)

    pointer = 0
    for sc, vs, ms in zip(scales, ms_feats, ms_masks):
        if len(ms.shape) == 0:
            continue

        height, width = ms.shape
        numel = torch.numel(ms)
        slc = slice(pointer, pointer+numel)
        pointer += numel

        desc[slc] = vs.squeeze(0).reshape(vs.shape[1], -1).T
        atts[slc] = ms.reshape(-1)
        width_arr = torch.arange(width, dtype=torch.int16)
        locs[slc, 0] = width_arr.repeat(height).to(device) # x axis
        height_arr = torch.arange(height, dtype=torch.int16)
        locs[slc, 1] = height_arr.view(-1, 1).repeat(1, width).reshape(-1).to(device) # y axis
        scls[slc] = sc

    keep_n = min(features_num, atts.shape[0]) if features_num is not None else atts.shape[0]
    idx = atts.sort(descending=True)[1][:keep_n]

    return desc[idx], atts[idx], locs[idx], scls[idx]
