#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def weighted_l1_loss(pred, gt, weight_map):
    if weight_map is None:
        return torch.abs(pred - gt).mean()

    weight_map = weight_map.to(device=pred.device, dtype=pred.dtype)
    if weight_map.ndim < pred.ndim:
        expand_dims = pred.ndim - weight_map.ndim
        for _ in range(expand_dims):
            weight_map = weight_map.unsqueeze(0)
    weighted_error = torch.abs(pred - gt) * weight_map
    return weighted_error.sum() / weight_map.sum().clamp_min(1e-8)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _slice_to_bchw(slice_tensor):
    if not isinstance(slice_tensor, torch.Tensor):
        slice_tensor = torch.as_tensor(slice_tensor, dtype=torch.float32)
    if not torch.is_floating_point(slice_tensor):
        slice_tensor = slice_tensor.float()
    if slice_tensor.ndim == 2:
        return slice_tensor.unsqueeze(0).unsqueeze(0)
    if slice_tensor.ndim == 3:
        return slice_tensor.unsqueeze(0)
    if slice_tensor.ndim == 4:
        return slice_tensor
    raise ValueError("Slice tensors must have shape (H, W), (1, H, W), or (B, 1, H, W).")


def slice_rendering_loss(volume_render, gt_volume, slice_axis, slice_idx):
    if not isinstance(gt_volume, torch.Tensor):
        gt_volume = torch.as_tensor(gt_volume, dtype=torch.float32)
    axis_map = {"z": 0, "y": 1, "x": 2, 0: 0, 1: 1, 2: 2}
    if slice_axis not in axis_map:
        raise ValueError("slice_axis must be one of {'z', 'y', 'x', 0, 1, 2}.")
    axis_index = axis_map[slice_axis]
    slice_idx = int(slice_idx)
    if axis_index == 0:
        gt_slice = gt_volume[slice_idx]
    elif axis_index == 1:
        gt_slice = gt_volume[:, slice_idx, :]
    else:
        gt_slice = gt_volume[:, :, slice_idx]

    pred = _slice_to_bchw(volume_render)
    gt = _slice_to_bchw(gt_slice).to(device=pred.device, dtype=pred.dtype)
    return 0.8 * weighted_l1_loss(pred, gt, None) + 0.2 * (1.0 - ssim(pred, gt))

