# -*- coding: utf-8 -*
from collections import Iterable
from typing import Tuple

import cv2
import numpy as np

from .bbox import cxywh2xyxy
from .misc import imarray_to_tensor

import torch
import torch.nn.functional as F


def get_axis_aligned_bbox(region):
    r"""
    Get axis-aligned bbox (used to transform annotation in VOT benchmark)

    Arguments
    ---------
    region: list (nested)
        (1, 4, 2), 4 points of the rotated bbox

    Returns
    -------
    tuple
        axis-aligned bbox in format (cx, cy, w, h)
    """
    try:
        region = np.array([
            region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
            region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]
        ])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
        np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h


def get_crop_numpy(im: np.ndarray, pos: np.ndarray, sample_sz: np.ndarray, output_sz: np.ndarray = None,
                   mode: str = 'constant', avg_chans=(0, 0, 0), max_scale_change=None):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.astype(np.int).copy()

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        # im_sz = torch.tensor([im.shape[2], im.shape[3]], device=im.device)
        # shrink_factor = (sample_sz.float() / im_sz)
        im_sz = np.array([im.shape[0], im.shape[1]])
        shrink_factor = (sample_sz.astype(np.float) / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        # sample_sz = (sample_sz.float() / shrink_factor).long()
        sample_sz = (sample_sz.astype(np.float) / shrink_factor).astype(np.int)

    # Compute pre-downsampling factor
    if output_sz is not None:
        # resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        resize_factor = np.min(sample_sz.astype(np.float) / output_sz.astype(np.float)).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    # sz = sample_sz.float() / df  # new size
    sz = sample_sz.astype(np.float) / df

    # Do downsampling
    if df > 1:
        os = posl % df  # offset
        posl = (posl - os) // df  # new position
        im2 = im[os[0].item()::df, os[1].item()::df, :]  # downsample
    else:
        im2 = im

    # compute size to crop
    # szl = torch.max(sz.round(), torch.tensor([2.0], dtype=sz.dtype, device=sz.device)).long()
    szl = np.maximum(np.round(sz), 2.0).astype(np.int)

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        # im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        # shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        im2_sz = np.array([im2.shape[0], im2.shape[1]], dtype=np.int)
        shift = np.clip(-tl, 0) - np.clip(br - im2_sz, 0)
        tl += shift
        br += shift

        # outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        # shift = (-tl - outside) * (outside > 0).long()
        outside = (np.clip(-tl, 0) - np.clip(br - im2_sz, 0)) // 2
        shift = (-tl - outside) * (outside > 0).astype(np.int)
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    crop_xyxy = np.array([tl[1], tl[0], br[1], br[0]])
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2] - M_13) / (output_sz[0] - 1)
    M_22 = (crop_xyxy[3] - M_23) / (output_sz[1] - 1)
    mat2x3 = np.array([
        M_11,
        0,
        M_13,
        0,
        M_22,
        M_23,
    ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im2,
                              mat2x3, (output_sz[0], output_sz[1]),
                              flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(map(int, avg_chans)))
    # Get image coordinates
    patch_coord = df * np.concatenate([tl, br]).reshape(1, 4)
    scale = output_sz / (np.array([br[1] - tl[1] + 1, br[0] - tl[0] + 1]) * df)
    return im_patch, patch_coord, scale


def get_crop_torch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        pad_mode = 'replicate'
        im_sz = torch.tensor([im.shape[2], im.shape[3]], device=im.device)
        shrink_factor = (sample_sz.float() / im_sz)
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.float() / df  # new size

    # Do downsampling
    if df > 1:
        os = posl % df  # offset
        posl = (posl - os) // df  # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
    else:
        im2 = im

    # compute size to crop
    szl = torch.max(sz.round(), torch.tensor([2.0], dtype=sz.dtype, device=sz.device)).long()

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]


    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]),
                         mode=pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # Get image coordinates
    patch_coord = df * torch.cat((tl, br)).view(1, 4)

    scale = output_sz / (torch.tensor(im_patch.shape, device=im_patch.device)[-2:] * df)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord, scale

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord, scale


def get_crop_single(im: np.ndarray, target_pos: np.ndarray, target_scale: float, output_sz: int, avg_chans: tuple):
    pos = target_pos[::-1]
    output_sz = np.array([output_sz, output_sz])
    sample_sz = target_scale * output_sz
    im_patch, _, scale_x = get_crop_numpy(im, pos, sample_sz, output_sz, avg_chans=avg_chans)
    return im_patch, scale_x[0].item()

