# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Dict

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target


@TRACK_TARGETS.register
class DenseboxTarget(TargetBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        q_size=289,
        score_size=25,
        score_offset=-1,
        total_stride=8,
        num_memory_frames=0,
    )

    def __init__(self) -> None:
        super().__init__()

    def update_params(self):
        hps = self._hyper_params
        hps['score_offset'] = (
            hps['q_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def __call__(self, sampled_data: Dict) -> Dict:
        data_m = sampled_data["data1"]
        im_ms = []
        bbox_ms = []
        nmf = self._hyper_params['num_memory_frames']
        for i in range(nmf):
            im_ms.append(data_m['image_{}'.format(i)])
            bbox_ms.append(data_m['anno_{}'.format(i)])

        data_q = sampled_data["data2"]
        im_q, bbox_q = data_q["image"], data_q["anno"]

        is_negative_pair = sampled_data["is_negative_pair"]

        # input tensor
        im_m = np.stack(im_ms, axis=0)
        im_m = im_m.transpose(0, 3, 1, 2)  # T, C, H, W
        im_q = im_q.transpose(2, 0, 1)

        # training target
        cls_label, ctr_label, box_label = make_densebox_target(
            bbox_q.reshape(1, 4), self._hyper_params)
        if is_negative_pair:
            cls_label[cls_label == 0] = -1
            cls_label[cls_label == 1] = 0

        bbox_m = np.stack(bbox_ms, axis=0)
        fg_bg_label_map = torch.zeros(size=(im_m.shape[0], *im_m.shape[-2:]), dtype=torch.float32)
        bz = bbox_m.astype(np.int)
        for i in range(nmf):
            fg_bg_label_map[i, bz[i, 1]:bz[i, 3] + 1, bz[i, 0]:bz[i, 2] + 1] = 1

        training_data = dict(
            im_m=im_m,
            im_q=im_q,
            bbox_m=bbox_m,
            bbox_q=bbox_q,
            cls_gt=cls_label,
            ctr_gt=ctr_label,
            box_gt=box_label,
            fg_bg_label_map=fg_bg_label_map,
            is_negative_pair=int(is_negative_pair),
        )

        return training_data
