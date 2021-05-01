# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.utils import load_image

from ..sampler_base import TRACK_SAMPLERS, VOS_SAMPLERS, SamplerBase


@TRACK_SAMPLERS.register
@VOS_SAMPLERS.register
class TrackPairSampler(SamplerBase):
    r"""
    Tracking data sampler
    Sample procedure:
    __getitem__
    │
    ├── _sample_track_pair
    │   ├── _sample_dataset
    │   ├── _sample_sequence_from_dataset
    │   ├── _sample_track_frame_from_static_image
    │   └── _sample_track_frame_from_sequence
    │
    └── _sample_track_frame
        ├── _sample_dataset
        ├── _sample_sequence_from_dataset
        ├── _sample_track_frame_from_static_image (x2)
        └── _sample_track_pair_from_sequence
            └── _sample_pair_idx_pair_within_max_diff
    Hyper-parameters
    ----------------
    negative_pair_ratio: float
        the ratio of negative pairs
    target_type: str
        "mask" or "bbox"
    """
    default_hyper_params = dict(negative_pair_ratio=0.0, target_type="bbox",
                                num_memory_frames=0,)

    def __init__(self,
                 datasets: List[DatasetBase] = [],
                 seed: int = 0,
                 data_filter=None) -> None:
        super().__init__(datasets, seed=seed)
        if data_filter is None:
            self.data_filter = [lambda x: False]
        else:
            self.data_filter = data_filter

        self._state["ratios"] = [
            d._hyper_params["ratio"] for d in self.datasets
        ]
        sum_ratios = sum(self._state["ratios"])
        self._state["ratios"] = [d / sum_ratios for d in self._state["ratios"]]
        self._state["max_diffs"] = [
            # max_diffs, or -1 (invalid value for video, but not used for static image dataset)
            d._hyper_params.get("max_diff", -1) for d in self.datasets
        ]

    def update_params(self) -> None:
        self.num_memory_frames = self._hyper_params['num_memory_frames']

    def __getitem__(self, item) -> dict:
        is_negative_pair = (self._state["rng"].rand() <
                            self._hyper_params["negative_pair_ratio"])
        is_static_image_m = False
        data_ms = []
        data2 = None
        sample_try_num = 0
        while any(list(map(self.data_filter, data_ms))) or self.data_filter(data2):
            if is_negative_pair:
                data_ms, data2, is_static_image_m = self._sample_neg_track_frames()
            else:
                data_ms, data2, is_static_image_m = self._sample_track_pair()
            for data_m in data_ms:
                data_m["image"] = load_image(data_m["image"])
            data2["image"] = load_image(data2["image"])
            sample_try_num += 1
        data1 = {}
        for i, data_m in enumerate(data_ms):
            for k, v in data_m.items():
                data1['{}_{}'.format(k, i)] = v
        # assert not is_static_image_m
        sampled_data = dict(
            data1=data1,
            data2=data2,
            is_negative_pair=is_negative_pair,
            # is_static_image_m=is_static_image_m,
        )

        return sampled_data

    def _get_len_seq(self, seq_data) -> int:
        return len(seq_data["image"])

    def _sample_track_pair(self) -> Tuple[List, Dict, bool]:
        dataset_idx, dataset = self._sample_dataset()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        len_seq = self._get_len_seq(sequence_data)
        is_static_image_m = False
        if len_seq == 1 and not isinstance(sequence_data["anno"][0], list):
            # static image dataset
            data1 = self._sample_track_frame_from_static_image(sequence_data)
            data2 = deepcopy(data1)
            data1 = [deepcopy(data1) for _ in range(self.num_memory_frames)]
            is_static_image_m = True
        else:
            assert len_seq >= self.num_memory_frames + 1, \
                'dataset_root: {}, len: {}, path: {}'.format(dataset._hyper_params["dataset_root"],
                                                  len_seq,
                                                  sequence_data["image"][0])
            # video dataset
            data1, data2 = self._sample_track_pair_from_sequence(
                sequence_data, self._state["max_diffs"][dataset_idx])

        return data1, data2, is_static_image_m

    def _sample_neg_sequences_from_dataset(self, dataset: DatasetBase) -> Tuple[Dict, Dict]:
        r"""
        """
        rng = self._state["rng"]
        len_dataset = len(dataset)
        idx = rng.choice(len_dataset)
        sequence_data1 = dataset[idx]

        others = list(range(0, idx)) + list(range(idx + 1, len_dataset))
        idx = int(rng.choice(others))
        sequence_data2 = dataset[idx]

        return sequence_data1, sequence_data2

    def _sample_track_frames_from_sequence(self, sequence_data, max_diff) -> List:
        rng = self._state["rng"]
        len_seq = self._get_len_seq(sequence_data)
        idx1, _ = self._sample_pair_idx_pair_within_max_diff(
            len_seq, max_diff)
        assert len(idx1) == self.num_memory_frames
        data1 = []
        for i in range(len(idx1)):
            assert 0 <= idx1[i] < len_seq, 'idx1[{}] = {}'.format(i, idx1[i])
            data_mi = {k: v[idx1[i]] for k, v in sequence_data.items()}
            data1.append(data_mi)
        return data1

    def _sample_neg_track_frames(self) -> Tuple[List, Dict, bool]:
        dataset_idx, dataset = self._sample_dataset()
        sequence_data1, sequence_data2 = self._sample_neg_sequences_from_dataset(dataset)
        len_seq1 = self._get_len_seq(sequence_data1)
        len_seq2 = self._get_len_seq(sequence_data2)
        if len_seq1 == 1:
            data_frame = self._sample_track_frame_from_static_image(sequence_data1)
            data_ms = [deepcopy(data_frame) for _ in range(self.num_memory_frames)]
        else:
            data_ms = self._sample_track_frames_from_sequence(sequence_data1, self._state["max_diffs"][dataset_idx])
        if len_seq2 == 1:
            data_q = self._sample_track_frame_from_static_image(sequence_data2)
            pass
        else:
            data_q = self._sample_track_frame_from_sequence(sequence_data2)
        return data_ms, data_q, len_seq1 == 1

    def _sample_track_frame(self) -> Dict:
        _, dataset = self._sample_dataset()
        sequence_data = self._sample_sequence_from_dataset(dataset)
        len_seq = self._get_len_seq(sequence_data)
        if len_seq == 1:
            # static image dataset
            data_frame = self._sample_track_frame_from_static_image(
                sequence_data)
        else:
            # video dataset
            data_frame = self._sample_track_frame_from_sequence(sequence_data)

        return data_frame

    def _sample_dataset(self):
        r"""
        Returns
        -------
        int
            sampled dataset index
        DatasetBase
            sampled dataset
        """
        dataset_ratios = self._state["ratios"]
        rng = self._state["rng"]
        dataset_idx = rng.choice(len(self.datasets), p=dataset_ratios)
        dataset = self.datasets[dataset_idx]

        return dataset_idx, dataset

    def _sample_sequence_from_dataset(self, dataset: DatasetBase) -> Dict:
        r"""
        """
        rng = self._state["rng"]
        len_dataset = len(dataset)
        idx = rng.choice(len_dataset)

        sequence_data = dataset[idx]

        return sequence_data

    def _generate_mask_for_vos(self, anno):
        mask = Image.open(anno[0])
        mask = np.array(mask, dtype=np.uint8)
        obj_id = anno[1]
        mask[mask != obj_id] = 0
        mask[mask == obj_id] = 1
        return mask

    def _sample_track_frame_from_sequence(self, sequence_data) -> Dict:
        rng = self._state["rng"]
        len_seq = self._get_len_seq(sequence_data)
        idx = rng.choice(len_seq)
        data_frame = {k: v[idx] for k, v in sequence_data.items()}
        # convert mask path to mask, specical for youtubevos and davis
        if self._hyper_params["target_type"] == "mask":
            if isinstance(data_frame["anno"], list):
                mask = self._generate_mask_for_vos(data_frame["anno"])
                data_frame["anno"] = mask
        return data_frame

    def _sample_track_pair_from_sequence(self, sequence_data: Dict,
                                         max_diff: int) -> Tuple[List, Dict]:
        """sample a pair of frames within max_diff distance
        
        Parameters
        ----------
        sequence_data : List
            sequence data: image= , anno=
        max_diff : int
            maximum difference of indexes between two frames in the  pair
        
        Returns
        -------
        Tuple[Dict, Dict]
            track pair data
            data: image= , anno=
        """
        len_seq = self._get_len_seq(sequence_data)
        idx1, idx2 = self._sample_pair_idx_pair_within_max_diff(
            len_seq, max_diff)
        assert len(idx1) == self.num_memory_frames
        data1 = []
        for i in range(len(idx1)):
            assert 0 <= idx1[i] < len_seq, 'idx1[{}] = {}'.format(i, idx1[i])
            data_mi = {k: v[idx1[i]] for k, v in sequence_data.items()}
            data1.append(data_mi)
        data2 = {k: v[idx2] for k, v in sequence_data.items()}
        return data1, data2

    def _sample_pair_idx_pair_within_max_diff(self, L, max_diff):
        r"""
        Draw a pair of index in range(L) within a given maximum difference
        Arguments
        ---------
        L: int
            difference
        max_diff: int
            difference
        """
        rng = self._state["rng"]
        idx1 = rng.choice(L)
        idx2_choices = list(range(idx1-max_diff, L)) + \
                    list(range(L+1, idx1+max_diff+1))
        idx2_choices = list(set(idx2_choices).intersection(set(range(L))))
        assert len(idx2_choices) >= self.num_memory_frames
        idxes = rng.choice(idx2_choices, self.num_memory_frames, replace=False)
        idxes = np.append(idxes, idx1)
        rng.shuffle(idxes)
        idxes = idxes.astype(np.int).tolist()
        return idxes[:-1], idxes[-1]

    def _sample_track_frame_from_static_image(self, sequence_data):
        rng = self._state["rng"]
        num_anno = len(sequence_data['anno'])
        if num_anno > 0:
            idx = rng.choice(num_anno)
            anno = sequence_data["anno"][idx]
        else:
            # no anno, assign a dummy one
            if self._hyper_params["target_type"] == "bbox":
                anno = [-1, -1, -1, -1]
            elif self._hyper_params["target_type"] == "mask":
                anno = np.zeros((sequence_data["image"][0].shape[:2]))
            else:
                logger.error("target type {} is not supported".format(
                    self._hyper_params["target_type"]))
                exit(0)
        data = dict(
            image=sequence_data["image"][0],
            anno=anno,
        )

        return data
