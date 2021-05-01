# -*- coding: utf-8 -*-
from typing import Dict, List

from videoanalyst.utils import convert_numpy_to_tensor

from ...sampler.sampler_base import SamplerBase
from ..datapipeline_base import (TRACK_DATAPIPELINES, VOS_DATAPIPELINES,
                                 DatapipelineBase)


@TRACK_DATAPIPELINES.register
@VOS_DATAPIPELINES.register
class RegularDatapipeline(DatapipelineBase):
    r"""
    Tracking datapipeline. Integrate sampler togethor with a list of processes

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict()

    def __init__(
            self,
            sampler: SamplerBase,
            pipeline: List = [],
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.pipeline = pipeline

    def __getitem__(self, item) -> Dict:
        r"""
        An interface to load batch data
        """
        sampled_data = self.sampler[item]
        # ###############################################################
        # import cv2
        # import numpy as np
        # from videoanalyst.utils.image import save_image
        # from copy import deepcopy
        # import time
        # number = int(time.time() * 1000)
        # # print('[{}]image_z_path:{}'.format(number, sampled_data['data1']['debug_img_path']))
        # # print('[{}]image_x_path:{}'.format(number, sampled_data['data2']['debug_img_path']))
        # for i in range(3):
        #     image_z_b = deepcopy(sampled_data['data1']['image_{}'.format(i)])
        #     bbox_z_b = sampled_data['data1']['anno_{}'.format(i)][:4].astype(np.int).tolist()
        #     cv2.rectangle(image_z_b, tuple(bbox_z_b[:2]), tuple(bbox_z_b[2:]), (255, 0, 0), 2)
        #     save_image(image_z_b, 'im_z-{}-before_proc-{}'.format(i, number))
        #
        # image_x_b = deepcopy(sampled_data['data2']['image'])
        # bbox_x_b = sampled_data['data2']['anno'][:4].astype(np.int).tolist()
        # cv2.rectangle(image_x_b, tuple(bbox_x_b[:2]), tuple(bbox_x_b[2:]), (0, 255, 255), 2)
        # save_image(image_x_b, 'im_x-before_proc-{}'.format(number))
        # ###############################################################

        for proc in self.pipeline:
            sampled_data = proc(sampled_data)

        # ################################################################
        # for i in range(3):
        #     image_z = deepcopy(sampled_data['im_z'][i]).transpose([1, 2, 0])
        #     bbox_z = sampled_data['bbox_z'][i].astype(np.int).tolist()
        #     cv2.rectangle(image_z, tuple(bbox_z[:2]), tuple(bbox_z[2:]), (0, 255, 0), 2)
        #     save_image(image_z, 'im_z-{}-after_proc-{}'.format(i, number))
        #
        # image_x = deepcopy(sampled_data['im_x']).transpose([1, 2, 0])
        # bbox_x = sampled_data['bbox_x'][:4].astype(np.int).tolist()
        # cv2.rectangle(image_x, tuple(bbox_x[:2]), tuple(bbox_x[2:]), (0, 0, 255), 2)
        # save_image(image_x, 'im_x-after_proc-{}'.format(number))
        # ################################################################

        sampled_data = convert_numpy_to_tensor(sampled_data)
        return sampled_data
