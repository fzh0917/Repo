# -*- coding: utf-8 -*

from loguru import logger

import torch

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class STMTrack(ModuleBase):

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)

    support_phases = ["train", "memorize", "track"]

    def __init__(self, backbone_m, backbone_q, neck_m, neck_q, head, loss=None):
        super(STMTrack, self).__init__()
        self.basemodel_m = backbone_m
        self.basemodel_q = backbone_q
        self.neck_m = neck_m
        self.neck_q = neck_q
        self.head = head
        self.loss = loss
        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def memorize(self, im_crop, fg_bg_label_map):
        fm = self.basemodel_m(im_crop, fg_bg_label_map)
        fm = self.neck_m(fm)
        fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W
        return fm

    def train_forward(self, training_data):
        memory_img = training_data["im_m"]
        query_img = training_data["im_q"]
        # backbone feature
        assert len(memory_img.shape) == 5
        B, T, C, H, W = memory_img.shape

        memory_img = memory_img.view(-1, C, H, W)  # no memory copy
        target_fg_bg_label_map = training_data["fg_bg_label_map"].view(-1, 1, H, W)

        fm = self.basemodel_m(memory_img, target_fg_bg_label_map)
        fm = self.neck_m(fm)  # B * T, C, H, W
        fm = fm.view(B, T, *fm.shape[-3:]).contiguous()  # B, T, C, H, W
        fm = fm.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W

        fq = self.basemodel_q(query_img)
        fq = self.neck_q(fq)

        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(fm, fq)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def forward(self, *args, phase=None):
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        elif phase == 'memorize':
            target_img, fg_bg_label_map = args
            fm = self.memorize(target_img, fg_bg_label_map)
            out_list = fm

        elif phase == 'track':
            assert len(args) == 2
            search_img, fm = args
            fq = self.basemodel_q(search_img)
            fq = self.neck_q(fq)  # B, C, H, W

            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                fm, fq, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

            extra = dict()
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        self._make_convs()
        self._initialize_conv()
        super().update_params()

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
