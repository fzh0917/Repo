# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from torch import nn

from .backbone import builder as backbone_builder
from .neck import builder as neck_builder
from .loss import builder as loss_builder
from .task_head import builder as head_builder
from .task_model import builder as task_builder
from .sync_batchnorm import convert_model


def build(
        task: str,
        cfg: CfgNode,
):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: model

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task == "track":
        backbone_m = backbone_builder.build(task, cfg.backbone_m)
        backbone_q = backbone_builder.build(task, cfg.backbone_q)
        neck_m = neck_builder.build(task, cfg.neck)
        neck_q = neck_builder.build(task, cfg.neck)
        head = head_builder.build(task, cfg.task_head)
        losses = loss_builder.build(task, cfg.losses)
        task_model = task_builder.build(task, cfg.task_model, backbone_m,
                                        backbone_q, neck_m, neck_q, head, losses)

    else:
        logger.error("model for task {} has not been implemented".format(task))
        exit(-1)
    if cfg.use_sync_bn:
        logger.warning('Convert BatchNorm to SyncBatchNorm.')
        task_model = convert_model(task_model)

    return task_model


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}

    for task in cfg_dict:
        cfg = cfg_dict[task]

        cfg["backbone_m"] = backbone_builder.get_config(task_list)[task]
        cfg["backbone_q"] = backbone_builder.get_config(task_list)[task]
        cfg["neck"] = neck_builder.get_config(task_list)[task]
        cfg["task_head"] = head_builder.get_config(task_list)[task]
        cfg["losses"] = loss_builder.get_config(task_list)[task]
        cfg["task_model"] = task_builder.get_config(task_list)[task]
        cfg["use_sync_bn"] = False

    return cfg_dict
