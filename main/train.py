# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import os.path as osp
import sys

import cv2
from loguru import logger

import torch

import random
import os
import numpy as np

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.engine.monitor.monitor_impl.tensorboard_logger import TensorboardLogger

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument(
        '-r',
        '--resume',
        default="",
        help=r"completed epoch's number, latest or one model path")
    parser.add_argument(
        '-v',
        '--validation',
        default="15",
        help=r"Epoch's number to start to evaluate the model on the validation set")

    return parser


if __name__ == '__main__':
    set_seed(1000000007)
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    # root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg.train)
    task_cfg.freeze()
    # log config
    log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(log_dir)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )
    # backup config
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from videoanalyst.config.config.cfg")
    cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # device config
    if task_cfg.device == "cuda":
        world_size = task_cfg.num_processes
        assert torch.cuda.is_available(), "please check your devices"
        assert torch.cuda.device_count(
        ) >= world_size, "cuda device {} is less than {}".format(
            torch.cuda.device_count(), world_size)
        devs = ["cuda:{}".format(i) for i in range(world_size)]
    else:
        devs = ["cpu"]
    # build model
    model = model_builder.build(task, task_cfg.model)
    model.set_device(devs[0])
    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    trainer.set_device(devs)
    trainer.resume(parsed_args.resume)

    # Validator initialization
    root_cfg.test.track.freeze()
    pipeline = pipeline_builder.build("track", root_cfg.test.track.pipeline, model)
    testers = tester_builder("track", root_cfg.test.track.tester, "tester", pipeline)
    epoch_validation = int(parsed_args.validation)
    logger.info("Start to evaluate the model on the validation set after the epoch #{}".format(epoch_validation))

    logger.info("Training begins.")
    while not trainer.is_completed():
        model.train()
        trainer.train()
        trainer.save_snapshot()
        if trainer._state['epoch'] >= epoch_validation:
            logger.info('Validation begins.')
            model.eval()
            for tester in testers:
                res = tester.test()
                benchmark = '{}/{}/{}'.format(tester.__class__.__name__,
                                              tester._hyper_params['subsets'][0],
                                              'AO')
                logger.info('{}: {}'.format(benchmark, res['main_performance']))
                tb_log = {benchmark: res['main_performance']}
                for mo in trainer._monitors:
                    if isinstance(mo, TensorboardLogger):
                        mo.update(tb_log)
                torch.cuda.empty_cache()
            logger.info('Validation ends.')

    # export final model
    trainer.save_snapshot(model_param_only=True)
    logger.info("Training completed.")
