# -*- coding: utf-8 -*
import random
from typing import Dict

import cv2
import numpy as np

from ..transformer_base import (TRACK_TRANSFORMERS, VOS_TRANSFORMERS,
                                TransformerBase)


class RandomBlur(object):
    def __init__(self, ratio=0.25):
        self.ratio = ratio

    def __call__(self, sample):
        if np.random.rand(1) < self.ratio:
            # random kernel size
            kernel_size = np.random.choice([3, 5, 7])
            # random gaussian sigma
            sigma = np.random.rand() * 5
            return cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
        else:
            return sample


def gray_aug(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def fb_brightness_aug(img, val):
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha

    return img


def fb_grayscale(img):
    w = np.array([0.114, 0.587, 0.299]).reshape(1, 1, 3)
    gs = np.zeros(img.shape[:2])
    gs = (img * w).sum(axis=2, keepdims=True)

    return gs


def fb_contrast_aug(img, val):
    gs = fb_grayscale(img)
    gs[:] = gs.mean()
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha + gs * (1 - alpha)

    return img


def fb_saturation_aug(img, val):
    gs = fb_grayscale(img)
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha + gs * (1 - alpha)

    return img


def fb_color_jitter(img, brightness, contrast, saturation):
    augs = [(fb_brightness_aug, brightness), (fb_contrast_aug, contrast),
            (fb_saturation_aug, saturation)]
    random.shuffle(augs)

    for aug, val in augs:
        img = aug(img, val)

    return img


def fb_lighting(img, std):
    eigval = np.array([0.2175, 0.0188, 0.0045])
    eigvec = np.array([
        [-0.5836, -0.6948, 0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675, 0.7192, 0.4009],
    ])
    if std == 0:
        return img

    alpha = np.random.randn(3) * std
    bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
    bgr = bgr.sum(axis=1).reshape(1, 1, 3)
    img = img + bgr

    return img


@TRACK_TRANSFORMERS.register
@VOS_TRANSFORMERS.register
class ImageAug(TransformerBase):
    default_hyper_params = dict(
        color_jitter_brightness=0.1,
        color_jitter_contrast=0.1,
        color_jitter_saturation=0.1,
        lighting_std=0.1,
    )

    def __init__(self, seed: int = 0) -> None:
        super(ImageAug, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        for img_name in ["data1", "data2"]:
            image = sampled_data[img_name]["image"]
            image = fb_color_jitter(
                image, self._hyper_params["color_jitter_brightness"],
                self._hyper_params["color_jitter_contrast"],
                self._hyper_params["color_jitter_saturation"])
            image = fb_lighting(image, self._hyper_params["lighting_std"])
            sampled_data[img_name]["image"] = image
        return sampled_data


class NewTransformerBase(TransformerBase):
    def __init__(self, seed: int = 0) -> None:
        super(NewTransformerBase, self).__init__(seed=seed)
        self.rng = self._state["rng"]

    def do_transform(self, image):
        raise NotImplementedError("Use concrete subclasses")

    def __call__(self, sampled_data: Dict) -> Dict:
        for img_name in ["data1", "data2"]:
            if "image" in sampled_data[img_name]:
                sampled_data[img_name]["image"] = self.do_transform(sampled_data[img_name]["image"])
            else:
                for i in range(self._hyper_params["num_memory_frames"]):
                    k = "image_{}".format(i)
                    sampled_data[img_name][k] = self.do_transform(sampled_data[img_name][k])
        return sampled_data


@TRACK_TRANSFORMERS.register
class GrayScaleTransformer(NewTransformerBase):
    default_hyper_params = dict(
        num_memory_frames=0,
        probability=0.5,
    )

    def __init__(self, seed: int = 0) -> None:
        super(GrayScaleTransformer, self).__init__(seed=seed)

    def do_transform(self, image):
        if self.rng.rand() >= self._hyper_params["probability"]:
            return image
        # TODO: check the order of color channels
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = np.stack([img_gray, img_gray, img_gray], axis=2)
        return img_gray


@TRACK_TRANSFORMERS.register
class BrightnessJitterTransformer(NewTransformerBase):
    default_hyper_params = dict(
        num_memory_frames=0,
        brightness_jitter=0.0,
    )

    def __init__(self, seed: int = 0) -> None:
        super(BrightnessJitterTransformer, self).__init__(seed=seed)

    def roll(self):
        brightness_jitter = self._hyper_params['brightness_jitter']
        return self.rng.uniform(max(0, 1 - brightness_jitter), 1 + brightness_jitter)

    def do_transform(self, image):
        return np.clip(image * self.roll(), 0, 255)
