import cv2
import os
import numpy as np
import shutil
import time

SAVE_ROOT = './logs/score_map_visualization/score_maps'


def rename_dir():
    if os.path.exists(SAVE_ROOT):
        shutil.move(SAVE_ROOT, SAVE_ROOT + str(int(time.time())))


def create_dir():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)


def visualize(score, score_size, crop, frame_num, name):
    create_dir()
    score = score.reshape(score_size, score_size)
    score = score[:, :, np.newaxis]
    score = cv2.normalize(score, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    color_map = cv2.applyColorMap(score, cv2.COLORMAP_VIRIDIS)
    dst_size = (crop.shape[1] * 2, crop.shape[0] * 2)
    color_map = cv2.resize(color_map, dst_size, interpolation=cv2.INTER_LINEAR)
    crop = cv2.resize(crop, dst_size, interpolation=cv2.INTER_LINEAR)
    final_img = color_map * 0.5 + crop * 0.5
    cv2.imwrite(os.path.join(SAVE_ROOT, '{}-{:04d}.jpg'.format(name, frame_num)), final_img)
