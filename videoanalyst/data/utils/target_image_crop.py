import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
from videoanalyst.pipeline.utils.bbox import xyxy2xywh, xywh2xyxy


def _transform_box_to_crop(box: np.ndarray, crop_box: np.ndarray, crop_sz: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.copy()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]
    assert scale_factor[0] == scale_factor[1], "The crop image should be square"

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor

    return box_out, scale_factor[0]


def _sample_target_adaptive(im, target_bb: np.ndarray, search_area_factor, output_sz,
                            mode: str = 'replicate',
                            max_scale_change=None,
                            warp_affine=False,
                            avg_chans=None,
                            translation: tuple = (0, 0)):
    if max_scale_change is None:
        max_scale_change = float('inf')
    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)
    # output_sz = torch.Tensor(output_sz)
    output_sz = np.array(output_sz)

    im_h = im.shape[0]
    im_w = im.shape[1]

    bbx, bby, bbw, bbh = target_bb.tolist()

    # Crop image
    crop_sz_x, crop_sz_y = np.ceil(output_sz * np.sqrt(target_bb[2:].prod() / output_sz.prod()) * search_area_factor)\
        .astype(np.int).tolist()

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        # Calculate rescaling factor if outside the image
        rescale_factor = [crop_sz_x / im_w, crop_sz_y / im_h]
        if mode == 'inside':
            rescale_factor = max(rescale_factor)
        elif mode == 'inside_major':
            rescale_factor = min(rescale_factor)
        rescale_factor = min(max(1, rescale_factor), max_scale_change)

        crop_sz_x = math.floor(crop_sz_x / rescale_factor)
        crop_sz_y = math.floor(crop_sz_y / rescale_factor)

    if crop_sz_x < 1 or crop_sz_y < 1:
        raise Exception('Too small bounding box.')

    x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5 + translation[0])
    x2 = x1 + crop_sz_x

    y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5 + translation[1])
    y2 = y1 + crop_sz_y

    # Move box inside image
    shift_x = max(0, -x1) + min(0, im_w - x2)
    x1 += shift_x
    x2 += shift_x

    shift_y = max(0, -y1) + min(0, im_h - y2)
    y1 += shift_y
    y2 += shift_y

    out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
    out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
    shift_x = (-x1 - out_x) * (out_x > 0)
    shift_y = (-y1 - out_y) * (out_y > 0)

    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    # crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])
    crop_box = np.array([x1, y1, x2 - x1, y2 - y1])

    if avg_chans is None:
        avg_chans = (np.mean(im[..., 0]), np.mean(im[..., 1]), np.mean(im[..., 2]))
    if warp_affine:
        crop_xyxy = np.array([x1, y1, x2, y2])
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
        im_out = cv2.warpAffine(im,
                                mat2x3, (output_sz[0], output_sz[1]),
                                flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=tuple(map(int, avg_chans)))
    else:
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im_w + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im_h + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_REPLICATE)
        # Resize image
        im_out = cv2.resize(im_crop_padded, tuple(output_sz.long().tolist()))

    bbox_out, scale_factor = _transform_box_to_crop(target_bb, crop_box, output_sz)

    return im_out, bbox_out, scale_factor


def crop(im, target_bb: np.ndarray, search_area_factor, output_sz,
         avg_chans=None, config: dict = None, rng: np.random = None):
    assert isinstance(im, np.ndarray), 'type(im): {}'.format(type(im))
    target_bb = xyxy2xywh(target_bb)

    # random scale and shift transformation.
    if isinstance(config, dict) and 'phase_mode' in config and config['phase_mode'] == 'train':
        s_max = 1 + config['max_scale']
        s_min = 1 / s_max
        search_area_factor *= rng.uniform(s_min, s_max)

        assert isinstance(output_sz, int), 'type(crop_sz): {}'.format(type(output_sz))
        crop_sz = int(np.ceil(np.sqrt(target_bb[2:].prod()) * search_area_factor))

        max_shift = config['max_shift']
        dx = rng.uniform(-max_shift, max_shift) * crop_sz / 2
        dy = rng.uniform(-max_shift, max_shift) * crop_sz / 2
    else:
        dx = dy = 0

    im_out, bbox_out, scale_factor = _sample_target_adaptive(im, target_bb, search_area_factor, output_sz,
                                                             warp_affine=True,
                                                             avg_chans=avg_chans,
                                                             translation=(dx, dy))
    # bbox_out = bbox_out.detach().cpu().numpy()
    bbox_out = xywh2xyxy(bbox_out)
    scale_factor = scale_factor.item()
    return im_out, bbox_out, scale_factor

