from datetime import datetime
import random

import torch
from torch.nn.functional import affine_grid, grid_sample


__all__ = ['xywh_to_xyxy', 'calculate_area', 'calculate_iou', 'repeat_loader', 'get_time_id',
           'mrcn_crop_pool_layer', 'jitter_roi', 'recursive_jitter_roi', 'alert_print']


def xywh_to_xyxy(box):
    """Convert xywh bbox to xyxy format."""
    return box[0], box[1], box[0]+box[2], box[1]+box[3]


def calculate_area(box):
    """Calculate area of bbox in xyxy format."""
    return (box[2] - box[0])*(box[3] - box[1])


def calculate_iou(box1, box2):
    """Calculate IoU of two bboxes in xyxy format."""
    max_L = max(box1[0], box2[0])
    min_R = min(box1[2], box2[2])
    max_T = max(box1[1], box2[1])
    min_B = min(box1[3], box2[3])
    if max_L < min_R and max_T < min_B:
        intersection = (min_B - max_T)*(min_R - max_L)
        union = calculate_area(box1) + calculate_area(box2) - intersection
        return intersection / union
    else:
        return 0.


def repeat_loader(loader):
    """Endlessly repeat given loader."""
    while True:
        for data in loader:
            yield data


def get_time_id():
    tt = datetime.now().timetuple()
    return '{:02d}{:02d}{:02d}{:02d}{:02d}'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)


def mrcn_crop_pool_layer(bottom, rois):
    x1 = rois[:, 0::4] / 16.0  # [batch_size, 1]
    y1 = rois[:, 1::4] / 16.0
    x2 = rois[:, 2::4] / 16.0
    y2 = rois[:, 3::4] / 16.0
    height = bottom.size(2)
    width = bottom.size(3)
    # Affine theta
    zero = torch.zeros_like(x1)
    theta = torch.cat([
        (x2 - x1)/(width - 1),
        zero,
        (x1 + x2 - width + 1)/(width - 1),
        zero,
        (y2 - y1)/(height - 1),
        (y1 + y2 - height + 1)/(height - 1)], 1).reshape(-1, 2, 3)  # [batch_size, 2, 3]
    if int(torch.__version__.split('.')[1]) < 3:
        grid = affine_grid(theta, torch.Size((rois.size(0), 1, 7, 7)))
        crops = grid_sample(bottom.expand(rois.size(0), -1, -1, -1), grid)
    else:
        grid = affine_grid(theta, torch.Size((rois.size(0), 1, 7, 7)), align_corners=True)
        crops = grid_sample(bottom.expand(rois.size(0), -1, -1, -1), grid, align_corners=True)
    return crops


def jitter_coordinate(x, L, w, image_l, image_r):
    r = w * (1 - L) / L
    l = w * (L - 1)

    rec = x + random.uniform(l, r)

    rec = max(rec, image_l)
    rec = min(rec, image_r)

    return rec


def jitter_roi(G, L, R, img_w, img_h):
    w = G[2] - G[0]
    h = G[3] - G[1]

    while True:
        x0 = jitter_coordinate(G[0], L, w, 0, img_w)
        x1 = jitter_coordinate(G[2], L, w, 0, img_w)
        y0 = jitter_coordinate(G[1], L, h, 0, img_h)
        y1 = jitter_coordinate(G[3], L, h, 0, img_h)
        jittered_roi = (x0, y0, x1, y1)
        if L <= calculate_iou(jittered_roi, G) <= R:
            return jittered_roi


def recursive_jitter_roi(G, L, R, img_w, img_h, max_interval=0.01):
    assert L < R
    if R - L > max_interval:
        mid = (L + R) / 2
        if random.random() < 0.5:
            return recursive_jitter_roi(G, L, mid, img_w, img_h, max_interval)
        else:
            return recursive_jitter_roi(G, mid, R, img_w, img_h, max_interval)
    else:
        w = G[2] - G[0]
        h = G[3] - G[1]
        while True:
            x0 = jitter_coordinate(G[0], L, w, 0, img_w)
            x1 = jitter_coordinate(G[2], L, w, 0, img_w)
            y0 = jitter_coordinate(G[1], L, h, 0, img_h)
            y1 = jitter_coordinate(G[3], L, h, 0, img_h)
            jittered_roi = (x0, y0, x1, y1)
            if L <= calculate_iou(jittered_roi, G) <= R:
                return jittered_roi


def alert_print(msg):
    print('\33[31m[ALERT] {}\33[0m'.format(msg))
