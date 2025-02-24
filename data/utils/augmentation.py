import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from typing import List

import cv2
import numpy as np
import numba
import torch_geometric.transforms as T


def _scale_and_clip(x, scale):
    return int(torch.clamp(x * scale, min=0, max=scale-1))

def _crop_events(events, left, right, not_crop_idx=None):
    if not_crop_idx is None:
        not_crop_idx = torch.all((events[:,:2] >= left) & (events[:,:2] <= right), dim=1)

    events = events[not_crop_idx]
    return events

@numba.njit
def _add_event(x, y, xlim, ylim, p, i, count, pos, mask, threshold=1):
    count[ylim, xlim] += float(p * (1 - abs(x - xlim)) * (1 - abs(y - ylim)))
    pol = 1 if count[ylim, xlim] > 0 else -1

    if pol * count[ylim, xlim] > threshold:
        count[ylim, xlim] -= pol * threshold

        mask[i] = True
        pos[i, 0] = xlim
        pos[i, 1] = ylim


@numba.njit
def _subsample(pos: np.ndarray, polarity: np.ndarray, mask: np.ndarray, count: np.ndarray, threshold=1):
    for i in range(len(pos)):
        x, y = pos[i]
        x0, x1 = int(x), int(x+1)
        y0, y1 = int(y), int(y+1)

        _add_event(x, y, x0, y0, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x1, y0, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x0, y1, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)
        _add_event(x, y, x1, y1, polarity[i,0], i=i, count=count, pos=pos, mask=mask, threshold=threshold)

class RandomHFlip:
    def __init__(self, cfg):
        cfg_flip = cfg.augmentation.h_flip
        self.p = cfg_flip.p

        self.width, self.height  = cfg.representation[cfg.representation.type].dim

    def __call__(self, events: torch.tensor) -> torch.tensor:
        if torch.rand(1) < self.p:
            events[:, 0] = self.width - events[:, 0] - 1
        return events

class RandomCrop:
    def __init__(self, cfg):
        cfg_crop = cfg.augmentation.random_crop

        width, height  = cfg.representation[cfg.representation.type].dim
        self.p = cfg_crop.p
        self.size = torch.as_tensor(cfg_crop.size)
        self.dim = cfg_crop.dim

        size = torch.IntTensor([width, height])
        self.size = torch.IntTensor([_scale_and_clip(s, ss) for s, ss in zip(self.size, size)])
        self.left_max = size - self.size


    def __call__(self, events: torch.tensor) -> torch.tensor:
        if torch.rand(1) > self.p:
            return events

        left = (torch.rand(len(self.dim)) * self.left_max).to(torch.int16)
        right = left + self.size

        events = _crop_events(events, left, right)
        return events

class RandomZoom:
    def __init__(self, cfg):
        cfg_zoom = cfg.augmentation.random_zoom
        self.zoom = cfg_zoom.zoom
        self.subsample = cfg_zoom.subsample

        self.width, self.height  = cfg.representation[cfg.representation.type].dim

        if self.subsample:
            self._count = None

    def _subsample(self, events, zoom, count):
        pos_zoom = events[:, :2].numpy()

        mask = np.zeros(len(events[:, :2]), dtype="bool")
        _subsample(pos_zoom, events[:, 2].numpy(), mask, count, threshold=1/(float(zoom)**2))

        events[:, :2] = torch.from_numpy(pos_zoom[mask].astype("int16")) # implicit cast to int
        events[:, 3] = events[:, 3][mask]
        events[:, 2] = events[:, 2][mask]

        return events

    def init(self, height, width):
        self._count = np.zeros((height + 1, width + 1), dtype="float32")

    def __call__(self, events):
        zoom = torch.rand(1) * (self.zoom[1] - self.zoom[0]) + self.zoom[0]
        H, W = self.height, self.width

        events[:, 0] = ((events[:, 0] - W // 2) * zoom + W // 2).to(torch.int16)
        events[:, 1] = ((events[:, 1] - H // 2) * zoom + H // 2).to(torch.int16)

        if self.subsample and zoom < 1:
            events = self._subsample(events, float(zoom), count=self._count.copy())

        return events

class RandomTranslate:
    def __init__(self, cfg):
        cfg_translate = cfg.augmentation.translate
        self.size = torch.as_tensor(cfg_translate.size).float()

        width, height  = cfg.representation[cfg.representation.type].dim

        size = [width, height]
        self.size = torch.IntTensor([_scale_and_clip(s, ss) for s, ss in zip(self.size, size)])

    def __call__(self, events):
        move_px = (self.size * (torch.rand(len(self.size)) * 2 - 1)).to(torch.int16)
        events[:,:2] = events[:,:2] + move_px

        return events

class Crop:
    def __init__(self, cfg):
        cfg_crop = cfg.augmentation.crop

        self.min = torch.as_tensor(cfg_crop.min)
        self.max = torch.as_tensor(cfg_crop.max)

        width, height  = cfg.representation[cfg.representation.type].dim

        size = [width, height]
        self.max = torch.IntTensor([_scale_and_clip(m, s) for m, s in zip(self.max, size)])
        self.min = torch.IntTensor([_scale_and_clip(m, s) for m, s in zip(self.min, size)])
        
    def __call__(self, events):
        events = _crop_events(events, self.min, self.max)


        return events