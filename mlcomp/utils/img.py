from typing import Tuple

import numpy as np
import cv2


def resize_saving_ratio(img: np.array, size: Tuple[int, int]):
    if not size:
        return img
    if size[0] and img.shape[0] > size[0]:
        k = size[0] / img.shape[0]
        img = cv2.resize(img, (int(k * img.shape[1]), size[0]))
    if size[1] and img.shape[1] > size[1]:
        k = size[1] / img.shape[1]
        img = cv2.resize(img, (size[1], int(k * img.shape[0])))
    return img


__all__ = ['resize_saving_ratio']
