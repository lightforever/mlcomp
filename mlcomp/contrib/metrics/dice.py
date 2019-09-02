import numpy as np


def dice_numpy(targets, outputs, threshold=None, min_area=None,
               empty_one: bool = True, eps=1e-6):
    if threshold is not None:
        # noinspection PyUnresolvedReferences
        outputs = (outputs >= threshold).astype(np.uint8)

    targets_sum = targets.sum()
    outputs_sum = outputs.sum()

    if min_area and outputs_sum < min_area:
        outputs = np.zeros(outputs.shape, dtype=np.uint8)
        outputs_sum = 0

    if empty_one and targets_sum == 0 and outputs_sum == 0:
        return 1

    intersection = (targets * outputs).sum()
    union = targets_sum + outputs_sum
    dice = 2 * intersection / (union + eps)
    return dice
