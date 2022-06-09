import nibabel as nib
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage

import glob
import importlib
import logging
import os

import numpy as np
import torch



def binary_mask2bbox(binary_mask):
    assert isinstance(binary_mask, np.ndarray) or isinstance(binary_mask, torch.Tensor)
    if binary_mask.ndim == 5:
        assert binary_mask.shape[1] == 1
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = torch.squeeze(binary_mask, 1)
        else:
            binary_mask = np.squeeze(binary_mask, 1)
    if binary_mask.ndim == 4:
        if isinstance(binary_mask, torch.Tensor):
            return torch.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
        else:
            return np.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
    assert binary_mask.ndim == 3
    if isinstance(binary_mask, torch.Tensor):
        device = binary_mask.device
        binary_mask = binary_mask.detach().cpu().numpy()
    else:
        device = None
    d_mask = (binary_mask.sum((1, 2)) > 0).astype(np.float32)
    d_start = int(np.argmax(d_mask))
    d_end = int(d_start + d_mask.sum() - 1)
    h_mask = (binary_mask.sum((0, 2)) > 0).astype(np.float32)
    h_start = int(np.argmax(h_mask))
    h_end = int(h_start + h_mask.sum() - 1)
    w_mask = (binary_mask.sum((0, 1)) > 0).astype(np.float32)
    w_start = int(np.argmax(w_mask))
    w_end = int(w_start + w_mask.sum() - 1)
    bbox = np.array([d_start, d_end, h_start, h_end, w_start, w_end], np.int32)
    if device is not None:
        bbox = torch.from_numpy(bbox).to(device)
    return bbox


def bbox2binary_mask(bbox, img_shape):
    assert isinstance(bbox, torch.Tensor) or isinstance(bbox, np.ndarray)
    assert len(img_shape) == 3
    if bbox.ndim == 2:
        if isinstance(bbox, torch.Tensor):
            return torch.stack([bbox2binary_mask(bbox[i], img_shape) for i in range(len(bbox))])
        else:
            return np.stack([bbox2binary_mask(bbox[i], img_shape) for i in range(len(bbox))])
    assert bbox.ndim == 1 and len(bbox) == 6
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.type(torch.int32)
        binary_mask = torch.zeros(img_shape, dtype=torch.float32, device=bbox.device)
    else:
        bbox = bbox.astype(np.int32)
        binary_mask = np.zeros(img_shape, np.float32)
    binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1, bbox[4]:bbox[5] + 1] = 1
    return binary_mask



def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor


## todo percentiles

def clip_range(img_numpy):
    """
    Cut off outliers that are related to detected black in the image (the air area)
    """
    # Todo median value!
    zero_value = (img_numpy[0, 0, 0] + img_numpy[-1, 0, 0] + img_numpy[0, -1, 0] + \
                  img_numpy[0, 0, -1] + img_numpy[-1, -1, -1] + img_numpy[-1, -1, 0] \
                  + img_numpy[0, -1, -1] + img_numpy[-1, 0, -1]) / 8.0
    non_zeros_idx = np.where(img_numpy >= zero_value)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    return y


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
