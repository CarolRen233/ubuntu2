from lib.medloaders import medical_image_process as img_loader

from lib.medloaders.medical_image_process import binary_mask2bbox
from lib.visual3D_temp import *
import logging
import SimpleITK as sitk
from scipy import sparse
import logging
import math
import os
import platform
import random
import threading
import time
from multiprocessing import Array
from queue import Empty

import SimpleITK as sitk
import numpy as np
import skimage.measure as measure
import torch.utils.data
from skimage.transform import resize


def get_dataimg_info(config, data_list, brain_dict, name_attribute):
    '''

    '''
    assert name_attribute == ['id', 'subset', 'cta', 'seg','brain_x_start','brain_y_start','brain_z_start','brain_x_end','brain_y_end','brain_z_end','is_ruptured','institution_id','age','gender','num_IAs', 'spacing1', 'spacing2','spacing3','original_size', 'origin', 'direction', 'slices','IA_voxels']

    dataimg_info_list = []
    # ##为每个文件构建一个字典
    for csv_row in sorted(data_list):

        assert csv_row[0] in ['train','test']

        img = dict()
        img['id'] = csv_row[1]
        img['subset'] = csv_row[0]

        img['cta'] = os.path.join(config['data']['data_dir'],'cta_img',img['id']+'_cta.nii.gz')
        img['seg'] = os.path.join(config['data']['data_dir'],'ane_seg',img['id']+'_seg.nii.gz')

        img_itk = sitk.ReadImage(img['cta'])
        seg_itk = sitk.ReadImage(img['seg'])

        '''
        try:
            x1,y1,z1,x2,y2,z2 = brain_dict[str(img['id'])]
            seg_np = sitk.GetArrayFromImage(seg_itk)
            if seg_np.sum() != seg_np[z1:z2,y1:y2,x1:x2].sum():
                img['brain_x_start'], img['brain_y_start'], img['brain_z_start'], img['brain_x_end'], img[
                    'brain_y_end'], img['brain_z_end'] = -1, -1, -1, -1, -1, -1
            else:
                img['brain_x_start'], img['brain_y_start'], img['brain_z_start'], img['brain_x_end'], img[
                    'brain_y_end'], img['brain_z_end'] = x1, y1, z1, x2, y2, z2
        except:
        '''
        img['brain_x_start'],img['brain_y_start'],img['brain_z_start'],img['brain_x_end'],img['brain_y_end'],img['brain_z_end'] = -1,-1,-1,-1,-1,-1


        img['is_ruptured'] = csv_row[5]
        img['institution_id'] = csv_row[2]
        img['age'] = csv_row[3]
        img['gender'] = csv_row[4]
        img['num_IAs'] = csv_row[6]


        img['spacing1'],img['spacing2'],img['spacing3'] = np.array(img_itk.GetSpacing(), np.float32)[[2, 1, 0]]
        img['original_size'] = np.array(img_itk.GetSize(), np.int32)[[2, 1, 0]]
        img['origin'] = img_itk.GetOrigin()
        img['direction'] = img_itk.GetDirection()

        img['slices'] = sitk.GetArrayFromImage(img_itk).shape[0]
        img['IA_voxels'] = sitk.GetArrayFromImage(seg_itk).sum()

        dataimg_info_list.append(img)

    return dataimg_info_list


#进来一个patient，输出patch
def ane_seg_patch_generator(dataimg, config, save_path, pos_neg_ratio=(1, 1), sliding_window=False, balance_label=True, data_aug=False, random_seed=None):
    """
    yield patches of aneurysm segmentation
    :param data: images dict
    :param config: config dict
    :param sliding_window: false to random select negative samples
    :param balance_label: if true, repeat positive samples to balance labels.
    :param data_aug: useful for training
    :param pos_neg_ratio: only work if balance_label is true
    :return: input_patch, label_patch
    """

    list = []  # 保存不同模态的patch和seg

    itk_image = sitk.ReadImage(dataimg['cta'])
    itk_seg = sitk.ReadImage(dataimg['seg'])

    input_glo_img = sitk.GetArrayFromImage(itk_image).astype(np.float32)
    label_glo_img  = sitk.GetArrayFromImage(itk_seg).astype(np.int32)
    if dataimg['brain_z_start'] != -1:
        brain_mask_glo_img = np.zeros(label_glo_img.shape, np.int32)
        brain_mask_glo_img[dataimg['brain_z_start']:dataimg['brain_z_end'],dataimg['brain_y_start']:dataimg['brain_y_end'],dataimg['brain_x_start']:dataimg['brain_x_end']]=1
    else:
        brain_mask_glo_img = np.ones(label_glo_img.shape, np.int32)
    patch_size = config['data']['patch_size']
    overlap_step = config['data']['overlap_step']
    assert len(patch_size) == len(overlap_step)

    if label_glo_img.shape != input_glo_img.shape or label_glo_img.shape != brain_mask_glo_img.shape:
        logging.warning(
            'Subject %s has different shapes among cta_img, brain_mask_img and aneyrysm_seg_img' % data['id'])
        return None
    if any([label_glo_img.shape[i] < patch_size[i] for i in range(3)]):
        logging.warning('Subject %s is too small and cannot fit in one patch.' % dataimg['id'])
        return None

    # ##TODO generate patch

    def _gen_patch(_starts,pos_or_neg):

        _ends = [_starts[i] + patch_size[i] for i in range(3)]
        patch_cta_img = input_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        patch_brain_mask_img = brain_mask_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        patch_label_img = label_glo_img[_starts[0]:_ends[0], _starts[1]:_ends[1], _starts[2]:_ends[2]].copy()
        if patch_cta_img.shape != patch_brain_mask_img.shape or patch_cta_img.shape != patch_label_img.shape:
            logging.warning('Different shapes for patch_cta_img, patch_brain_mask_img and patch_label_img: %s, %s, %s'
                           % (patch_cta_img.shape, patch_brain_mask_img.shape, patch_label_img.shape))
            return None
        if data_aug:
            patch_cta_img += np.random.normal(0.0, 1.0, patch_cta_img.shape)
            all_arrays = [patch_cta_img, patch_brain_mask_img, patch_label_img]
            bundle = np.stack(all_arrays)
            ran_1 = [np.random.rand() > 0.5 for _ in range(3)]
            bundle = random_flip_all(bundle, ran_1)

            patch_cta_img, patch_brain_mask_img, patch_label_img = np.split(bundle, 3)
            patch_cta_img = np.squeeze(patch_cta_img.copy(), 0)
            patch_brain_mask_img = np.squeeze(patch_brain_mask_img.copy(), 0)
            patch_label_img = np.squeeze(patch_label_img.copy(), 0)

        patch_cta_img = torch.from_numpy(patch_cta_img)
        patch_label_img = torch.from_numpy(patch_label_img)

        filename = save_path + '/' + str(dataimg["id"]) + '_' + pos_or_neg + '_' + str(_starts[0]) + '_' + str(_starts[1]) + '_' + str(_starts[2]) + '.npz'


        np.savez_compressed(filename, cta=patch_cta_img, seg=patch_label_img)

        return filename

    if not sliding_window:
        # compute patches number (50-300 samples per study)
        sum_brain_mask_number = 1 * np.sum(brain_mask_glo_img) // (patch_size[0] * patch_size[1] * patch_size[2])
        logging.info('number of patches generated in %s is %s' % (dataimg['id'], sum_brain_mask_number))

        pos_region_centers = get_positive_region_centers(label_glo_img)
        num_pos_region = len(pos_region_centers)
        count = 0
        index_pos = 0
        random_shakes = [int(patch_size[i] * 0.3) for i in range(3)]
        if random_seed is not None:
            np.random.seed(random_seed)

        all_patch_starts_pos, all_patch_starts_neg = [],[]
        while count < sum_brain_mask_number:
            # positive sample
            for _ in range(pos_neg_ratio[0]):
                print (' positive sample:',_)
                if balance_label and index_pos < num_pos_region:
                    starts = [min(
                        max(0, int(pos_region_centers[index_pos][i]) - patch_size[i] // 2
                            + np.random.randint(1 - random_shakes[i], random_shakes[i]))
                        , label_glo_img.shape[i] - patch_size[i]) for i in range(3)]
                    pos_or_neg = 'pos'
                    patch_list = _gen_patch(starts, pos_or_neg)
                    all_patch_starts_pos.append(starts)
                    list.append(patch_list)
                    count += 1
                    print ('pos starts:',starts)
                    print ('pos count:',count)
                    index_pos = (index_pos + 1) % num_pos_region
                else:
                    index_pos += 1
            # negative sample
            for _ in range(pos_neg_ratio[1]):
                patch_found = False  # only yield samples whose centers hit the reference_mask
                while not patch_found:
                    starts = [np.random.randint(0, brain_mask_glo_img.shape[i] - patch_size[i] + 1) for i in range(3)]
                    all_patch_starts_neg.append(starts)
                    if brain_mask_glo_img[
                        starts[0] + patch_size[0] // 2, starts[1] + patch_size[1] // 2, starts[2] + patch_size[
                            2] // 2] > 0:
                        # avoid inputing all black imgs
                        clipped_mask = (input_glo_img[starts[0]:starts[0] + patch_size[0],
                                        starts[1]:starts[1] + patch_size[1],
                                        starts[2]:starts[2] + patch_size[2]] > 0).astype(np.float32)
                        if np.mean(clipped_mask) > 0.05:
                            patch_found = True
                            pos_or_neg = 'neg'
                            patch_path =_gen_patch(starts,pos_or_neg)
                            print ('neg starts',starts)
                            list.append(patch_path)
                            count += 1
                            print ('neg count:',count)
                        else:
                            logging.debug('all black inputs')

    logging.info('generate patches num:{}'.format(len(list)))
    logging.info('pos patch starts: \n{}\n neg patch starts:\n{}'.format(all_patch_starts_pos, all_patch_starts_neg))

    return list

def get_sliding_window_patch_starts(input_img: np.ndarray, patch_size, overlap_step, reference_mask=None):
    assert input_img.ndim == 3
    d, h, w = input_img.shape
    d_starts = list(range(0, d - patch_size[0], overlap_step[0])) + [d - patch_size[0]]
    h_starts = list(range(0, h - patch_size[1], overlap_step[1])) + [h - patch_size[1]]
    w_starts = list(range(0, w - patch_size[2], overlap_step[2])) + [w - patch_size[2]]

    patch_starts = []
    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                if reference_mask is None:
                    patch_starts.append([ds, hs, ws])
                else:
                    if reference_mask[ds:ds + patch_size[0], hs:hs + patch_size[1], ws:ws + patch_size[2]].sum() \
                            > 0.05 * patch_size[0] * patch_size[1] * patch_size[2]:
                        patch_starts.append([ds, ws, hs])  # only compute patches who have more than 5% reference mask
    return patch_starts


def get_positive_region_centers(label, return_object_wise_label=False):
    label = measure.label(label)
    pros = measure.regionprops(label)
    centers = [c.centroid for c in pros if c.area > 5]  # ignore small noise region
    if return_object_wise_label:
        return centers, label
    else:
        return centers


def resize_image(image, old_spacing=None, new_spacing=None, new_shape=None, order=1):
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple([int(np.round(old_spacing[i] / new_spacing[i] * float(image.shape[i]))) for i in range(3)])
    resized_image = resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)
    return resized_image


def resize_segmentation(segmentation, old_spacing=None, new_spacing=None, new_shape=None, order=0, cval=0):
    '''
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple(
            [int(np.round(old_spacing[i] / new_spacing[i] * float(segmentation.shape[i]))) for i in range(3)])
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True,
                      anti_aliasing=False).astype(tpe)
    else:
        unique_labels = np.unique(segmentation)
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True,
                                       anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


class GlobalLocalizer:
    def __init__(self, reference_mask):
        assert reference_mask.ndim == 3
        self.original_shape = reference_mask.shape
        self.mask = reference_mask
        starts = [0, 0, 0]
        ends = list(self.mask.shape)
        while starts[0] < self.original_shape[0]:
            if self.mask[starts[0], :, :].sum() > 0:
                break
            starts[0] += 1
        while ends[0] > starts[0]:
            if self.mask[ends[0] - 1, :, :].sum() > 0:
                break
            ends[0] -= 1
        while starts[1] < self.original_shape[1]:
            if self.mask[:, starts[1], :].sum() > 0:
                break
            starts[1] += 1
        while ends[1] > starts[1]:
            if self.mask[:, ends[1] - 1, :].sum() > 0:
                break
            ends[1] -= 1
        while starts[2] < self.original_shape[2]:
            if self.mask[:, :, starts[2]].sum() > 0:
                break
            starts[2] += 1
        while ends[2] > starts[2]:
            if self.mask[:, :, ends[2] - 1].sum() > 0:
                break
            ends[2] -= 1

        self.starts = starts
        self.ends = ends

    def cut_edge(self, img, new_shape=None, is_mask=False):
        assert img.shape == self.original_shape
        cut_img = img[self.starts[0]:self.ends[0], self.starts[1]:self.ends[1], self.starts[2]:self.ends[2]].copy()
        if new_shape is not None:
            cut_img = self.reshape_keep_ratio(cut_img, new_shape, is_mask)
        return cut_img

    def get_cut_reference_mask(self, new_shape=None):
        return self.cut_edge(self.mask, new_shape, is_mask=True)

    def reshape_keep_ratio(self, img, new_shape, is_mask=False):
        assert len(new_shape) == 3
        ori_shape = img.shape
        rel_index = np.argmin(np.array([new_shape[i] / ori_shape[i] for i in range(3)]))
        pad_shape = [round(ori_shape[rel_index] * new_shape[i] / new_shape[rel_index]) for i in range(3)]
        padded_img = np.pad(img, tuple([((pad_shape[i] - ori_shape[i]) // 2,) for i in range(3)]),
                            mode='constant', constant_values=img.min())
        if is_mask:
            new_img = resize_segmentation(padded_img, new_shape=new_shape)
        else:
            new_img = resize_image(padded_img, new_shape=new_shape)
        return new_img

    def get_position_map(self, starts, ends, new_shape=None):
        position_map = np.zeros(self.original_shape, np.float32)
        position_map[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] = 1
        cut_position_map = self.cut_edge(position_map)
        if new_shape is not None:
            cut_position_map = self.reshape_keep_ratio(cut_position_map, new_shape, is_mask=True)
        return cut_position_map

    def get_position_bbox(self, starts, ends, new_shape=None):
        if new_shape is None:
            new_shape = self.original_shape
        reference_index = np.argmin(np.array([new_shape[i] / self.original_shape[i] for i in range(3)]))
        new_starts = [min(max(0, round((
                                               2 * starts[i] * new_shape[reference_index] + new_shape[i] *
                                               self.original_shape[reference_index] -
                                               self.original_shape[i] * new_shape[reference_index]) / (
                                                   2 * self.original_shape[reference_index]))), new_shape[i] - 1) for i
                      in
                      range(3)]
        new_ends = [min(max(new_starts[i], round((
                                                         2 * ends[i] * new_shape[reference_index] + new_shape[i] *
                                                         self.original_shape[reference_index] -
                                                         self.original_shape[i] * new_shape[
                                                             reference_index]) / (
                                                         2 * self.original_shape[reference_index])) - 1),
                        new_shape[i] - 1) for i in
                    range(3)]
        new_bbox = np.array([new_starts[0], new_ends[0], new_starts[1], new_ends[1], new_starts[2], new_ends[2]])
        return new_bbox


def random_flip_all(img, do_it=(None, None, None)):
    img = random_flip(img, 1, do_it[0])
    img = random_flip(img, 2, do_it[1])
    img = random_flip(img, 3, do_it[2])
    return img


def random_rotate_all(img, do_it=(None, None, None)):
    img = random_rotate(img, 1, do_it[0])
    img = random_rotate(img, 2, do_it[1])
    img = random_rotate(img, 3, do_it[2])
    return img


def random_flip(img, dim, do_it=None):
    assert len(img.shape) == 4  # c, d, w, h
    assert dim in [1, 2, 3]
    norm_img = img

    if do_it is None:
        if np.random.rand() > 0.5:
            do_it = False
        else:
            do_it = True
    if do_it:
        out_img = np.flip(norm_img, [dim])
    else:
        out_img = norm_img
    return out_img


def random_rotate(img, dim, do_it=None):
    assert len(img.shape) == 4  # c, d, w, h
    assert dim in [1, 2, 3]

    norm_img = img

    if dim == 1:
        perm = [0, 1, 3, 2]
    elif dim == 2:
        perm = [0, 3, 2, 1]
    else:
        perm = [0, 2, 1, 3]

    if do_it is None:
        if np.random.rand() > 0.5:
            do_it = True
        else:
            do_it = False
    if do_it:
        out_img = np.transpose(norm_img, perm)
    else:
        out_img = norm_img
    return out_img

