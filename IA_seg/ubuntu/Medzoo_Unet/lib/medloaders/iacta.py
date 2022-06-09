import glob
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import ane_seg_patch_generator,get_dataimg_info
import pandas as pd



def iacta_generate_all_train_patches(config,origin_csv_path, save_dir, subset='all_'):

    pos_neg_ratio = config['data'].get('train_pos_neg_ratio', [1, 1])
    dataimg_info_list_save = os.path.join(save_dir, subset + '_dataimg_info_list.csv')
    patches_save_path = os.path.join(save_dir, 'generated')
    utils.make_dirs(patches_save_path)

    all_npy_save_name = os.path.join(save_dir, subset + '_ratio_' + str(pos_neg_ratio)+'.txt')

    brain_data = utils.load_brain_coords(config['data']['brain_file'])
    brain_dict = utils.get_brain_dict(brain_data)

    # 生成每个case的信息
    df1 = pd.read_csv(origin_csv_path)
    train_csv = df1[df1['subset'] == 'train']
    train_list = np.array(train_csv).tolist()
    assert len(train_list) == config['data']['split_train']

    name_attribute = ['id', 'cta', 'seg', 'original_spacing','original_size', 'origin', 'direction', 'slices', 'IA_voxels','brain']
    dataimg_info_list = get_dataimg_info(config,train_list,brain_dict,name_attribute)

    writerCSV = pd.DataFrame(columns=name_attribute, data=dataimg_info_list)
    writerCSV.to_csv(dataimg_info_list_save, encoding='utf-8')

    npy_filepaths_list = []
    # 对每一个patient进行处理
    for j, item in enumerate(dataimg_info_list):
        print('processing ', item, '......')
        npy_fp_list = ane_seg_patch_generator(dataimg_info_list[j], config,patches_save_path, pos_neg_ratio=pos_neg_ratio, sliding_window=False,balance_label=True, data_aug=True)

        for filepath in npy_fp_list:
            assert len(filepath) == 2
            npy_filepaths_list.append(filepath)

    utils.save_list(all_npy_save_name, npy_filepaths_list)
    return npy_filepaths_list,all_npy_save_name

class IACTA(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, fold, mode):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param split_idx: LIST, ['eval', '316888', nan, 'cta_img/316888_cta.nii.gz', 'ane_seg/316888_seg.nii.gz']
        :param samples: number of sub-volumes that you want to create
        """

        self.subset = mode
        self.config = config
        self.fold = fold
        self.save_name = os.path.join(config['data'][self.fold]['save_path'], self.fold +'_'+ mode + '_ratio_[1, 1].txt')

        self.augmentation = args.augmentation
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)

        self.npy_filepaths_list = utils.load_list(self.save_name)


    def __len__(self):
        return len(self.npy_filepaths_list)

    def __getitem__(self, index):
       f_1, f_seg = self.npy_filepaths_list[index]
       img_1, img_seg = np.load(f_1),  np.load(f_seg)
       if self.subset == 'train' and self.augmentation:
           [img_1], img_seg = self.transform([img_1], img_seg )
           return torch.FloatTensor(img_1.copy()).unsqueeze(0),torch.FloatTensor(img_seg.copy())

       return torch.FloatTensor(img_1.copy()).unsqueeze(0), torch.FloatTensor(img_seg.copy())





class IACTA_TESTDATA(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, test_list, mode):

        self.subset = mode
        self.config = config
        self.output_path = args.save
        self.dataimg_info_list = []

        brain_data = utils.load_brain_coords(config['data']['brain_file'])
        brain_dict = utils.get_brain_dict(brain_data)

        dataimg_info_list_save = os.path.join(self.output_path, self.subset + '_dataimg_info_list.csv')

        assert len(test_list) == config['data']['split_test']
        name_attribute = ['id', 'cta', 'seg', 'original_spacing', 'original_size', 'origin', 'direction', 'slices','IA_voxels', 'brain']
        self.dataimg_info_list = get_dataimg_info(config, test_list, brain_dict, name_attribute)

        writerCSV = pd.DataFrame(columns=name_attribute, data=self.dataimg_info_list)
        writerCSV.to_csv(dataimg_info_list_save, encoding='utf-8')


    def __len__(self):
        return len(self.dataimg_info_list)

    def __getitem__(self, index):
        dataimg_info = self.dataimg_info_list[index]

        return dataimg_info
