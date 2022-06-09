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
import pprint, pickle


def cellpress_generate_all_train_patches(config,train_df, save_dir, subset='all_'):

    pos_neg_ratio = config['data'].get('train_pos_neg_ratio', [1, 1])

    # patch save
    patches_save_path = os.path.join(save_dir,'generated')
    utils.make_dirs(patches_save_path)
    # list
    all_npy_save_name = os.path.join(save_dir, subset + config['data']['dataset'] + str(config['data']['data_num']) + '_ratio_' + str(pos_neg_ratio)+'.txt')


    npz_filepaths_list = []
    # 对每一个incetance提取patch
    for j in range(len(train_df)):
        iD = train_df.iloc[j]["id"]
        print('processing ', iD, '......')
        logging.info('--------------------{}---------------------'.format(iD))
        npz_fp_list = ane_seg_patch_generator(train_df.iloc[j], config,patches_save_path, pos_neg_ratio=pos_neg_ratio, sliding_window=False,balance_label=True, data_aug=False)

        for npz_path in npz_fp_list:
            npz_filepaths_list.append(npz_path)

    utils.save_list(all_npy_save_name, npz_filepaths_list)
    return npz_filepaths_list,all_npy_save_name

class CELLPRESS(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, mode):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param split_idx: LIST, ['eval', '316888', nan, 'cta_img/316888_cta.nii.gz', 'ane_seg/316888_seg.nii.gz']
        :param samples: number of sub-volumes that you want to create
        """

        self.subset = mode
        self.config = config
        self.fold = 'fold_' + str(args.fold)
        self.npz_data_file = os.path.join(os.path.dirname(args.save), self.fold + '_'+ mode + config['data']['dataset'] + str(config['data']['data_num']) + '_ratio_' + str(config['data']['train_pos_neg_ratio']) +'.txt')


        self.augmentation = args.augmentation
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)

        self.npz_filepaths_list = utils.load_list(self.npz_data_file)


    def __len__(self):
        return len(self.npz_filepaths_list)

    def __getitem__(self, index):
        patch_file = self.npz_filepaths_list[index]
        img_0, img_seg = np.load(patch_file)['cta'],np.load(patch_file)['seg']

        img_list = []
        hu_intervals = self.config['data']['hu_values']
        img_0_tensor = torch.from_numpy(img_0)
        for j in range(len(hu_intervals)):
            hu_channel = torch.clamp(img_0_tensor, hu_intervals[j][0], hu_intervals[j][1])
            tensor_image = (hu_channel - hu_intervals[j][0]) / (hu_intervals[j][1] - hu_intervals[j][0])
            img_list.append(tensor_image.numpy())
        img_1,img_2,img_3 = img_list

        if self.subset == 'train' and self.augmentation:
            [img_1,img_2,img_3], img_seg = self.transform([img_1,img_2,img_3], img_seg )

            return torch.FloatTensor(img_1.copy()).unsqueeze(0),torch.FloatTensor(img_2.copy()).unsqueeze(0),torch.FloatTensor(img_3.copy()).unsqueeze(0),torch.FloatTensor(img_seg.copy())

        return torch.FloatTensor(img_1.copy()).unsqueeze(0),torch.FloatTensor(img_2.copy()).unsqueeze(0),torch.FloatTensor(img_3.copy()).unsqueeze(0),torch.FloatTensor(img_seg.copy())


class Radiology_TEST(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, mode):

        self.subset = mode
        self.config = config
        self.output_path = args.save

        pkl_file = open(config['inference']['pkl_info_path'], 'rb')
        data1 = pickle.load(pkl_file)
        pkl_file.close()
        
        self.test_pickle = data1
        self.keys_list = [name for name in data1.keys()]
        
        try:
            self.test_pickle.pop('ExtA0009')
            print (self.keys_list[8])
            self.keys_list.pop(8)
            self.test_pickle.pop('ExtA0032')
            print (self.keys_list[30])
            self.keys_list.pop(30)
            
            
        except:
            print ('not a')

    def __len__(self):
        return len(self.test_pickle)

    def __getitem__(self, index):
        dataimg_info = self.test_pickle[self.keys_list[index]]

        return dataimg_info,self.keys_list[index]

    
class XJTsN_TEST(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, mode):

        self.subset = mode
        self.config = config
        self.output_path = args.save

        pkl_file = open(config['inference']['pkl_info_path'], 'rb')
        data1 = pickle.load(pkl_file)
        pkl_file.close()
        
        self.test_pickle = data1
        self.keys_list = [name for name in data1.keys()]
        print ('self.keys_list',len(self.keys_list))
        print ('self.keys_list:',self.keys_list[:5])
        

    def __len__(self):
        return len(self.test_pickle)

    def __getitem__(self, index):
        dataimg_info = self.test_pickle[self.keys_list[index]]

        return dataimg_info,self.keys_list[index]
    
class CELLPRESS_TEST(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, config, mode):

        self.subset = mode
        self.config = config
        self.output_path = args.save
        self.dataimg_info_list = []

        fold = 'fold_' + str(args.fold)
        df1 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(args.save)), config['data'][fold]['save_path']))
        self.test_csv = df1[df1['subset'] == 'test']
        
        


    def __len__(self):
        return len(self.test_csv)

    def __getitem__(self, index):
        dataimg_info = self.test_csv.iloc[index]

        return dataimg_info