import glob
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils_brats import create_sub_volumes


class ADAM2020(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='/root/workspace/renyan/data', classes=2, crop_dim=(96, 96, 96), split_idx=81,
                 samples=10,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: split how many training,validation
        :param samples: number of sub-volumes that you want to create per case per modility
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + '/ADAM-split/Adam2020_Training/'
        self.testing_path = self.root + '/ADAM-split/Adam2020_Testing/'
        self.full_vol_dim = (140, 512, 512)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)
        self.save_name = self.root + '/ADAM-split/adam2020-list-' + mode + '-samples-' + str(samples) + '.txt'
        logging.info('------Training and testing path are {},{}'.format(self.training_path,self.testing_path))
        logging.info('------Full_vol_dim: {}  Split_idx: {}   Samples: {}'.format(self.full_vol_dim,split_idx,samples))

        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            list_IDsTOF = sorted(glob.glob(os.path.join(self.training_path, '*/pre/TOF.nii.gz')))
            self.affine = img_loader.load_affine_matrix(list_IDsTOF[0])
            logging.info('------Already have pre-generated data, load success')
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/ADAM-split/MICCAI_ADAM_2020_Data_Training/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        list_IDsTOF = sorted(glob.glob(os.path.join(self.training_path, '*/pre/TOF.nii.gz')))
        #list_IDsStruct = sorted(glob.glob(os.path.join(self.training_path, '*/pre/struct.nii.gz')))
        list_IDsStrAli = sorted(glob.glob(os.path.join(self.training_path, '*/pre/struct_aligned.nii.gz')))
        list_Coord = sorted(glob.glob(os.path.join(self.training_path, '*/location.txt')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*/aneurysms.nii.gz')))

        list_IDsTOF, list_IDsStrAli, labels, list_Coord = utils.shuffle_lists(list_IDsTOF,list_IDsStrAli,labels, list_Coord,seed=17)
        logging.info('\n\n=============================Dataset Lists=================================')
        logging.info('list_IDsTOF, list_IDsStrAli, labels, list_Coord:{}\n{}\n{}\n{}'.format(list_IDsTOF, list_IDsStrAli, labels, list_Coord))
        self.affine = img_loader.load_affine_matrix(list_IDsTOF[0])

        if self.mode == 'train':
            print('ADAM2020, Total data:', len(list_IDsTOF))
            list_IDsTOF = list_IDsTOF[:split_idx]
            # list_IDsStruct = list_IDsStruct[:split_idx]
            list_IDsStrAli = list_IDsStrAli[:split_idx]
            list_Coord = list_Coord[:split_idx]
            labels = labels[:split_idx]
            logging.info('\n\n=============================Creating training sub volume=================================')
            self.list = create_sub_volumes(list_IDsTOF, list_IDsStrAli, labels, list_Coord,
                                           dataset_name="adam2020", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

        elif self.mode == 'val':
            list_IDsTOF = list_IDsTOF[split_idx:]
            #list_IDsStruct = list_IDsStruct[split_idx:]
            list_IDsStrAli = list_IDsStrAli[split_idx:]
            list_Coord = list_Coord[split_idx:]
            labels = labels[split_idx:]
            logging.info('=============================Creating val sub volume=================================')
            self.list = create_sub_volumes(list_IDsTOF, list_IDsStrAli, labels, list_Coord,
                                           dataset_name="adam2020", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)
        elif self.mode == 'test':
            self.list_IDsTOF = sorted(glob.glob(os.path.join(self.testing_path, '*/pre/TOF.nii.gz')))
            #self.list_IDsStruct = sorted(glob.glob(os.path.join(self.testing_path, '*/pre/struct.nii.gz')))
            self.list_IDsStrAli = sorted(glob.glob(os.path.join(self.testing_path, '*/pre/struct_aligned.nii.gz')))
            self.labels = None
            # Todo inference code here

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        f_TOF,  f_StrAli,  f_seg = self.list[index]
        img_TOF, img_StrAli, img_seg = np.load(f_TOF),  np.load(f_StrAli), np.load(f_seg)
        if self.mode == 'train' and self.augmentation:
            [img_TOF, img_StrAli], img_seg = self.transform([img_TOF, img_StrAli],
                                                                            img_seg)

            return torch.FloatTensor(img_TOF.copy()).unsqueeze(0),  torch.FloatTensor(img_StrAli.copy()).unsqueeze(0),  torch.FloatTensor(img_seg.copy())

        return torch.FloatTensor(img_TOF.copy()).unsqueeze(0),  torch.FloatTensor(img_StrAli.copy()).unsqueeze(0),  torch.FloatTensor(img_seg.copy())