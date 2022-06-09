from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import SimpleITK as sitk
import cv2
import os, glob
from skimage.transform import resize
import imageio
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import torch



class Dataset(BaseDataset):
    CLASSES = ['backgound', 'brain']

    def __init__(
            self,
            ids,
            images_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        ##print('self.class_values:', self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data

        #print('is bone data exits?', os.path.exists(self.images_fps[i]))
        #bone_cta_nii = sitk.ReadImage(self.images_fps[i])
        #bone_cta_np = sitk.GetArrayFromImage(bone_cta_nii).astype(np.float32)

        #bone_image = abs(bone_cta_np)
        #bone_image = (bone_image - np.min(bone_image)) / (np.max(bone_image) - np.min(bone_image))

        #image_MIP_2 = bone_mip = np.max(bone_image, axis=2)
        # print ('image_MIP_2:',np.unique(image_MIP_2))


        image_1 = np.load(self.images_fps[i] + '_bone_mip1.npz')['bone_mip1']
        image_2 = np.load(self.images_fps[i] + '_bone_mip2.npz')['bone_mip2']

        shape1 = image_1.shape
        shape2 = image_2.shape
        name = os.path.basename(self.images_fps[i])
        image_1 = cv2.resize(image_1, dsize= (512, 512), interpolation=cv2.INTER_NEAREST)
        image_2 = cv2.resize(image_2, dsize= (512, 512), interpolation=cv2.INTER_NEAREST)

        return shape1,shape2, name, torch.FloatTensor(np.tile(image_1.copy(), (3, 1, 1))),torch.FloatTensor(np.tile(image_2.copy(), (3, 1, 1)))

    def __len__(self):
        return len(self.ids)
    
    
class TrainDataset(BaseDataset):
    CLASSES = ['backgound', 'brain']

    def __init__(
            self,
            ids,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.ids = ids

        self.masks_fps = [os.path.join(masks_dir, os.path.basename(image_id)) for image_id in self.ids]
        self.images_fps = [os.path.join(images_dir, os.path.basename(image_id).replace('mask','bone')) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, os.path.basename(image_id).replace('bone','mask')) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.npz_mask_key = [os.path.basename(image_id)[-13:-4] for image_id in self.ids]
        self.npz_image_key = [os.path.basename(image_id).replace('mask','bone')[-13:-4] for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        

        image = np.load(self.images_fps[i])[self.npz_image_key[i]]
        mask = np.load(self.masks_fps[i])[self.npz_mask_key[i]]
        

        #print ('image shape:',image.shape)
        #print('mask shape:', mask.shape)
        assert image.shape == mask.shape

        image = np.resize(image, (512, 512))
        mask = np.resize(mask, (512, 512))

        ##print('done')
        # .unsqueeze(0) np.tile(image.copy(),(3,1,1))
        # torch.FloatTensor(np.tile(image.copy(),(3,1,1)))
        # torch.FloatTensor(image.copy())
        return torch.FloatTensor(np.tile(image.copy(),(3,1,1))), torch.FloatTensor(mask.copy())


    def __len__(self):
        return len(self.ids)



