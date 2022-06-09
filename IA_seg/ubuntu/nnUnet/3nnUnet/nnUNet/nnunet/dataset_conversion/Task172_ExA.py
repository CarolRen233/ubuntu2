#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from collections import OrderedDict
import glob
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
import torch


def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_itk(save_np_img, origin_itk_img, output_file):
    save_itk_image = sitk.GetImageFromArray(save_np_img)
    save_itk_image.CopyInformation(origin_itk_img)
    sitk.WriteImage(save_itk_image, output_file)

def copy_CellPress_CTA_and_generate_HU(in_file, out_file1,out_file2,out_file3):
    # use this for CTA only!!!
    # # HU value intervals  for input images. Length should be same


    hu_intervals = [[0, 100], [100, 200], [200, 800]]

    img_itk = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img_itk)

    img_list = []
    img_0_tensor = torch.from_numpy(img_npy)
    for j in range(len(hu_intervals)):
        hu_channel = torch.clamp(img_0_tensor, hu_intervals[j][0], hu_intervals[j][1])
        tensor_image = (hu_channel - hu_intervals[j][0]) / (hu_intervals[j][1] - hu_intervals[j][0])
        img_list.append(tensor_image.numpy())
    img_1, img_2, img_3 = img_list

    save_itk(img_1, img_itk, out_file1)
    save_itk(img_2, img_itk, out_file2)
    save_itk(img_3, img_itk, out_file3)



if __name__ == "__main__":


    task_name = "Task172_ExA"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_create_path(target_imagesTr)
    maybe_create_path(target_imagesTs)
    maybe_create_path(target_labelsTr)
    maybe_create_path(target_labelsTs)

    original_dir = '/media/ubuntu/Seagate Expansion Drive/IACTA/CellPress1338/output/ExA_Headcut'
    datalist = pd.read_csv('/home/ubuntu/codes/radiology/3nnUnet/Task172_ExA.csv')

    print (len(datalist))
    tr_patient_names,ts_patient_names = [],[]
    for i in range(len(datalist)):
        patient_name = datalist.iloc[i]['id']
        if patient_name in ['XJTr0151','XJTr0212','XJTr0265']:
            continue
        print('processing ', patient_name)
        if datalist.iloc[i]['subset'] == 'test':
            ts_patient_names.append(patient_name)
        else:
            tr_patient_names.append(patient_name)


        image_dict = {'test':target_imagesTs,'train':target_imagesTr,'eval':target_imagesTr}
        target_imagesFolder = image_dict[datalist.iloc[i]['subset']]
        img0_path = join(target_imagesFolder, patient_name + "_0000.nii.gz")
        img1_path = join(target_imagesFolder, patient_name + "_0001.nii.gz")
        img2_path = join(target_imagesFolder, patient_name + "_0002.nii.gz")

        label_dict = {'test': target_labelsTs, 'train': target_labelsTr, 'eval': target_labelsTr}
        target_labelFolder = label_dict[datalist.iloc[i]['subset']]
        seg_path = join(target_labelFolder, patient_name + ".nii.gz")

        if (all([os.path.exists(img0_path),os.path.exists(img1_path),os.path.exists(img2_path),os.path.exists(seg_path)])):
            print (img0_path,img1_path,img2_path,seg_path)
            print ('%s exists' %patient_name)
            continue


        copy_CellPress_CTA_and_generate_HU(os.path.join(original_dir,'cta_img',patient_name+'.nii.gz'), img0_path,img1_path,img2_path)
        shutil.copy(os.path.join(original_dir,'ane_seg',patient_name+'.nii.gz'),seg_path )


    json_dict = OrderedDict()
    json_dict['name'] = "ExA"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see CellPress"
    json_dict['licence'] = "see CellPress license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1":"HU100200",
        "2": "HU200800"}
    json_dict['labels'] = {
        "0": "background",
        "1": "IA"
    }
    json_dict['numTraining'] = 0
    json_dict['numTest'] = 71
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in tr_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in ts_patient_names]

    save_json(json_dict, join(target_base, "dataset.json"))
    print ('json saved!')