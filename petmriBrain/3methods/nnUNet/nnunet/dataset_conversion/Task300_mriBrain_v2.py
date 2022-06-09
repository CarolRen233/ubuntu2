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
import glob, os
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


if __name__ == "__main__":

    print(nnUNet_raw_data)
    task_name = "Task300_mriBrain_v2"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_create_path(target_imagesTr)
    maybe_create_path(target_imagesTs)
    maybe_create_path(target_labelsTr)
    maybe_create_path(target_labelsTs)

    original_dir = '/media/carol/workspace/data/petmrBrain'
    datalist = pd.read_csv('/media/carol/workspace/codes/petmriBrain/0datasplit/pet_mri_Brain_info_v2.csv')

    tr_patient_names, ts_patient_names = [], []
    for i in range(len(datalist)):
        patient_name = str(datalist.iloc[i]['id']).zfill(3)
        if datalist.iloc[i]['subset'] == 'test':
            ts_patient_names.append(patient_name)
        else:
            tr_patient_names.append(patient_name)

        image_dict = {'test': target_imagesTs, 'train': target_imagesTr}
        target_imagesFolder = image_dict[datalist.iloc[i]['subset']]
        img0_path = join(target_imagesFolder, patient_name + "_0000.nii.gz")
        # img1_path = join(target_imagesFolder, patient_name + "_0001.nii.gz")

        label_dict = {'test': target_labelsTs, 'train': target_labelsTr}
        target_labelFolder = label_dict[datalist.iloc[i]['subset']]
        seg_path = join(target_labelFolder, patient_name + ".nii.gz")

        if (all([os.path.exists(img0_path), os.path.exists(seg_path)])):
            print(img0_path, seg_path)
            print('%s exists' % patient_name)
            continue

        # ori_pet_file = os.path.join(original_dir, 'raw_pet', 'pet_' + patient_name + '.nii.gz')
        ori_mri_file = os.path.join(original_dir, 'raw_mri', 'mri_' + patient_name + '.nii.gz')
        assert os.path.exists(ori_mri_file)
        ori_mask_file = os.path.join(original_dir, 'raw_mask', 'mask_' + patient_name + '.nii.gz')
        assert os.path.exists(ori_mask_file)
        os.symlink(ori_mri_file, img0_path)
        # os.symlink(ori_pet_file,img1_path)
        os.symlink(ori_mask_file, seg_path)

    json_dict = OrderedDict()
    json_dict['name'] = "petmriBrain"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI"}

    label_dict = OrderedDict()
    label_dict['0'] = "Background"
    label_dict['1'] = "Left-Cerebral-White-Matter"
    label_dict['2'] = "Left-Cerebral-Cortex"
    label_dict['3'] = "Left-Lateral-Ventricle"
    label_dict['4'] = "Left-Inf-Lat-Vent"
    label_dict['5'] = "Left-Cerebellum-White-Matter"
    label_dict['6'] = "Left-Cerebellum-Cortex"
    label_dict['7'] = "Left-Thalamus-Proper"
    label_dict['8'] = "Left-Caudate"
    label_dict['9'] = "Left-Putamen"
    label_dict['10'] = "Left-Pallidum"
    label_dict['11'] = "3rd-Ventricle"
    label_dict['12'] = "4th-Ventricle"
    label_dict['13'] = "Brain-Stem"
    label_dict['14'] = "Left-Hippocampus"
    label_dict['15'] = "Left-Amygdala"
    label_dict['16'] = "CSF"
    label_dict['17'] = "Left-Accumbens-area"
    label_dict['18'] = "Left-VentralDC"
    label_dict['19'] = "Left-vessel"
    label_dict['20'] = "Left-choroid-plexus"
    label_dict['21'] = "Right-Cerebral-White-Matter"
    label_dict['22'] = "Right-Cerebral-Cortex"
    label_dict['23'] = "Right-Lateral-Ventricle"
    label_dict['24'] = "Right-Inf-Lat-Vent"
    label_dict['25'] = "Right-Cerebellum-White-Matter"
    label_dict['26'] = "Right-Cerebellum-Cortex"
    label_dict['27'] = "Right-Thalamus-Proper"
    label_dict['28'] = "Right-Caudate"
    label_dict['29'] = "Right-Putamen"
    label_dict['30'] = "Right-Pallidum"
    label_dict['31'] = "Right-Hippocampus"
    label_dict['32'] = "Right-Amygdala"
    label_dict['33'] = "Right-Accumbens-area"
    label_dict['34'] = "Right-VentralDC"
    label_dict['35'] = "Right-vessel"
    label_dict['36'] = "Right-choroid-plexus"
    label_dict['37'] = "WM-hypointensities"
    label_dict['38'] = "non-WM-hypointensities"
    label_dict['39'] = "Optic-Chiasm"
    label_dict['40'] = "CC_Posterior"
    label_dict['41'] = "CC_Mid_Posterior"
    label_dict['42'] = "CC_Central"
    label_dict['43'] = "CC_Mid_Anterior"
    label_dict['44'] = "CC_Anterior"

    json_dict['labels'] = label_dict

    json_dict['numTraining'] = 60
    json_dict['numTest'] = 60
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             tr_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in ts_patient_names]

    save_json(json_dict, join(target_base, "dataset.json"))
    print('json saved!')