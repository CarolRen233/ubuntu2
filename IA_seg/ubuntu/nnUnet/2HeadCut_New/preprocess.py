import pandas as pd
import os,glob
import numpy as np
from copy import deepcopy
import SimpleITK as sitk
import argparse
import torch
from matplotlib import pyplot as plt
from PIL import Image
import imageio
from utils import *
from batchgenerators.utilities.file_and_folder_operations import *







def preprocess_test_and_train(save_dir,csv,ori_brain_matlab):
    
    datalist = pd.read_csv(csv)
    
    ####=======================test 2D Input=================================
    input_2Dbone = os.path.join(save_dir, 'testset','input_2Dbone')
    brain_mip_2d_save = os.path.join(save_dir,  'testset','output_brain_mip_2d')
    just_head_cta_save = os.path.join(save_dir, 'cta_img')
    just_head_seg_save = os.path.join(save_dir, 'ane_seg')
    just_head_properties_save = './after_headcut_properties.pkl'

    maybe_create_path(brain_mip_2d_save)
    maybe_create_path(just_head_cta_save)
    maybe_create_path(just_head_seg_save)
    maybe_create_path(input_2Dbone)
    
    
    #all patients to be tested
    patients_names = []
    for i in range(len(datalist)):
        patient_name = datalist.iloc[i]['new_id']
        #if datalist.iloc[i]['subset'] == 'test':
        patients_names.append(patient_name)
    print('patients_names:', patients_names)
    
    
    all_properties = load_pickle('./rename_All_Data_Info.pkl')
    
    
    # generate 2d input
    for patient in patients_names:
        patient_input_save =os.path.join(input_2Dbone, patient)
        if not os.path.exists(patient_input_save + '_bone_mip2.npz'):
            full_cta = sitk.ReadImage(all_properties[patient]['full_cta_file'])
            cta_np = sitk.GetArrayFromImage(full_cta).astype(np.float32)
            hu_channel = np.clip(cta_np, 200, 800)
            bone_win = (hu_channel - 200) / (800 - 200)
            bone_mip0, bone_mip1, bone_mip2 = generate_2Dbone_mask(bone_win)
            np.savez(patient_input_save + '_bone_mip0.npz', bone_mip0=bone_mip0)
            np.savez(patient_input_save + '_bone_mip1.npz', bone_mip1=bone_mip1)
            np.savez(patient_input_save + '_bone_mip2.npz', bone_mip2=bone_mip2)
            visualize(patient_input_save + '.png',bone_mip0=bone_mip0,bone_mip1=bone_mip1,bone_mip2=bone_mip2,)
    ####===================================================================
            
        
        
    
    ####======================train set preprocess==========================
    processed_brain_mask = os.path.join(save_dir, 'trainset','train_labels')
    maybe_create_path(processed_brain_mask)
    train_2dbone_save = os.path.join(save_dir, 'trainset','train_images')
    maybe_create_path(train_2dbone_save)
    
    
    #get rename list
    rename_dict = {}
    rename_path = "./raneme.txt"
    with open(rename_path) as f:
        rename = f.readlines()
        renames = [c.strip() for c in rename]
        f.close()
    for name_line in renames:
        ori_name,new_name = name_line.split(' ')[0],name_line.split(' ')[1]
        rename_dict[ori_name] = new_name



    # generate train
    train_ids = []
    for ori_brain_name in os.listdir(ori_brain_matlab):
    #for ori_brain_name in ['XJTr0000.nii.gz', 'XJTr0001.nii.gz', 'XJTr0002.nii.gz', 'XJTr0003.nii.gz', 'XJTr0004.nii.gz']:
        
        #ori_name = ori_brain_name.split('_')[0]
        #print ('ori_name:',ori_name)
        #new_name = rename_dict[ori_name]
        #print ('new_name:',new_name)

        new_name = os.path.basename(ori_brain_name)[:-7]
        
        
        # copy bone
        src_bone = os.path.join(save_dir,'testset','input_2Dbone',new_name)
        dst_bone = os.path.join(train_2dbone_save,new_name)
        if not os.path.exists(dst_bone + '.png'):
            os.symlink(src_bone+'_bone_mip1.npz',dst_bone+'_bone_mip1.npz')
            os.symlink(src_bone+'_bone_mip2.npz',dst_bone+'_bone_mip2.npz')
            os.symlink(src_bone+'.png',dst_bone+'.png')
            
            bone_mip0 = np.load(src_bone+'_bone_mip0.npz')['bone_mip0']
            bone_mip1 = np.load(src_bone+'_bone_mip1.npz')['bone_mip1']
            bone_mip2 = np.load(src_bone+'_bone_mip2.npz')['bone_mip2']
            
        
        ori_brain = os.path.join(ori_brain_matlab,ori_brain_name)
        brain_2Dmask_save = os.path.join(processed_brain_mask,new_name)

        if not os.path.exists(brain_2Dmask_save + '.png'):
            mask_mip0, mask_mip1, mask_mip2 = generate_2Dbrain_mask(ori_brain)
    
            
            #one_region_mask_mip0,_,_ = remove_all_but_the_largest_connected_component(mask_mip0, [1])
            #one_region_mask_mip1,_,_ = remove_all_but_the_largest_connected_component(mask_mip1, [1])
            #one_region_mask_mip2,_,_ = remove_all_but_the_largest_connected_component(mask_mip2, [1])
            
            np.savez(brain_2Dmask_save + '_mask_mip0.npz', mask_mip0=mask_mip0)
            np.savez(brain_2Dmask_save + '_mask_mip1.npz', mask_mip1=mask_mip1)
            np.savez(brain_2Dmask_save + '_mask_mip2.npz', mask_mip2=mask_mip2)
            
            #visualize(brain_2Dmask_save + '.png',
                      #mask_mip0=mask_mip0,
                      #mask_mip1=mask_mip1,
                      #mask_mip2=mask_mip2,)
            visualize_overlap(save_f = brain_2Dmask_save + '.png',
                              mask0=mask_mip0, 
                              mask1=mask_mip1,
                              mask2=mask_mip2,
                              bone_mip0 = bone_mip0,
                              bone_mip1 = bone_mip1,
                              bone_mip2 = bone_mip2,)

        train_ids.append(new_name+'_mask_mip1.npz')
        train_ids.append(new_name+'_mask_mip2.npz')
        
    return train_ids
    
