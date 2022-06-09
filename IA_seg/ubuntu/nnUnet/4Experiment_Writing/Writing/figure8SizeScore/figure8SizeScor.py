import sys, glob, os, argparse
import SimpleITK as sitk
import pandas as pd
import shutil
import pprint, pickle
import numpy as np
from skimage import measure
import math

parser = argparse.ArgumentParser(description='dcom to nii')
parser.add_argument('--ori_folder', type=str, help='orifolder')
parser.add_argument('--property_file', type=str, help='')
parser.add_argument('--save_csv_name', type=str, help='')

def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_csv(csv_rows,csv_path,name_attribute):
    with open(csv_path, mode='w') as file:
        writerCSV = pd.DataFrame(columns=name_attribute, data=csv_rows)
        writerCSV.to_csv(csv_path, encoding='utf-8', index=False)
    

def save_itk_from_numpy(numpy_data, property):

    pred_itk_image = sitk.GetImageFromArray(numpy_data)
    pred_itk_image.SetSpacing(property["itk_spacing"])
    pred_itk_image.SetOrigin(property["itk_origin"])

    return pred_itk_image




def compt_real_size(original_spacing):
    space_prod = np.prod(original_spacing)
    small_voxels = int(((4/3)*(math.pi)*((5/2)**3))/space_prod)
    middle_voxels = int(((4/3)*(math.pi)*((15/2)**3))/space_prod)
    return small_voxels,middle_voxels

def generate_datainfo_csv(name,small_IAs_num,middle_IAs_num,large_IAs_num):
 
    #name_attribute = ['id','small_IAs_num','middle_IAs_num','large_IAs_num']
    
    img = dict()
    img['id'] = name
    img['small_IAs_num'] = small_IAs_num
    img['middle_IAs_num'] = middle_IAs_num
    img['large_IAs_num'] = large_IAs_num
    
    return img





def main():
    args = parser.parse_args()
    
    
    maybe_create_path(os.path.join(args.ori_folder,'small'))
    maybe_create_path(os.path.join(args.ori_folder,'middle'))
    maybe_create_path(os.path.join(args.ori_folder,'large'))
    
    ori_predictions_list = sorted(glob.glob(os.path.join(args.ori_folder,'*.nii.gz')))
    print ('patients num:',len(ori_predictions_list))
    print (ori_predictions_list[:2])
    
    pkl_file = open(args.property_file, 'rb')
    propertes = pickle.load(pkl_file)
    
    csv_info_list = []
    for prediction in ori_predictions_list:

        small_IAs_num, middle_IAs_num, large_IAs_num = 0,0,0

        patientID = os.path.basename(prediction)[:-7]
        
        print (patientID)

        pred_nii = sitk.ReadImage(prediction)
        pred_np = sitk.GetArrayFromImage(pred_nii).astype(np.int32)

        pred_bin_np = pred_np > 0
        pred_lbls_np = measure.label(pred_bin_np)
        labels = np.unique(pred_lbls_np)

        if len(labels) == 1 and labels[0] == 0:
            shutil.copyfile(prediction,os.path.join(args.ori_folder,'small',patientID+'.nii.gz'))
            shutil.copyfile(prediction,os.path.join(args.ori_folder,'middle',patientID+'.nii.gz'))
            shutil.copyfile(prediction,os.path.join(args.ori_folder,'large',patientID+'.nii.gz'))

        elif len(labels) > 1:
            labels = labels[1:]

            small_save_np = np.zeros_like(pred_lbls_np)
            middle_save_np = np.zeros_like(pred_lbls_np)
            large_save_np = np.zeros_like(pred_lbls_np)

            small_v,middle_v = compt_real_size(propertes[patientID]['original_spacing'])

            for lbl in labels:
                pred_lbl_np = np.zeros_like(pred_lbls_np)
                lbl_voxels = np.sum(pred_lbls_np == lbl)
                assert lbl_voxels > 0
                if lbl_voxels < small_v:
                    small_save_np[pred_lbls_np == lbl] = 1
                    small_IAs_num += 1
                elif (lbl_voxels >= small_v) and (lbl_voxels < middle_v):
                    middle_save_np[pred_lbls_np == lbl] = 1
                    middle_IAs_num += 1
                elif lbl_voxels >= middle_v:
                    large_save_np[pred_lbls_np == lbl] = 1
                    large_IAs_num += 1

            small_nii = save_itk_from_numpy(small_save_np, propertes[patientID])
            sitk.WriteImage(small_nii, os.path.join(args.ori_folder,'small',patientID+'.nii.gz'))

            middle_nii = save_itk_from_numpy(middle_save_np, propertes[patientID])
            sitk.WriteImage(middle_nii, os.path.join(args.ori_folder,'middle',patientID+'.nii.gz'))

            large_nii = save_itk_from_numpy(large_save_np, propertes[patientID])
            sitk.WriteImage(large_nii, os.path.join(args.ori_folder,'large',patientID+'.nii.gz'))

        csv_info_list.append(generate_datainfo_csv(patientID,small_IAs_num, middle_IAs_num, large_IAs_num))

    name_attribute = ['id','small_IAs_num','middle_IAs_num','large_IAs_num']
    save_csv(csv_info_list,args.save_csv_name,name_attribute)


    print('All done!')


if __name__ == '__main__':
    main()