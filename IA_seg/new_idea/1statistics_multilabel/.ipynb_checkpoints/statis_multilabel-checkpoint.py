import sys, glob, os, argparse
import SimpleITK as sitk
import pandas as pd
import shutil
import pprint, pickle
import numpy as np
from skimage import measure
import math
from utils import *

parser = argparse.ArgumentParser(description='dcom to nii')
parser.add_argument('--ori_folder', type=str, help='orifolder')
parser.add_argument('--property_file', type=str, help='')
parser.add_argument('--save_csv_name', type=str, help='')
parser.add_argument('--ori_csv', type=str, help='')



def main():
    args = parser.parse_args()
    
    maybe_create_path(os.path.join(args.ori_folder,'multilabel'))
    
    ori_predictions_list = sorted(glob.glob(os.path.join(args.ori_folder,'*.nii.gz')))
    print ('patients num:',len(ori_predictions_list))
    print (ori_predictions_list[:2])
    
    pkl_file = open(args.property_file, 'rb')
    propertes = pickle.load(pkl_file)
    
    csv_info_list = []
    
    
    datalist = pd.read_csv(args.ori_csv)
    
    for i in range(len(datalist)):
        patientID = datalist.iloc[i]['instance_id']
        
        prediction = os.path.join(args.ori_folder,patientID+'.nii.gz')

        small_IAs_num, middle_IAs_num, large_IAs_num = 0,0,0

        print (patientID)

        pred_nii = sitk.ReadImage(prediction)
        pred_np = sitk.GetArrayFromImage(pred_nii).astype(np.int32)

        pred_bin_np = pred_np > 0
        pred_lbls_np = measure.label(pred_bin_np)
        labels = np.unique(pred_lbls_np)
        
        diameters = []
        voxels = []

        if len(labels) == 1 and labels[0] == 0:
            shutil.copyfile(prediction,os.path.join(args.ori_folder,'multilabel',patientID+'.nii.gz'))


        elif len(labels) > 1:
            labels = labels[1:]

            new_multilabel_save_np = np.zeros_like(pred_lbls_np)

            space_prod = np.prod(propertes[patientID]['original_spacing'])
            
            for lbl in labels:
                pred_lbl_np = np.zeros_like(pred_lbls_np)
                lbl_voxels = np.sum(pred_lbls_np == lbl)
                
                diameter = 2*(((space_prod*lbl_voxels)*(3/(4*math.pi)))**(1/3))
                
                assert diameter > 0
                
                if lbl_voxels < 10:
                    new_multilabel_save_np[pred_lbls_np == lbl] = 0
                    print ('worong with ', patientID)
                    
                elif diameter < 5:
                    new_multilabel_save_np[pred_lbls_np == lbl] = 1
                    small_IAs_num += 1
                    voxels.append(lbl_voxels)
                    diameters.append(diameter)
                    
                    
                elif (diameter >= 5) and (diameter < 15):
                    new_multilabel_save_np[pred_lbls_np == lbl] = 2
                    middle_IAs_num += 1
                    voxels.append(lbl_voxels)
                    diameters.append(diameter)
                    
                    
                elif diameter >= 15:
                    new_multilabel_save_np[pred_lbls_np == lbl] = 3
                    large_IAs_num += 1
                    voxels.append(lbl_voxels)
                    diameters.append(diameter)
                

            new_multilabel_nii = save_itk_from_numpy(new_multilabel_save_np, propertes[patientID])
            sitk.WriteImage(new_multilabel_nii, os.path.join(args.ori_folder,'multilabel',patientID+'.nii.gz'))


        csv_info_list.append(generate_datainfo_csv(patientID,small_IAs_num, middle_IAs_num, large_IAs_num,diameters,voxels,propertes[patientID],datalist.iloc[i]))

    name_attribute = ['instance_id','institution_id','age','gender','is_ruptured','num_IAs','spacing','slices','size','small_IAs_num','middle_IAs_num','large_IAs_num','diameters','voxels']
    
    
    save_csv(csv_info_list,args.save_csv_name,name_attribute)


    print('All done!')


if __name__ == '__main__':
    main()