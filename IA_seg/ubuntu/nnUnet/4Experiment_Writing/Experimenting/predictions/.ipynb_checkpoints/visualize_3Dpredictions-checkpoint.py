import SimpleITK as sitk
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import os
import glob
import cv2
#import pygame
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import shutil

parser = argparse.ArgumentParser(description='3D visualize')
parser.add_argument('--gt_dir', type=str, help='')
parser.add_argument('--nnunet_dir', type=str, help='')
parser.add_argument('--glia_dir', type=str, help='')
parser.add_argument('--medzoo_dir', type=str, help='')
parser.add_argument('--save_dir', type=str, help='')

def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_itk(path):
    data = sitk.ReadImage(path)
    spacing = data.GetSpacing()
    scan = sitk.GetArrayFromImage(data)
    scan = scan.transpose(2,1,0)
    
    return scan

def compose_image_with_hstack(img_list,iD,save_dir):
    images = []
    for img in img_list:
        images.append(cv2.imread(img))
    image = np.hstack(images)
    cv2.imwrite(os.path.join(save_dir,iD + '_all_3D.png'), image)
    


def plot_3d(scan,mark):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    
    iD = os.path.basename(scan)[:-7]
    image = read_itk(scan)

    img = image.transpose(2,1,0)
    position = np.nonzero(img)
    

    fig = plt.figure(figsize=(10, 10))
    
    
    
    ax = fig.add_subplot(111, projection='3d')
    
    
    mesh = ax.scatter(position[0],position[1],position[2])
    face_color = [0.5,0.5,1]
    
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, img.shape[0])
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[2])
    ax.set_title(iD + ' ' +mark)

    #plt.show()
    
    save_filename = os.path.join(os.path.dirname(scan),iD + '_3D.png')
    fig.savefig(save_filename)
    
    
    plt.close('all')
    return save_filename
    




def main():
    args = parser.parse_args()
    
    prediction_list = sorted(glob.glob(os.path.join(args.nnunet_dir,'*.nii.gz')))
    names = [os.path.basename(file) for file in prediction_list]
    
    ori_data_dir = os.path.dirname(args.gt_dir)
    
    for name in names:
        
        iD = name[:-12]
        
        if os.path.exists(os.path.join(args.save_dir,iD,iD+'_medzoo.nii.gz')):
            print (iD,' exists!')
            continue
        
        gt = os.path.join(args.gt_dir,iD+'.nii.gz')
        nnunet = os.path.join(args.nnunet_dir,iD+'_full.nii.gz')
        glia = os.path.join(args.glia_dir,iD+'.nii.gz')
        medzoo = os.path.join(args.medzoo_dir,iD+'.nii.gz')
        
        gt_f = plot_3d(gt,'gt')
        nnunet_f = plot_3d(nnunet,'nnunet')
        glia_f = plot_3d(glia,'glia')
        medzoo_f = plot_3d(medzoo,'medzoo')
        
        maybe_create_path(os.path.join(args.save_dir,iD))
        
        compose_image_with_hstack([gt_f,nnunet_f,glia_f,medzoo_f],iD,os.path.join(args.save_dir,iD))
        
        shutil.copyfile(os.path.join(ori_data_dir,'cta_img',iD+'.nii.gz'),os.path.join(args.save_dir,iD,iD+'_cta.nii.gz'))
        shutil.copyfile(os.path.join(ori_data_dir,'ane_seg',iD+'.nii.gz'),os.path.join(args.save_dir,iD,iD+'_gt.nii.gz'))
        shutil.copyfile(os.path.join(args.nnunet_dir,iD+'_full.nii.gz'),os.path.join(args.save_dir,iD,iD+'_nnunet.nii.gz'))
        shutil.copyfile(os.path.join(args.glia_dir,iD+'.nii.gz'),os.path.join(args.save_dir,iD,iD+'_glia.nii.gz'))
        shutil.copyfile(os.path.join(args.medzoo_dir,iD+'.nii.gz'),os.path.join(args.save_dir,iD,iD+'_medzoo.nii.gz'))
        
        os.remove(gt_f)
        os.remove(nnunet_f)
        os.remove(glia_f)
        os.remove(medzoo_f)

    print('All done!')


if __name__ == '__main__':
    main()