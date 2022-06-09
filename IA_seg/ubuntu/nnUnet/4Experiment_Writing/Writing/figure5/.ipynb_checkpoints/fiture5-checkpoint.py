import SimpleITK as sitk
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import os
import glob
import cv2
#import pygame
from PIL import Image
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_itk(path):
    data = sitk.ReadImage(path)
    spacing = data.GetSpacing()
    scan = sitk.GetArrayFromImage(data)
    return scan

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_3d(save_filename,scan,head,color=[1,0,1],threshold=100):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    
    
    #mask
    iD = os.path.basename(scan)[:-7]
    image = read_itk(scan)

    img = image.transpose(2,1,0)
    position = np.nonzero(img)

    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = ax.scatter(position[0],position[1],position[2])

    face_color = color
    
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    
    # head
    head_np = read_itk(head)
    head_img = head_np.transpose(2,1,0)
    head_img = np.clip(head_img, 0, 800)
    verts, faces, _, x = measure.marching_cubes_lewiner(head_img, threshold)
    head_mesh = Poly3DCollection(verts[faces], alpha=0.1)
    head_color = [0, 0, 0]
    head_mesh.set_facecolor(head_color)
    ax.add_collection3d(head_mesh)
    
    

    ax.set_xlim(0, img.shape[0])
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[2])
    ax.set_title(iD)
    
    #save_filename = os.path.join(os.path.dirname(scan),iD + '_3D.png')
    
    ax.axis('off')
    plt.savefig(save_filename,dpi=500,bbox_inches = 'tight')
    #plt.show()
    plt.close()
    return save_filename
    

# ExtB headcut
    
Ids = ['ExtB0020']
name = Ids[0]

gt_folder = '/media/ubuntu/Seagate Expansion Drive/IACTA/CellPress1338/output/ExB_Headcut/ane_seg'
nnunet_pred_folder = '/home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB'
medzoo_folder = '/home/ubuntu/codes/MyMedicalZoo/output/ExB/_headcut_pred'
GLIA_folder = '/home/ubuntu/codes/GLIA-Net/exp/ExB_headcut/best_checkpoint-0.3875'
img_folder = '/media/ubuntu/Seagate Expansion Drive/IACTA/CellPress1338/output/ExB_Headcut/cta_img' 
img = os.path.join(img_folder,name+'.nii.gz')


t = os.path.join(gt_folder,name+'.nii.gz')
plot_3d(name + '_gt' +'.png',t,img,color = [0,1,1],threshold=200)
print (t, ' saved!')



# XJ data full

'''
Ids = ['XJTr0002','XJTr0016','XJTr0012','XJTr0013','XJTr0027']
name = Ids[0]

gt_folder = '/mnt/f/data/xianjin_data/ane_seg'
nnunet_pred_folder = '/home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest/full'
medzoo_folder = ''
GLIA_folder = '/home/ubuntu/codes/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875'
img_folder = '/mnt/f/data/xianjin_data/cta_img'

img = os.path.join(img_folder,name+'.nii.gz')

for n in name:
    t = os.path.join(nnunet_pred_folder,name+'.nii.gz')
    plot_3d(name + 'nnunet' +'.png',t,img,threshold=200)
    print (t, ' saved!')
    
    #t = os.path.join(medzoo_folder,name+'.nii.gz')
    #plot_3d(name + '_medzoo' +'.png',t,img,threshold=200)
    
    
    t = os.path.join(GLIA_folder,name+'.nii.gz')
    plot_3d(name + '_GLIA' +'.png',t,img,threshold=200)
    print (t, ' saved!')
    
    
    t = os.path.join(gt_folder,name+'.nii.gz')
    plot_3d(name + '_gt' +'.png',t,img,color = [0,1,1],threshold=200)
    print (t, ' saved!')
    
    
'''
    
    
    


