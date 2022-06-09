
import os,glob
import math
import SimpleITK as sitk
import time
import logging
from shutil import copyfile
from skimage.transform import resize
import numpy as np
import skimage.measure as measure
from collections import OrderedDict
from scipy.ndimage import label
from matplotlib import pyplot as plt 
from scipy.ndimage import binary_fill_holes


def generate_2Dbrain_mask(brain_matlab_path):
    # 值为1并且没有hole
    brain_nii = sitk.ReadImage(brain_matlab_path)
    brain_np = sitk.GetArrayFromImage(brain_nii).astype(np.int32)
    #brain_mask = brain_np != 0
    #nonhole_mask_bool = binary_fill_holes(brain_mask)
    #nonhole_mask_01 = nonhole_mask_bool.astype(int)

    masks_mips = []
    # 保存最大强度投影
    for i in range(3):
        mask_mip = np.max(brain_np, axis=i)
        masks_mips.append(mask_mip)
    return masks_mips


def generate_2Dbone_mask(bone_np):
    #bone_nii = sitk.ReadImage(ori_bone_path)
    #bone_np = sitk.GetArrayFromImage(bone_nii).astype(np.int32)
    bone_mips = []
    # 保存最大强度投影
    for i in range(3):
        bone_mip = np.max(bone_np, axis=i)
        bone_mips.append(bone_mip)
    return bone_mips





def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def visualize(save_f,**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    plt.savefig(save_f)
    plt.close('all')
    plt.close()


def visualize_overlap(save_f,mask0, mask1,mask2, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    mask = []
    mask.append(mask0)
    mask.append(mask1)
    mask.append(mask2)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        plt.imshow(mask[i], cmap='jet', alpha=0.5)

    #plt.show()
    plt.savefig(save_f)
    plt.close('all')
    plt.close()

def visualize_overlap2(save_f,mask0, mask1, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    mask = []
    mask.append(mask0)
    mask.append(mask1)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        plt.imshow(mask[i], cmap='jet', alpha=0.5)

    #plt.show()
    plt.savefig(save_f)
    plt.close('all')

def save_itk_from_numpy(numpy_data, property):

    pred_itk_image = sitk.GetImageFromArray(numpy_data)
    pred_itk_image.SetSpacing(property["itk_spacing"])
    pred_itk_image.SetOrigin(property["itk_origin"])

    return pred_itk_image



def save_itk(save_np_img, origin_itk_img, output_file):
    save_itk_image = sitk.GetImageFromArray(save_np_img)
    save_itk_image.CopyInformation(origin_itk_img)
    sitk.WriteImage(save_itk_image, output_file)
    
    
def nnunet_raw_update(target_base,patient_name,subset,cta_file):
    '''
    datalist.iloc[i]['subset']
    '''
    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_imagesTs = os.path.join(target_base, "imagesTs")
    target_labelsTr = os.path.join(target_base, "labelsTr")
    target_labelsTs = os.path.join(target_base, "labelsTs")
    
    image_dict = {'test':target_imagesTs,'train':target_imagesTr,'eval':target_imagesTr}
    target_imagesFolder = image_dict[subset]
    
    img0_path = os.path.join(target_imagesFolder, patient_name + "_0000.nii.gz")
    img1_path = os.path.join(target_imagesFolder, patient_name + "_0001.nii.gz")
    img2_path = os.path.join(target_imagesFolder, patient_name + "_0002.nii.gz")
    
    hu_intervals = [[0, 100], [100, 200], [200, 800]]

    img_itk = sitk.ReadImage(cta_file)
    img_npy = sitk.GetArrayFromImage(img_itk)

    img_list = []
    for j in range(len(hu_intervals)):
        hu_channel = np.clip(img_npy, hu_intervals[j][0], hu_intervals[j][1])
        hu_image = (hu_channel - hu_intervals[j][0]) / (hu_intervals[j][1] - hu_intervals[j][0])
        img_list.append(hu_image)
    img_1, img_2, img_3 = img_list

    save_itk(img_1, img_itk, img0_path)
    save_itk(img_2, img_itk, img1_path)
    save_itk(img_3, img_itk, img2_path)
    
    print (img0_path,' 012 saved!')
    

def get_datalist_properties(patients_names,datalist,original_dir,bone_2Dmask_save):
    all_properties = OrderedDict()
    for i in range(len(datalist)):
        patient_name = datalist.iloc[i]['id']
        
        if not (patient_name in patients_names):
            continue
        
        print (patient_name, ' propertiy generate...')
        subset = 'Tr'
        if datalist.iloc[i]['subset'] == 'test':
            subset = 'Ts'

        property = OrderedDict()
        property['subset'] = subset
        property['ori_cta'] = os.path.join(original_dir, 'cta_img', patient_name + '_cta.nii.gz')
        property['ori_seg'] = os.path.join(original_dir, 'ane_seg', patient_name + '_seg.nii.gz')
        #

        data_itk = sitk.ReadImage(property['ori_cta'])
        
        # update 
        #if update_nn:
            #nnunet_raw_update(nnunet_raw,patient_name,datalist.iloc[i]['subset'],data_itk)
        
        # geerate testset
        cta_np = sitk.GetArrayFromImage(data_itk).astype(np.float32)
        hu_channel = np.clip(cta_np,200,800)
        bone_win = (hu_channel - 200) / (800 - 200)
        
        bone_mip0, bone_mip1, bone_mip2 = generate_2Dbone_mask(bone_win)
        np.savez(bone_2Dmask_save + '_bone_mip1.npz', bone_mip1=bone_mip1)
        np.savez(bone_2Dmask_save + '_bone_mip2.npz', bone_mip2=bone_mip2)
        visualize(bone_2Dmask_save + '.png',bone_mip0=bone_mip0,bone_mip1=bone_mip1,bone_mip2=bone_mip2,)
        

        property["original_size_of_raw_data"] = np.array(data_itk.GetSize())[[2, 1, 0]]
        property["original_spacing"] = np.array(data_itk.GetSpacing())[[2, 1, 0]]
        property["itk_origin"] = data_itk.GetOrigin()
        property["itk_spacing"] = data_itk.GetSpacing()
        property["itk_direction"] = data_itk.GetDirection()

        all_properties[patient_name] = property

    return all_properties


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        #print('lmap:', lmap.shape)
        #print('num_objects:', num_objects)

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum()

        #print('object_sizes:', object_sizes)

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def get_brain_coord(image_0,image_1, image_2,brain_max_z, brain_min_z):
    print('image_0 shape:', image_0.shape)

    for i in range(image_2.shape[0] // 2, image_2.shape[0]):
        bone_max_z = image_0.shape[0] - 1
        # print (i)
        if np.sum(image_2[i]) != 0:
            continue
        else:
            bone_max_z = i
            break
    max_z = max([bone_max_z, brain_max_z])
    min_z = brain_min_z

    for j in range(image_0.shape[0] // 2, image_0.shape[0]):
        max_y = image_0.shape[0] - 1
        if np.sum(image_0[j]) != 0:
            continue
        else:
            max_y = j
    for k in reversed(range(0, image_0.shape[0] // 2)):
        min_y = 0
        if np.sum(image_0[k]) != 0:
            continue
        else:
            min_y = k

    for m in range(image_0.shape[1] // 2, image_0.shape[1]):
        max_x = image_0.shape[1] - 1
        if np.sum(image_0[:, m]) != 0:
            continue
        else:
            max_x = m
    for n in reversed(range(0, image_0.shape[1] // 2)):
        min_x = 0
        if np.sum(image_0[:, n]) != 0:
            continue
        else:
            min_x = n

    # threedimage = threedimage[min_z:max_z,min_y:max_y,min_x:max_x]
    return min_z, max_z, min_y, max_y, min_x, max_x

'''
def get_brain_coord(threedimage, brain_max_z, brain_min_z):
    print ('threedimage shape:',threedimage.shape)
    for i in range(threedimage.shape[0] // 2, threedimage.shape[0]):
        threedimage_max_z =  threedimage.shape[0]-1
        # print (i)
        if np.sum(threedimage[i]) != 0:
            continue
        else:
            threedimage_max_z = i
            break
    max_z = max([threedimage_max_z, brain_max_z])
    min_z = brain_min_z
    
    #max_y,min_y = threedimage.shape[1]-1,0
    #max_x,min_x = threedimage.shape[2]-1,0
    
    for j in range(threedimage.shape[1] // 2, threedimage.shape[1]):
        max_y =  threedimage.shape[1]-1
        if np.sum(threedimage[:, j, :]) != 0:
            continue
        else:
            max_y = j
    for k in reversed(range(0, threedimage.shape[1] // 2)):
        min_y = 0
        if np.sum(threedimage[:, j, :]) != 0:
            continue
        else:
            min_y = k

    for m in range(threedimage.shape[2] // 2, threedimage.shape[2]):
        max_x =  threedimage.shape[2]-1
        if np.sum(threedimage[:, :, m]) != 0:
            continue
        else:
            max_x = m
    for n in reversed(range(0, threedimage.shape[2] // 2)):
        min_x = 0
        if np.sum(threedimage[:, :, n]) != 0:
            continue
        else:
            min_x = n

    # threedimage = threedimage[min_z:max_z,min_y:max_y,min_x:max_x]
    
    return min_z, max_z, min_y, max_y, min_x, max_x

'''
def get_cube_as_coords(image,min_z,max_z,min_y,max_y,min_x,max_x):
    cube_image = image[min_z:max_z,min_y:max_y,min_x:max_x]
    #cube_image = np.zeros(image.shape)
    #cube_image[min_z:max_z,min_y:max_y,min_x:max_x] = image[min_z:max_z,min_y:max_y,min_x:max_x]
    return cube_image



    
    

