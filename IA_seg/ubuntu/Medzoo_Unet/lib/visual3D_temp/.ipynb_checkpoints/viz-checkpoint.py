import math
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import time
import logging
from shutil import copyfile
from skimage.transform import resize
import numpy as np
import skimage.measure as measure


def roundup(x, base=32):
    return int(math.ceil(x / base)) * base


def refine_every_case(config, binary_prediction,pred_img_itk,save):
    refined_prediction = refine_segment(binary_prediction, config['inference']['kernel_size'], config['inference']['area_threshold'], config['inference']['thin_threshold'])
    save_prediction(refined_prediction, pred_img_itk, save)
    print (save,' done!')
    return refined_prediction

def inference_every_case(args, config, img, output_path, model):

    logging.info('---------------------------id {}------------------------'.format(img['id']))
    start1 = time.time()
    reader = sitk.ImageFileReader()

    
    if config['inference']['just_head']:
        cta_file = img['just_head_cta_save']
        seg_file = img['just_head_seg_save']
    else:
        cta_file = img['ori_cta']
        seg_file = img['ori_seg']

    # load 原始图像
    reader.SetFileName(cta_file)
    pred_img_itk = reader.Execute()
    pred_img_np = sitk.GetArrayFromImage(pred_img_itk).astype(np.float32)
    print ('pred_img_np shape:',pred_img_np.shape)
    logging.info('pred_img_np shape:\n{}'.format(pred_img_np.shape))

    # load 标注mask
    reader.SetFileName(seg_file)
    gt_img_itk = reader.Execute()
    gt_img_np = sitk.GetArrayFromImage(gt_img_itk).astype(np.int32)

    pred_img = torch.tensor(pred_img_np, dtype=torch.float32).cuda()
    gt_img = torch.tensor(gt_img_np, dtype=torch.int8).cuda()
    
    #print('pred_img shape:', pred_img.shape)
    #print('gt_img shape:', gt_img.shape)

    end1 = time.time()
    interval1= end1-start1
    print('Loading  ', img['id'], ' time spent:', interval1)
    logging.info('Loading {},time spent:{} '.format(img['id'],interval1))
    prediction_instance_shape = img['original_size_of_raw_data']

    w1,h1,d1,w2,h2,d2 = -1,-1,-1,-1,-1,-1

    #print('prediction_instance_shape :', type(prediction_instance_shape), prediction_instance_shape)
    prediction_patch_starts = get_sliding_window_patch_starts(gt_img_np, config['data']['patch_size'],
                                                              config['data']['overlap_step'], w1,h1,d1,w2,h2,d2, reference_mask=None)
    prediction_patch_size = config['data']['patch_size']
    #print('prediction_instance_shape:', prediction_instance_shape)
    #logging.info('prediction_patch_starts:\n{}'.format(prediction_patch_starts))

    prediction = np.zeros(prediction_instance_shape, dtype=np.float32)
    overlap_count = np.zeros(prediction_instance_shape, dtype=np.float32)

    hu_intervals = config['data']['hu_values']
    modalities = len(hu_intervals)
    each_modality = []
    for j in range(modalities):
        hu_channel = torch.clamp(pred_img, hu_intervals[j][0], hu_intervals[j][1])
        tensor_image = (hu_channel - hu_intervals[j][0]) / (hu_intervals[j][1] - hu_intervals[j][0])
        each_modality.append(tensor_image)

    for patch_starts in prediction_patch_starts:
        patch_ends = [patch_starts[i] + prediction_patch_size[i] for i in range(3)]
        # hu值，三个模态
        img_1 = each_modality[0][patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                patch_starts[2]:patch_ends[2]]
        img_2 = each_modality[1][patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                patch_starts[2]:patch_ends[2]]
        img_3 = each_modality[2][patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                patch_starts[2]:patch_ends[2]]
        

        #print ('img 123 shape:',img_1.shape,img_2.shape,img_3.shape)

        img_1 = img_1.unsqueeze(0).unsqueeze(0)
        img_2 = img_2.unsqueeze(0).unsqueeze(0)
        img_3 = img_3.unsqueeze(0).unsqueeze(0)

        # print('img 123 shape unsquees:', img_1.shape, img_2.shape, img_3.shape)

        input_tensor = torch.cat((img_1, img_2, img_3), dim=1)

        # print ('input  tensor shape after cat:',input_tensor.shape)

        patch_pred = model.inference(input_tensor)
        # print ('patch_pred shape:',patch_pred.shape)
        patch_pred = patch_pred.squeeze()
        # print ('patch_pred shape:', patch_pred.shape)
        patch_prob, patch_binary = patch_pred.max(dim=0)
        # print('patch_prob ,patch_binary shape:', patch_prob.shape,patch_binary.shape)
        patch_pred = patch_prob * patch_binary
        # print ('patch_pred shape:',patch_pred.shape)

        prediction[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
        patch_starts[2]:patch_ends[2]] += patch_pred.numpy()
        overlap_count[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
        patch_starts[2]:patch_ends[2]] += 1

    overlap_count = np.where(overlap_count == 0, np.ones_like(overlap_count), overlap_count)
    prediction = prediction / overlap_count

    original_spacing = list(map(float, img['original_spacing']))
    
    if config['inference']['just_head']:
        size_of_origin = img['after_size']
    else:
        size_of_origin = img['original_size_of_raw_data']
    
    original_size = list(map(int,size_of_origin ))
    prediction = restore_spacing(prediction, original_spacing,original_size , is_mask=False)

    binary_prediction = (prediction > config['eval']['probability_threshold']).astype(np.int32)

    brain_flag = ''
    if config['inference']['just_head']:
        brain_flag = '_headcut_'
    save_prediction(binary_prediction, pred_img_itk, output_path + brain_flag + '_pred.nii.gz')
    save_prediction(prediction, pred_img_itk, output_path + brain_flag +'_prob.nii.gz')

    end2 = time.time()
    interval2 = end2 - end1
    print('Process ', img['id'], ' time spent:', interval2)
    logging.info('Process {},time spent:{} '.format(img['id'],interval2))
    refined_prediction = refine_segment(binary_prediction, config['inference']['kernel_size'], config['inference']['area_threshold'], config['inference']['thin_threshold'])
    save_prediction(refined_prediction, pred_img_itk, output_path + brain_flag +'_refined.nii.gz')
    #create_2d_views(refined_prediction, gt_img_np, output_path + brain_flag +'_refined_visual.png')

    #copyfile(img['seg'], output_path + '_gt.nii.gz')
    interval3 = time.time() - end2
    print('Refine  ', img['id'], ' time spent:', interval3)
    logging.info('Refine {},time spent:{} '.format(img['id'],interval3))
    
    return (img['id'] + ' ' + str(interval1) + ' ' +  str(interval2) + ' ' +str(interval3))



# TODO TEST
def create_3d_subvol(full_volume, dim):
    list_modalities = []

    modalities, slices, height, width = full_volume.shape

    print('---modalities, slices, height, width:', modalities, slices, height, width)
    print('dim:', dim)
    full_vol_size = tuple((slices, height, width))
    dim = find_crop_dims(full_vol_size, dim)
    print('find_crop_dims dim:', dim)
    for i in range(modalities):
        print('modility:', i)
        TARGET_VOL = modalities - 1

        if i != TARGET_VOL:
            img_tensor = full_volume[i, ...]
            img = grid_sampler_sub_volume_reshape(img_tensor, dim)
            list_modalities.append(img)
        else:
            target = full_volume[i, ...]

    input_tensor = torch.stack(list_modalities, dim=1)
    print('create sub 3d volume input shape:', input_tensor.shape)
    print('create sub 3d volume target shape:', target.shape)

    return input_tensor, target


def grid_sampler_sub_volume_reshape(tensor, dim):
    return tensor.view(-1, dim[0], dim[1], dim[2])


def find_crop_dims(full_size, mini_dim, adjust_dimension=2):
    a, b, c = full_size
    d, e, f = mini_dim

    voxels = a * b * c
    subvoxels = d * e * f

    if voxels % subvoxels == 0:
        return mini_dim

    static_voxels = mini_dim[adjust_dimension - 1] * mini_dim[adjust_dimension - 2]
    print(static_voxels)
    if voxels % static_voxels == 0:
        temp = int(voxels / static_voxels)
        print("temp=", temp)
        mini_dim_slice = mini_dim[adjust_dimension]
        step = 1
        while True:
            slice_dim1 = temp % (mini_dim_slice - step)
            slice_dim2 = temp % (mini_dim_slice + step)
            if slice_dim1 == 0:
                slice_dim = int(mini_dim_slice - step)
                break
            elif slice_dim2 == 0:
                slice_dim = int(temp / (mini_dim_slice + step))
                break
            else:
                step += 1
        return (d, e, slice_dim)

    full_slice = full_size[adjust_dimension]

    return tuple(desired_dim)


# Todo  test!
def save_3d_vol(predictions, spacing, origin, save_path):
    out = sitk.GetImageFromArray(predictions)
    out.SetSpacing(spacing)
    out.SetOrigin(origin)
    sitk.WriteImage(out, save_path + '.nii.gz')
    print('3D vol saved')
    # alternativly  pred_nifti_img.tofilename(str(save_path))


def save_prediction(prediction, origin_itk_img, output_file):
    pred_itk_image = sitk.GetImageFromArray(prediction)
    pred_itk_image.CopyInformation(origin_itk_img)
    sitk.WriteImage(pred_itk_image, output_file)


def get_sliding_window_patch_starts(input_img: np.ndarray,patch_size, overlap_step, w1,h1,d1,w2,h2,d2, reference_mask=None):
    assert input_img.ndim == 3
    d, h, w = input_img.shape
    print('w1,h1,d1,w2,h2,d2:', w1, h1, d1, w2, h2, d2)
    if w1 == -1:
        print ('no brain coords')
        d_starts = list(range(0, d - patch_size[0], overlap_step[0])) + [d - patch_size[0]]
        h_starts = list(range(0, h - patch_size[1], overlap_step[1])) + [h - patch_size[1]]
        w_starts = list(range(0, w - patch_size[2], overlap_step[2])) + [w - patch_size[2]]
    else:
        print('buyingg')
        d_starts = list(range(d1, d2 - patch_size[0], overlap_step[0])) + [d2 - patch_size[0]]
        h_starts = list(range(h1, h2 - patch_size[1], overlap_step[1])) + [h2 - patch_size[1]]
        w_starts = list(range(w1, w2 - patch_size[2], overlap_step[2])) + [w2 - patch_size[2]]


    patch_starts = []
    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                if reference_mask is None:
                    patch_starts.append([ds, hs, ws])
                else:
                    if reference_mask[ds:ds + patch_size[0], hs:hs + patch_size[1], ws:ws + patch_size[2]].sum() \
                            > 0.05 * patch_size[0] * patch_size[1] * patch_size[2]:
                        patch_starts.append([ds, ws, hs])  # only compute patches who have more than 5% reference mask
    return patch_starts


def restore_spacing(prediction, spacing, original_size, is_mask=True):
    if spacing is not None:
        if is_mask:
            return resize_segmentation(prediction, new_shape=original_size)
        else:
            return resize_image(prediction, new_shape=original_size)
    return prediction


def resize_segmentation(segmentation, old_spacing=None, new_spacing=None, new_shape=None, order=0, cval=0):
    '''
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple(
            [int(np.round(old_spacing[i] / new_spacing[i] * float(segmentation.shape[i]))) for i in range(3)])
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True,
                      anti_aliasing=False).astype(tpe)
    else:
        unique_labels = np.unique(segmentation)
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True,
                                       anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resize_image(image, old_spacing=None, new_spacing=None, new_shape=None, order=1):
    assert new_shape is not None or (old_spacing is not None and new_spacing is not None)
    if new_shape is None:
        new_shape = tuple([int(np.round(old_spacing[i] / new_spacing[i] * float(image.shape[i]))) for i in range(3)])
    resized_image = resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)
    return resized_image


def refine_segment(input_img, kernel_size, area_threshold, thin_threshold):
    devices = torch.device('cuda')
    # morph close
    morph_close_img = torch.tensor(input_img,device=devices)
    morph_close_img = torch.unsqueeze(torch.unsqueeze(morph_close_img, 0), 0).type(torch.float32)
    padding = kernel_size // 2
    # Dilated
    morph_close_img = torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(morph_close_img)
    # Eroded
    morph_close_img = 1.0 - torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(1.0 - morph_close_img)
    morph_close_img = torch.squeeze(torch.squeeze(morph_close_img, 0), 0).type(torch.int32)

    morph_close_img = morph_close_img.detach().cpu().numpy()

    # remove small or thin targets
    morph_close_label, morph_close_label_num = measure.label(morph_close_img, return_num=True)
    morph_close_props = measure.regionprops(morph_close_label)
    output_label = morph_close_label.copy()
    remove_small_count = 0
    remove_thin_count = 0
    for prop in morph_close_props:
        if prop.area <= area_threshold:
            output_label = np.where(output_label == prop.label,
                                    np.zeros_like(output_label),
                                    output_label)
            remove_small_count += 1
        else:
            for j in range(len(prop.bbox) // 2):
                if prop.bbox[j + len(prop.bbox) // 2] - prop.bbox[j] <= thin_threshold:
                    output_label = np.where(output_label == prop.label,
                                            np.zeros_like(output_label),
                                            output_label)
                    remove_thin_count += 1
                    break
    output_img = (output_label > 0).astype(np.int32)

    return output_img



