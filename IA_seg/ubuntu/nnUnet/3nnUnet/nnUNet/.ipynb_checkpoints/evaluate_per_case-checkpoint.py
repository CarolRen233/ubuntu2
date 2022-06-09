import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from collections import OrderedDict
import logging, datetime
import SimpleITK as sitk
import numpy as np
import torch
import glob
import utils as utils
import faulthandler
faulthandler.enable()
import pickle

parser = argparse.ArgumentParser(description='AneurysmSeg study evaluation')
parser.add_argument('-c', '--config', type=str, required=False, default='eval_per_case_nnunet',
                    help='config name. default: \'study_evaluate\'')
parser.add_argument('-d', '--device', type=str, required=False, default='cpu',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-m', '--mask', type=utils.str2bool, default='true', required=False,
                    help='If prediction file or folder is segmentation mask. Else probability distribution. ')
parser.add_argument('-h', '--justhead', action='store_true', help='head cut evaluate')
parser.add_argument('-pkl', '--pkl_file', type=str, required=True, default='')
parser.add_argument('-gt', '--gt_file_or_folder', type=str, required=True, default='')
parser.add_argument('-pf', '--pred_file_or_folder', type=str, required=True, default='')

args = parser.parse_args()
ori_config = utils.load_config(args.config + '.yaml')
args.logging_folder = args.pred_file_or_folder


def save_itk_from_numpy(numpy_data, properti):

    pred_itk_image = sitk.GetImageFromArray(numpy_data)
    pred_itk_image.SetSpacing(properti["itk_spacing"])
    pred_itk_image.SetOrigin(properti["itk_origin"])

    return pred_itk_image


def study_evaluate(config, gt_file_or_folder, pred_file_or_folder,pkl_filename, justhead, devices, pred_type='mask'):
    assert pred_type in ['mask', 'prob']
    logging.info('use device %s' % args.device)
    logging.info('gt_file_or_folder: %s' % gt_file_or_folder)
    logging.info('pred_file_or_folder: %s' % pred_file_or_folder)
    logging.info('mask or probability distribution: %s' % pred_type)
    if pred_type == 'prob':
        logging.info('threshold: %1.2f' % config['eval'].get('probability_threshold', 0.5))
        drop_phrase = None
        require_phrase = '_prob'
    else:
        drop_phrase = '_prob'
        require_phrase = None
    logging.info('Begin to scan gt_folder_or_file %s...' % gt_file_or_folder)
    gt_instances = sorted(glob.glob(os.path.join(gt_file_or_folder, '*.nii.gz')))
    logging.info('Begin to scan pred_folder_or_file %s...' % pred_file_or_folder)
    pred_instances = sorted(glob.glob(os.path.join(pred_file_or_folder, '*.nii.gz')))

    gt_new = []
    for gt in gt_instances:
        for pred in pred_instances:
            if os.path.basename(gt) == os.path.basename(pred):
                gt_new.append(gt)
    gt_instances =gt_new


    assert len(gt_instances) == len(pred_instances), 'numbers of gt_instances and pred_instances do not match'
    logging.info('instance number: %d. start evaluating...' % len(gt_instances))
    
    if not justhead:
        logging.info('===================evaluate full====================')
        print ('===================evaluate full====================')
        pkl_file = open(pkl_filename, 'rb')
        properties = pickle.load(pkl_file)
    else:
        logging.info('===================evaluate headcut====================')
        print ('===================evaluate headcut====================')



    eval_metric_fns, eval_curve_fns = utils.get_evaluation_metric(config, devices[0])
    for metric_fn in eval_metric_fns.values():
        metric_fn.reset()

    reader = sitk.ImageFileReader()
    for i, (gt_ins, pred_ins) in enumerate(zip(gt_instances, pred_instances)):
        ins_id = os.path.basename(gt_ins).split('.')[0]
        reader.SetFileName(gt_ins)
        gt_img = reader.Execute()
        gt_img = sitk.GetArrayFromImage(gt_img).astype(np.int32)


        reader.SetFileName(pred_ins)
        pred_img = reader.Execute()
        pred_img = sitk.GetArrayFromImage(pred_img).astype(np.float32)
        
        
        if not justhead:
  
            min_z, max_z, min_y, max_y, min_x, max_x = properties[ins_id]['coords']
            print('ins_id,', ins_id)
            print(min_z, max_z, min_y, max_y, min_x, max_x)
            prediction_instance_shape = properties[ins_id]['before_size']

            prediction = np.zeros(prediction_instance_shape, dtype=np.float32)
            prediction[min_z:max_z, min_y:max_y, min_x:max_x] =pred_img

            pred_itk_image = sitk.GetImageFromArray(prediction)
            pred_itk_image.SetSpacing(properties[ins_id]["itk_spacing"])
            pred_itk_image.SetOrigin(properties[ins_id]["itk_origin"])

            sitk.WriteImage(pred_itk_image, pred_ins[:-7]+'_full.nii.gz')

            pred_img = prediction


        gt_img = torch.unsqueeze(torch.tensor(gt_img, dtype=torch.int8, device=devices[0]), 0)  # [b, ...]
        pred_img = torch.unsqueeze(torch.tensor(pred_img, dtype=torch.float32, device=devices[0]), 0)
        pred_img = torch.stack([1.0 - pred_img, pred_img], 1)  # [b, c, ...]

        current_metrics = OrderedDict()
        depth = pred_img.shape[2]
        if pred_img.shape[2] > 500:
            for key, metric_fn in eval_metric_fns.items():
                current_metrics[key] = metric_fn(pred_img[:, :, :depth // 2], gt_img[:, :depth // 2])
                current_metrics[key] = metric_fn(pred_img[:, :, depth // 2:], gt_img[:, depth // 2:])
                if isinstance(current_metrics[key], float):
                    current_metrics[key] = current_metrics[key] / 2
        else:
            for key, metric_fn in eval_metric_fns.items():
                current_metrics[key] = metric_fn(pred_img, gt_img)

        logging_info = '(%d in %d) %s:' % (i + 1, len(gt_instances), ins_id)
        print ('(%d in %d) %s:' % (i + 1, len(gt_instances), ins_id))
        for metric_name, metric_value in current_metrics.items():
            if isinstance(metric_value.item(), int):
                logging_info += '\t%s: %d' % (metric_name, metric_value.item())
                print ('\t%s: %d' % (metric_name, metric_value.item()))
            else:
                logging_info += '\t%s: %1.4f' % (metric_name, metric_value.item())
                print ('\t%s: %1.4f' % (metric_name, metric_value.item()))
        logging.info(logging_info)

    logging_info = 'overall:'
    print ('==========OVERALL=========')
    for metric_name, metric_fn in eval_metric_fns.items():
        if isinstance(metric_fn.result.item(), int):
            logging_info += '\t%s: %d' % (metric_name, metric_fn.result.item())
            print ('\t%s: %d' % (metric_name, metric_fn.result.item()))
        else:
            logging_info += '\t%s: %1.4f' % (metric_name, metric_fn.result.item())
            print ('\t%s: %1.4f' % (metric_name, metric_fn.result.item()))
    logging.info(logging_info)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.logging_folder,
                                                                  'log_evaluate_per_case_' + datetime.datetime.now().strftime(
                                                                      '%Y-%m-%d#%H-%M-%S') + '.txt'))

    ori_config = utils.load_config( args.config + '.yaml')
    utils.save_config(os.path.join(args.logging_folder, 'evaluate_per_case_config.yaml'), ori_config)

    config = OrderedDict()
    config['model'] = {'num_classes': 2}
    if args.mask:
        config['eval'] = ori_config['eval_mask']
    else:
        config['eval'] = ori_config['eval_prob']
    devices = utils.get_devices(args.device)
    print ('aa')
    pred_type = 'mask' if args.mask else 'prob'
    study_evaluate(config, args.gt_file_or_folder,args.pred_file_or_folder, args.pkl_file, args.justhead,devices, pred_type)
