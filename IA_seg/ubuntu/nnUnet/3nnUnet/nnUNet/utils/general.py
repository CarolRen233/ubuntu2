import json
import random
import shutil
import time
import pickle
import pandas as pd
import torch.cuda
import torch.backends.cudnn as cudnn
import yaml
import datetime
import logging
import logging.handlers
import os
import sys

import numpy as np
import torch

def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True



def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def shuffle_lists(*ls, seed=777):
    l = list(zip(*ls))
    random.seed(seed)
    random.shuffle(l)
    return zip(*l)


def prepare_input(config,input_tuple, inModalities=-1, inChannels=-1, cuda=False, args=None):
    if args is not None:
        modalities = config['train']['inModalities']
        channels = config['train']['inChannels']
        in_cuda = args.cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = cuda
    if modalities == 4:
        if channels == 4:
            img_1, img_2, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
        elif channels == 3:
            # t1 post constast is ommited
            img_1, _, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_3, img_4), dim=1)
        elif channels == 2:
            # t1 and t2 only
            img_1, _, img_3, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_3), dim=1)
        elif channels == 1:
            # t1 only
            input_tensor, _, _, target = input_tuple
    if modalities == 3:
        if channels == 3:
            #print ('modility 3 channel 3')
            img_1, img_2, img_3, target = input_tuple
            #print ('img_1, img_2, img_3, target shape',img_1.shape, img_2.shape, img_3.shape, target.shape)
            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
            #print ('input_tensor = torch.cat((img_1, img_2, img_3), dim=1):',input_tensor.shape)

        elif channels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif channels == 1:
            input_tensor, _, _, target = input_tuple
    elif modalities == 2:
        if channels == 2:
            img_t1, img_t2, target = input_tuple

            input_tensor = torch.cat((img_t1, img_t2), dim=1)

        elif channels == 1:
            input_tensor, _, target = input_tuple
    elif modalities == 1:
            input_tensor, target = input_tuple

    if in_cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()


    return input_tensor, target


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file


def get_brain_dict(brain_file):
    assert os.path.exists(brain_file)
    brain_data = []
    with open(brain_file) as f:
        for line in f.readlines():
            temp = line.split()
            brain_data.append(temp)
    brain_dict = {}
    for i in range(len(brain_data)):
        _iD = brain_data[i][0][:-8]
        brain_dict[_iD] = list(map(int, brain_data[i][1:]))
    return brain_dict

def load_config(config_filename) -> dict:
    with open(config_filename, 'r', encoding='utf8') as f:
        data = yaml.safe_load(f)
    return data

def save_config(config_filename, data):
    maybe_create_path(os.path.dirname(config_filename))
    with open(config_filename, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def save_csv(csv_rows,csv_path,name_attribute):
    with open(csv_path, mode='w') as file:
        writerCSV = pd.DataFrame(columns=name_attribute, data=csv_rows)
        writerCSV.to_csv(csv_path, encoding='utf-8', index=False)


# ###########
def load_data_info(csv_filename,idx_num):
    '''
    :param idx_num: csv info的数量等于数据ind_num
    :return:{'id': '0813', 'data': '...nii.gz', 'original_spacing': '[0.625    0.488281 0.488281]', 'origin':
     '(-127.9000015258789, -174.89999389648438, -324.5)',
     'direction': '(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)', 'spacing': '[0.625    0.488281 0.488281]',
     'size': '[557 512 512]'}
    '''
    df1 = pd.read_csv(csv_filename)
    data_info_list = np.array(df1).tolist()

    print ('len(data_info_list) "',len(data_info_list) )
    print ('idx_num:',idx_num)
    assert len(data_info_list) == idx_num

    data_info_dict = []
    for info in data_info_list:
        data_info = {}
        for i in range(1, len(df1.columns.values)):
            data_info[df1.columns.values[i]] = info[i]
        data_info_dict.append(data_info)

    return data_info_dict


def get_devices(devices_arg: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = devices_arg.replace(' ', '').split(',')
    if len(devices) > 1 and 'cpu' in devices:
        print ('cannot run on both cpu and gpu. use gpu')
        devices.remove('cpu')
    if devices_arg == 'cpu' or not torch.cuda.is_available():
        print ('use cpu')
        return [torch.device('cpu')]

    cuda_count = torch.cuda.device_count()

    for dev in devices:
        if int(dev) >= cuda_count or int(dev) < 0:
            print ('device %s is not available.' % dev)
            devices.remove(dev)
            continue
    if len(devices) == 0:
        print ('no selected device is available, use cpu')
        return [torch.device('cpu')]
    return [torch.device('cuda', int(i)) for i in devices]


def transpose_move_to_end(arr, index=1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    assert arr.ndim > index
    permute = list(range(arr.ndim))
    permute.pop(index)
    permute.append(index)
    if isinstance(arr, np.ndarray):
        return np.transpose(arr, permute)
    else:
        return arr.permute(permute)

def one_hot(arr, num_classes, axis=-1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    shape = list(arr.shape)
    if isinstance(arr, np.ndarray):
        arr = np.reshape(arr, -1)
        arr = np.eye(num_classes, dtype=arr.dtype)[arr]
        arr = np.reshape(arr, shape + [num_classes])
    else:
        arr = torch.reshape(arr, [-1])
        arr = torch.eye(num_classes, dtype=arr.dtype, device=arr.device)[arr]
        arr = torch.reshape(arr, shape + [num_classes])
    if axis != -1:
        arr = transpose(arr, axis, -1)
    return arr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print ('Unsupported value encountered.')
        
def transpose(arr, first_index=1, second_index=-1):
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)
    assert arr.ndim > max(first_index, second_index)
    if isinstance(arr, np.ndarray):
        permute = list(range(arr.ndim))
        temp = permute[first_index]
        permute[first_index] = permute[second_index]
        permute[second_index] = temp
        return np.transpose(arr, permute)
    else:
        return torch.transpose(arr, first_index, second_index)