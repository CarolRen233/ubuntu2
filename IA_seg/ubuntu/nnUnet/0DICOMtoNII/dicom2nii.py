import sys, glob, os, argparse
import numpy as np
# import dicom2nifti
import pandas as pd
import logging
import time
import shutil
from multiprocessing import Pool, Value, Lock
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='dcom to nii')
parser.add_argument('--dicom_folder', type=str, help='dicom folder')
parser.add_argument('--output_path', type=str, help='output path')
parser.add_argument("-n", "--num_process", type=int, default=30, help='num of workers')

count = Value('i', 0)
lock = Lock()


def dcm2nii(dcms_path, nii_path):
    
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)


def process(opts):
    label, args = opts

    patientID = os.path.basename(label)[:-7]
    original_dicom_directory = os.path.dirname(label)

    output_file = args.output_path + '/cta_img/' + patientID + '.nii.gz'
    label_dst = args.output_path + '/ane_seg/' + patientID + '.nii.gz'
    
    if not os.path.exists(output_file):
        dcm2nii(original_dicom_directory, output_file)
    else:
        print (output_file,'exists!')
    if not os.path.exists(label_dst):
        shutil.copyfile(label, label_dst)
    else:
        print (label_dst,'exists!')

    global lock
    global count

    with lock:
        count.value += 1


def run(args):
    opts_list = []
    
    

    print ('patients num:',len(label_list))
    for label in label_list:
        print(label)
        opts_list.append((label, args))

    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main():
    args = parser.parse_args()
    run(args)
    print('All done!')


if __name__ == '__main__':
    main()