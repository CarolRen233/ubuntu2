import csv
import os,glob
import nibabel as nib
import numpy as np
import random
import logging
import lib.medloaders as medical_loaders
import argparse
import lib.utils as utils
import pandas as pd
from lib.medloaders.medical_loader_utils import ane_seg_patch_generator,get_dataimg_info

def main():

    args = get_arguments()
    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))
    utils.save_config(os.path.join(args.save, 'generate_patches_config.yaml'), config)
    if not os.path.exists(args.save):
        utils.make_dirs(args.save)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.save, 'log_generate_patches.txt'))


    # 构建每个instance的信息
    brain_dict = utils.get_brain_dict(config['data']['brain_file'])
    df1 = pd.read_csv((config['data']['instances_csv_path']))
    data_list = np.array(df1).tolist()
    name_attribute = ['id', 'subset', 'cta', 'seg','brain_x_start','brain_y_start','brain_z_start','brain_x_end','brain_y_end','brain_z_end','is_ruptured','institution_id','age','gender','num_IAs', 'spacing1', 'spacing2','spacing3','original_size', 'origin', 'direction', 'slices','IA_voxels']
    dataimg_info_list = get_dataimg_info(config, data_list, brain_dict, name_attribute)
    utils.save_csv(dataimg_info_list, config['data']['csv_info_path'], name_attribute)

    # generate patches
    patches_paths,save_txt_file = medical_loaders.generate_all_patchs(config, config['data']['csv_info_path'], args.save)

    # generate 5 cross fold data
    cp_df = pd.read_csv(config['data']['csv_info_path'])
    all_train_index = cp_df[cp_df['subset'] == 'train'].index.tolist()

    utils.make_dirs(os.path.join(args.save,'fold'))
    for j in range(int(config['data']['fold_num'])):
        fold = 'fold_' + str(j+1)
        random.seed(config['data'][fold]['random_seed'])
        print ('all_train_index:',all_train_index)
        valid_index = random.sample(all_train_index, config['data']['split_fold_valid'])
        print ('valid_index:',valid_index)
        train_index = []

        for t in all_train_index:
            if t in  valid_index:
                continue
            train_index.append(t)
        print('train_index:', train_index)
        fold_df = cp_df
        fold_df.loc[valid_index, 'subset'] = 'eval'

        fold_save_dir = os.path.join(args.save,'fold',fold)
        utils.make_dirs(fold_save_dir)
        fold_df.to_csv(os.path.join(fold_save_dir,config['data'][fold]['save_path']))

        get_fold_path_list(config,patches_paths, valid_index, train_index, fold_save_dir,fold,save_txt_file,fold_df)







def get_fold_path_list(config,all_list_file, valid_index, train_index,fold_save_dir,fold,save_txt_file,fold_df):


    fold_train_name = fold_df.iloc[train_index]['id'].tolist()
    fold_valid_name = fold_df.iloc[valid_index]['id'].tolist()

    # save patch filepath txt
    fold_save_train_list, fold_save_valid_list = [],[]
    for i, npz in enumerate(all_list_file):
        name = os.path.basename(npz).split('_')[0]
        if name in fold_train_name:
            fold_save_train_list.append(npz)
        elif name in fold_valid_name:
            fold_save_valid_list.append(npz)

    fold_save_train_path = os.path.join(fold_save_dir,fold + '_train' + os.path.basename(save_txt_file)[4:])
    fold_save_valid_path = os.path.join(fold_save_dir,fold + '_valid' + os.path.basename(save_txt_file)[4:])
    utils.save_list(fold_save_train_path, fold_save_train_list)
    utils.save_list(fold_save_valid_path, fold_save_valid_list)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False, default='CellPress1338_defult',
                        help='config name. default: \'default\'')
    args = parser.parse_args()

    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))
    args.save = os.path.join(config['data']['patch_dir'], config['data']['dataset'] + str(config['data']['data_num']))


    return args


if __name__ == '__main__':
    main()



