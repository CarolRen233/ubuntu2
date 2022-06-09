from torch.utils.data import DataLoader
import pandas as pd
from .iacta import IACTA,IACTA_TESTDATA,iacta_generate_all_train_patches
from .cellpress import CELLPRESS,CELLPRESS_TEST,cellpress_generate_all_train_patches,Radiology_TEST,XJTsN_TEST
import os
import numpy as np
import logging
import lib.utils as utils


def generate_all_patchs(config, origin_csv_path, save_dir):
    if config['data']['dataset'] == "IACTA":
        all_list,list_name = iacta_generate_all_train_patches(config, origin_csv_path, save_dir, subset='all_')
    elif config['data']['dataset'] == "CellPress":
        cp_df = pd.read_csv(config['data']['csv_info_path'])
        logging.info('all data num:{}'.format(len(cp_df)))
        train_df = cp_df[cp_df['subset'] == 'train']
        logging.info('train data num:{}'.format(len(train_df)))
        assert len(train_df)==config['data']['train_num']
        all_list, list_name = cellpress_generate_all_train_patches(config, train_df, save_dir, subset='all_')



    return all_list,list_name



def generate_datasets(args: object, config: object) -> object:
    params = {'batch_size': config['train']['batchSz'],
              'shuffle': True,
              'num_workers': 2}

    if config['data']['dataset'] == "IACTA":
        fold = 'fold_' + str(args.fold)
        df1 = pd.read_csv(os.path.join(config['data'][fold]['save_path'],'aneurysm_seg_' + fold + '.csv'))


        if args.mode == 'inference':
            test_csv = df1[df1['subset'] == 'test']
            test_list = np.array(test_csv).tolist()

            logging.info('test_list num:{}'.format(len(test_list)))

            test_loader = IACTA_TESTDATA(args, config, test_list, 'test')

            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")

            return test_loader


        logging.info('train_list num:{}'.format(len(df1[df1['subset']=='train'])))
        logging.info('valid_list num:{}'.format(len(df1[df1['subset'] == 'eval'])))

        train_loader = IACTA(args, config,fold, 'train')
        val_loader = IACTA(args, config,fold, 'valid')

    elif config['data']['dataset'] == "CellPressTest":
        if args.mode == 'inference':
            test_loader = Radiology_TEST(args, config, 'test')
            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")
            return test_loader
    elif config['data']['dataset'] == "ExA":
        if args.mode == 'inference':
            test_loader = Radiology_TEST(args, config, 'test')
            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")
            return test_loader
    elif config['data']['dataset'] == "ExB":
        if args.mode == 'inference':
            test_loader = Radiology_TEST(args, config, 'test')
            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")
            return test_loader
    elif config['data']['dataset'] == "XJTsN":
        if args.mode == 'inference':
            test_loader = XJTsN_TEST(args, config, 'test')
            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")
            return test_loader
    elif config['data']['dataset'] == "XJ18" or config['data']['dataset'] == 'XJ18_headcut':
        if args.mode == 'inference':
            test_loader = XJTsN_TEST(args, config, 'test')
            print("DATA TEST HAVE BEEN GENERATED SUCCESSFULLY")
            return test_loader

        train_loader = CELLPRESS(args, config, 'train')
        val_loader = CELLPRESS(args, config, 'valid')


    logging.info('training data num: {}'.format(len(train_loader)))
    logging.info('Validation data num: {}'.format(len(val_loader)))

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)


    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")

    return training_generator, val_generator

