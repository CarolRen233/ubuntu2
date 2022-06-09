import glob
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import *
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from SegNet import VGG16_deconv
from data import Dataset,TrainDataset
from torch.utils.data import DataLoader
import cv2
import imageio
from torch import nn
from tensorboardX import SummaryWriter
from preprocess import preprocess_test_and_train
import argparse
import random


def main():
    
    args = get_arguments()
    #_ = preprocess_test_and_train(args.save_dir,args.csv,args.ori_brain_matlab)

    y_train_dir = os.path.join(args.save_dir, 'trainset','train_labels')
    x_train_dir = os.path.join(args.save_dir, 'trainset','train_images')

    labels1 = glob.glob(os.path.join(y_train_dir,'*_mask_mip1.npz'))
    labels2 = glob.glob(os.path.join(y_train_dir,'*_mask_mip2.npz'))

    label_list = labels1 + labels2
    print('labels:', len(label_list), label_list[:10])
    labels = random.shuffle(label_list)

    ids = [os.path.basename(label) for label in label_list]
    print ('ids:',len(ids),ids[:10])


    train_dataset = TrainDataset(ids,x_train_dir, y_train_dir, classes=['brain'])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_deconv()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])


    epochs = 200
    writer = SummaryWriter('./log')
    result_path = './log'
    save = './model/'
    
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            #print ('image shape',images.shape)
            #print ('labels shape', labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            log_ps = model(images)


            ###print ('log_ps',type(log_ps[0]))
            ###print('log_ps', len(log_ps))
            #ps1 = log_ps[1]
            ###print ('labels',type(labels))
            ###print('labels', labels.shape)

            loss = criterion(log_ps, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if e % 10 ==0:
                torch.save(model, save + 'epoch{}_model.pth'.format(e))
        else:
            print(f"Training loss: {running_loss / len(train_loader)}")


        writer.add_scalars(result_path + 'Train_loss', {result_path + 'train_loss': running_loss},e)


    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--original_dir', type=str, required=False, default='/mnt/f/data/xianjin_data',
                        help='the folder include cta_img and ane_seg')
    parser.add_argument('-csv', '--csv', type=str, required=False, default='./All_Renamed_Data_Info_split.csv',
                        help='ids')
    parser.add_argument('-o', '--save_dir', type=str, required=False, default='/home/ubuntu/codes/radiology/2HeadCut_New/XJ_headcut',
                        help='output folder')
    parser.add_argument('-mt', '--ori_brain_matlab', type=str, required=False, default='/home/ubuntu/codes/radiology/2HeadCut_New/brain_mask',
                        help='input original brain folder')
    #parser.add_argument('-nn', '--nnunet_raw', type=str, required=False, default= '/root/workspace/renyan/output/nnUNet/nnUNet_raw_data/Task154_CellPress110new_justhead',
                        #help='output folder')
    
    #parser.add_argument('--update_nn', action='store_true', default=False, help='update nnunet raw data')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()