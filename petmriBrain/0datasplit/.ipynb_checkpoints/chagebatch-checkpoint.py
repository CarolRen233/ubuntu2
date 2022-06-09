
import pickle
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np



path = 'F:/codes/prtmriBrain/3methods/nnUnet_windows/nnUNet_w/output/nnUNet_preprocessed/Task200_mriBrain/nnUNetPlansv2.1_plans_3D.pkl'

f = open(path, 'rb')
plans = pickle.load(f)

print(plans['plans_per_stage'][0]['batch_size'])
#print(plans['plans_per_stage'][1]['batch_size'])

plans['plans_per_stage'][0]['batch_size'] = 1
#plans['plans_per_stage'][1]['batch_size'] = 1

save_pickle(plans, path)
