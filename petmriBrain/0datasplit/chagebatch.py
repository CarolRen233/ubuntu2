
import pickle
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np



path = '/root/workspace/reny/petmriBrain/code/3nnUNet-master/output/nnUNet_preprocessed/Task201_petmriBrain/nnUNetPlansv2.1_plans_3D.pkl'

f = open(path, 'rb')
plans = pickle.load(f)

print(plans['plans_per_stage'][0]['batch_size'])
print(plans['plans_per_stage'][1]['batch_size'])

plans['plans_per_stage'][0]['batch_size'] = 1
plans['plans_per_stage'][1]['batch_size'] = 1

save_pickle(plans, path)
