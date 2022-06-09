from lib.visual3D_temp import viz
import lib.utils as utils
import SimpleITK as sitk
import numpy as np
import os,glob

config = utils.load_config(os.path.join('./eval_per_case_nnunet.yaml'))

inference_dir = '/home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest'

utils.maybe_create_path(os.path.join(inference_dir,'refined'))

prob_list = sorted(glob.glob(os.path.join(inference_dir,'*.nii.gz')))

for prob in prob_list:
    name = os.path.basename(prob)
    prob_nii = sitk.ReadImage(prob)
    cta_np = sitk.GetArrayFromImage(prob_nii).astype(np.int32)
    save = os.path.join(inference_dir,'refined',name)
    if os.path.exists(save):
        print (save,' exists!')
        continue
    viz.refine_every_case(config, cta_np, prob_nii,save)
    

