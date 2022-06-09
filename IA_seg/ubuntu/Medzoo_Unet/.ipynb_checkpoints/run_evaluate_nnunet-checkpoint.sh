python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task172_ExA/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA -pkl after_headcut_properties_A.pkl --justhead 

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task172_ExA/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/refined -pkl after_headcut_properties_A.pkl --justhead 


#python ./evaluate_per_case_nnunet.py -gt /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/ane_seg -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/full -pkl after_headcut_properties_A.pkl

#----------------------------------------------------------------------------------------------




python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task173_ExB/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB -pkl after_headcut_properties_B.pkl --justhead 

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task173_ExB/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/refined -pkl after_headcut_properties_B.pkl --justhead 

#python ./evaluate_per_case_nnunet.py -gt /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/B/ane_seg -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/full -pkl after_headcut_properties_B.pkl


#----------------------------------------------------------------------------------------------

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task174_CellPressTest152/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152 -pkl Task174_CellPressTest152.pkl --justhead 

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task174_CellPressTest152/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152/refined -pkl Task174_CellPressTest152.pkl --justhead 



#python ./evaluate_per_case_nnunet.py -gt /mnt/f/data/CellPress1338/ane_seg_ts -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152 -pkl after_headcut_properties_CellPressTest.pkl 
#----------------------------------------------------------------------------------------------

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task171_XJallTest/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest -pkl /home/ubuntu/codes/radiology/file/after_headcut_properties_XJTsN.pkl --justhead 

python ./evaluate_per_case_nnunet.py -gt /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task171_XJallTest/labelsTs -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest/refined -pkl /home/ubuntu/codes/radiology/file/after_headcut_properties_XJTsN.pkl --justhead 

#python ./evaluate_per_case_nnunet.py -gt /mnt/f/data/xianjin_data/ane_seg_rename -pf /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest -pkl after_headcut_properties.pkl



