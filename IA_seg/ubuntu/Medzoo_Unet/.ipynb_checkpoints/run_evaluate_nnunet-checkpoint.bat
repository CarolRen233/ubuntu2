call D:\software\Anaconda3\Scripts\activate.bat D:\software\Anaconda3\envs\nnunet

::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExA_Headcut/ane_seg/small -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/small -pkl after_headcut_properties_A.pkl --justhead 

::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExB_Headcut/ane_seg/small -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/small -pkl after_headcut_properties_B.pkl --justhead 


::python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/headcut/ane_seg/small -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152/small -pkl after_headcut_properties_CellPressTest.pkl --justhead 

:: LARGE


::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExA_Headcut/ane_seg/large -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/large -pkl after_headcut_properties_A.pkl --justhead 

::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExB_Headcut/ane_seg/large -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/large -pkl after_headcut_properties_B.pkl --justhead 


::python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/headcut/ane_seg/large -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152/large -pkl after_headcut_properties_CellPressTest.pkl --justhead 


:: MIDDLE

::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExA_Headcut/ane_seg/middle -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/middle -pkl after_headcut_properties_A.pkl --justhead 

::python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/output/ExB_Headcut/ane_seg/middle -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/middle -pkl after_headcut_properties_B.pkl --justhead 


::python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/headcut/ane_seg/middle -pf F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152/middle -pkl after_headcut_properties_CellPressTest.pkl --justhead 


:----------GLIANEt

:----------------Medzoo


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/A/ane_seg/small -pf F:/codes/ubuntu/Medzoo_Unet/output/ExA/pred/small

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/A/ane_seg/middle -pf F:/codes/ubuntu/Medzoo_Unet/output/ExA/pred/middle

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/A/ane_seg/large -pf F:/codes/ubuntu/Medzoo_Unet/output/ExA/pred/large


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/B/ane_seg/small -pf F:/codes/ubuntu/Medzoo_Unet/output/ExB/pred/small

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/B/ane_seg/middle -pf F:/codes/ubuntu/Medzoo_Unet/output/ExB/pred/middle

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt E:/IACTA/CellPress1338/external/B/ane_seg/large -pf F:/codes/ubuntu/Medzoo_Unet/output/ExB/pred/large


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/CellPress1338/ane_seg_ts/small -pf F:/codes/ubuntu/Medzoo_Unet/output/CellPressTest/pred/small


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/CellPress1338/ane_seg_ts/middle -pf F:/codes/ubuntu/Medzoo_Unet/output/CellPressTest/pred/middle

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/CellPress1338/ane_seg_ts/large -pf F:/codes/ubuntu/Medzoo_Unet/output/CellPressTest/pred/large


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/xianjin_data/ane_seg/small -pf F:/codes/ubuntu/Medzoo_Unet/output/XJTsN/pred/small

python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/xianjin_data/ane_seg/middle -pf F:/codes/ubuntu/Medzoo_Unet/output/XJTsN/pred/middle


python ./evaluate_per_case.py -c eval_per_case_medzoo -gt F:/data/xianjin_data/ane_seg/large -pf F:/codes/ubuntu/Medzoo_Unet/output/XJTsN/pred/large

