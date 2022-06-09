#python ./nnunet/dataset_conversion/Task171_XJallTest.py
#nnUNet_predict -i /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task171_XJallTest/imagesTs/ -o /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest/ -t 152 -m 3d_fullres -chk model_best_110 -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 3

python ./nnunet/dataset_conversion/Task174_CellPressTest152.py
nnUNet_predict -i /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task174_CellPressTest152/imagesTs/ -o /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152/ -t 152 -m 3d_fullres -chk model_best_110 -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 3


#python ./nnunet/dataset_conversion/Task172_ExA.py
#nnUNet_predict -i /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task172_ExA/imagesTs/ -o /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA/ -t 152 -m 3d_fullres -chk model_best_110 -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 3


#python ./nnunet/dataset_conversion/Task173_ExB.py
#nnUNet_predict -i /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_raw_data/Task173_ExB/imagesTs/ -o /home/ubuntu/codes/radiology/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB/ -t 152 -m 3d_fullres -chk model_best_110 -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 3

