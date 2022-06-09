#cd .\apex-master\
#python setup.py install

#cd nnUNet_w
#pip install pathos
#pip install batchgenerators==0.21




nnUNet_plan_and_preprocess -t 200 -pl2d None
nnUNet_train 3d_fullres nnUNetTrainerV2 Task200_mriBrain 0 --npz
nnUNet_predict -i F:/codes/prtmriBrain/3methods/nnUnet_windows/nnUNet_w/output/nnUNet_raw/nnUNet_raw_data/Task200_mriBrain/imagesTs/ -o F:/codes/prtmriBrain/3methods/nnUnet_windows/nnUNet_w/output/nnUNet_trained_models/nnUNet/3d_fullres/Task200_mriBrain/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Testimages -t 200 -m 3d_fullres -chk model_best  -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 1 --num_threads_nifti_save 1