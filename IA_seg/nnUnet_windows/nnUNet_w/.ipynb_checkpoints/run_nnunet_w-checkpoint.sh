cd .\apex-master\
python setup.py install

cd nnUNet_w
pip install pathos
pip install batchgenerators==0.21
nnUNet_predict -i F:/codes/nnUnet_windows/nnUNet_w/output/nnUNet_raw/nnUNet_raw_data/Task176_XJ18_full/imagesTs/ -o F:/codes/nnUnet_windows/nnUNet_w/output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task176_XJ18_full -t 152 -m 3d_fullres -chk model_best_110 -f 0 --all_in_gpu True --mode fastest --num_threads_preprocessing 2 --num_threads_nifti_save 2