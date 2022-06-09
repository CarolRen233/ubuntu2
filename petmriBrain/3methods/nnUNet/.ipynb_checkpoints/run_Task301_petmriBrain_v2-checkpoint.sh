python ./nnunet/dataset_conversion/Task301_petmriBrain_v2.py
nnUNet_plan_and_preprocess -t 301
nnUNet_train 2d nnUNetTrainerV2 Task301_petmriBrain_v2 0 --npz

nnUNet_train 2d nnUNetTrainerV2 Task301_petmriBrain_v2 0 --npz -c
nnUNet_train 2d nnUNetTrainerV2 Task301_petmriBrain_v2 0 --npz -c


nnUNet_predict -i /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_raw_data/Task301_petmriBrain_v2/imagesTs/ -o /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_trained_models/nnUNet/2d/Task301_petmriBrain_v2/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task301_best_infer/ -t 301 -m 2d -chk model_best -f 0 --all_in_gpu True --num_threads_preprocessing 1 --num_threads_nifti_save 1

nnUNet_evaluate_folder -ref /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_raw_data/Task301_petmriBrain_v2/labelsTs/ -pred /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_trained_models/nnUNet/2d/Task301_petmriBrain_v2/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task301_best_infer/ -l 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44


python summary2csv.py -p /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_trained_models/nnUNet/2d/Task301_petmriBrain_v2/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task301_best_infer/

ln -s /media/carol/workspace/codes/petmriBrain/3methods/nnunet_output/nnUNet_trained_models/nnUNet/2d/Task301_petmriBrain_v2/nnUNetTrainerV2__nnUNetPlansv2.1/* /media/carol/workspace/codes/petmriBrain/4ouput/Task301_petmriBrain_v2/

zip -q -r Task301_petmriBrain_v2.zip /media/carol/workspace/codes/petmriBrain/4ouput/Task301_petmriBrain_v2




