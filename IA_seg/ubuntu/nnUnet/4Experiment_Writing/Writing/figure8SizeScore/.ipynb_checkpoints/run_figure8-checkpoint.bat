
call D:\software\Anaconda3\Scripts\activate.bat D:\software\Anaconda3\envs\nnunet
::cd F:\codes\ubuntu\nnUnet\4Experiment_Writing\Writing\figure8SizeScore


::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder E:/IACTA/CellPress1338/output/ExA_Headcut/ane_seg  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl  --save_csv_name ExA_Headcut_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder E:/IACTA/CellPress1338/output/ExB_Headcut/ane_seg  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl  --save_csv_name ExB_Headcut_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder F:/data/CellPress1338/headcut/ane_seg  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl  --save_csv_name CellPressTest_Headcut_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder E:/IACTA/CellPress1338/external/A/ane_seg  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl  --save_csv_name ExA_Full_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder E:/IACTA/CellPress1338/external/B/ane_seg  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl  --save_csv_name ExB_Full_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder F:/data/CellPress1338/ane_seg_ts  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl  --save_csv_name CellPressTest_full_gt.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder F:/data/xianjin_data/ane_seg --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl --save_csv_name XJdata_full_gt.csv


::D:/Anaconda3/envs/nnunet/python.exe figure8SizeScor.py --ori_folder F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task172_ExA  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl  --save_csv_name ExA_headcut_nnunet.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task173_ExB  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl  --save_csv_name ExB_headcut_nnunet.csv

::python figure8SizeScor.py --ori_folder F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task174_CellPressTest152  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl  --save_csv_name CellPressTest_headcut_nnunet.csv

::D:\Anaconda3\envs\nnunet\python.exe figure8SizeScor.py --ori_folder F:/codes/ubuntu/nnUnet/3nnUnet/nnUNet_output/nnUNet_trained_models/nnUNet/3d_fullres/Task152_CellPress110/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task171_XJallTest  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl  --save_csv_name XJdata_headcut_nnunet.csv

::python figure8SizeScor.py --ori_folder F:/codes/ubuntu/GLIA-Net/exp/ExA/best_checkpoint-0.3875  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl  --save_csv_name ExA_full_GLIA_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/GLIA-Net/exp/ExB/best_checkpoint-0.3875  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl  --save_csv_name ExB_full_GLIA_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl  --save_csv_name CellPressTest_full_GLIA_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl  --save_csv_name XJdata_full_GLIA_unrefined.csv


::--------

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/Medzoo_Unet/output/ExA/pred  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl  --save_csv_name ExA_full_Medzoo_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/Medzoo_Unet/output/ExB/pred  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl  --save_csv_name ExB_full_Medzoo_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/Medzoo_Unet/output/CellPressTest/pred  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl  --save_csv_name CellPressTest_full_Medzoo_unrefined.csv

python figure8SizeScor.py --ori_folder F:/codes/ubuntu/Medzoo_Unet/output/XJTsN/pred  --property_file F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl  --save_csv_name XJdata_full_Medzoo_unrefined.csv


