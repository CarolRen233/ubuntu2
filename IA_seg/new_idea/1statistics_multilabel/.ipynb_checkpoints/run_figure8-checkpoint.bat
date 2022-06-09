
call D:\software\Anaconda3\Scripts\activate.bat D:\software\Anaconda3\envs\nnunet


python statis_multilabel.py --ori_folder E:/IACTA/CellPress1338/output/ExA_Headcut/ane_seg  --property_file ./after_headcut_properties_A.pkl  --save_csv_name ExA_complete_info.csv  --ori_csv E:/IACTA/CellPress1338/external/A/Ainstances.csv

python statis_multilabel.py --ori_folder E:/IACTA/CellPress1338/output/ExB_Headcut/ane_seg  --property_file ./after_headcut_properties_B.pkl  --save_csv_name ExB_complete_info.csv  --ori_csv E:/IACTA/CellPress1338/external/B/Binstances.csv


python statis_multilabel.py --ori_folder F:/data/CellPress1338/headcut_all/ane_seg  --property_file ./after_headcut_properties_CellPressAll.pkl  --save_csv_name CellPressAll_complete_info.csv  --ori_csv E:/IACTA/CellPress1338/CellPress1338/instances_cop1.csv

