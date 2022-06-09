
python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/A/ane_seg/small -pf F:/codes/ubuntu/GLIA-Net/exp/ExA/best_checkpoint-0.3875/small -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl

python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/A/ane_seg/middle -pf F:/codes/ubuntu/GLIA-Net/exp/ExA/best_checkpoint-0.3875/middle -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl 

python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/A/ane_seg/large -pf F:/codes/ubuntu/GLIA-Net/exp/ExA/best_checkpoint-0.3875/large -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_A.pkl 



python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/B/ane_seg/small -pf F:/codes/ubuntu/GLIA-Net/exp/ExB/best_checkpoint-0.3875/small -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl 

python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/B/ane_seg/middle -pf F:/codes/ubuntu/GLIA-Net/exp/ExB/best_checkpoint-0.3875/middle -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl 

python ./evaluate_per_case_nnunet.py -gt E:/IACTA/CellPress1338/external/B/ane_seg/large -pf F:/codes/ubuntu/GLIA-Net/exp/ExB/best_checkpoint-0.3875/large -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_B.pkl 


python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/ane_seg_ts/small -pf F:/codes/ubuntu/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875/small -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl 

python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/ane_seg_ts/middle -pf F:/codes/ubuntu/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875/middle -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl 

python ./evaluate_per_case_nnunet.py -gt F:/data/CellPress1338/ane_seg_ts/large -pf F:/codes/ubuntu/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875/large -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_CellPressTest.pkl 



python ./evaluate_per_case_nnunet.py -gt F:/data/xianjin_data/ane_seg/small -pf F:/codes/ubuntu/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875/small -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl 

python ./evaluate_per_case_nnunet.py -gt F:/data/xianjin_data/ane_seg/middle -pf F:/codes/ubuntu/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875/middle -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl 

python ./evaluate_per_case_nnunet.py -gt F:/data/xianjin_data/ane_seg/large -pf F:/codes/ubuntu/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875/large -pkl F:/codes/ubuntu/nnUnet/file/after_headcut_properties_XJdata.pkl 

