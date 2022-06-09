


#-----------GLIA---------

python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/B/ane_seg -p /home/ubuntu/codes/MyMedicalZoo/output/ExB/pred



python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/B/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExB/best_checkpoint-0.3875


python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/ExB/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/ExB/best_checkpoint-0.3875/refined


python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/B/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExB/best_checkpoint-0.3875/refined


python inference.py -c inference_ExA -i /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/output/ExB_Headcut/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/ExB_headcut/best_checkpoint-0.3875



python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/ExB_headcut/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/ExB_headcut/best_checkpoint-0.3875/refined


python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/output/ExB_Headcut/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExB_headcut/best_checkpoint-0.3875/refined


python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/ExA/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/ExA/best_checkpoint-0.3875/refined


python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExA/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExA/best_checkpoint-0.3875/refined


python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/ExA_headcut/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/ExA_headcut/best_checkpoint-0.3875/refined


python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExA_headcut/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/ExA_headcut/best_checkpoint-0.3875/refined



python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875/refined



python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/CellPress1338/ane_seg_ts -p /home/ubuntu/codes/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/CellPress1338/ane_seg_ts -p /home/ubuntu/codes/GLIA-Net/exp/CellPressTest/best_checkpoint-0.3875/refined



python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/CellPressTest_headcut/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/CellPressTest_headcut/best_checkpoint-0.3875/refined



python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/CellPress1338/ane_seg_ts -p /home/ubuntu/codes/GLIA-Net/exp/CellPressTest_headcut/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/CellPress1338/ane_seg_ts -p /home/ubuntu/codes/GLIA-Net/exp/CellPressTest_headcut/best_checkpoint-0.3875/refined


python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875/refined



python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/xianjin_data/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /mnt/f/data/xianjin_data/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/XJTsN/best_checkpoint-0.3875/refined




python refine_segmentation.py -i /home/ubuntu/codes/GLIA-Net/exp/XJTsN_headcut/best_checkpoint-0.3875 -o /home/ubuntu/codes/GLIA-Net/exp/XJTsN_headcut/best_checkpoint-0.3875/refined



python evaluate_per_case.py -c eval_per_case -g /home/ubuntu/codes/radiology/2HeadCut_New/XJ_headcut/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/XJTsN_headcut/best_checkpoint-0.3875

python evaluate_per_case.py -c eval_per_case -g /home/ubuntu/codes/radiology/2HeadCut_New/XJ_headcut/ane_seg -p /home/ubuntu/codes/GLIA-Net/exp/XJTsN_headcut/best_checkpoint-0.3875/refined

































