#python inference.py -c inference_ExA -i /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/A/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/ExA


python inference.py -c inference_ExA -i /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/output/ExA_Headcut/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/ExA_headcut
python inference.py -c inference_ExA -i /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/output/ExB_Headcut/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/ExB_headcut
python inference.py -c inference_ExA -i /mnt/f/data/CellPress1338/headcut/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/CellPressTest_headcut
python inference.py -c inference_ExA -i /home/ubuntu/codes/radiology/2HeadCut_New/XJ_headcut/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/XJTsN_headcut


#python inference.py -c inference_ExA -i /media/ubuntu/Seagate\ Expansion\ Drive/IACTA/CellPress1338/external/B/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/ExB
#python inference.py -c inference_ExA -i /mnt/f/data/CellPress1338/cta_img_ts  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/CellPressTest
#python inference.py -c inference_ExA -i /mnt/f/data/xianjin_data/cta_img  -t nii -o /home/ubuntu/codes/GLIA-Net/exp/XJTsN

python inference.py -c inference_XJ18 -i F:/data/XJ18/cta_img  -t nii -o F:/codes/ubuntu/GLIA-Net/exp/XJ18

python inference.py -c inference_XJ18 -i F:/codes/ubuntu/nnUnet/2HeadCut_New/XJ18_headcut/cta_img  -t nii -o F:/codes/ubuntu/GLIA-Net/exp/XJ18_headcut






