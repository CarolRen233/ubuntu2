来自论文Toward human intervention-free clinical diagnosis of intracranial aneurysm via deep neural network

是先进院的医生团队做的，最最最贴近我的工作的代码




python inference.py \
    -c inference_GLIA-Net \
    -i /root/workspace/renyan/datasets/xianjinData/medical_data/ \
    -d 0,1 \
    -t nii \
    -o /root/workspace/renyan/xianjin/GLIA-Net/out/predictions/
    
    
    
    
    
python evaluate_per_case.py \
    -c eval_per_case \
    -g /root/workspace/renyan/datasets/xianjinData/medical_data/ane_seg/ \
    -p /root/workspace/renyan/xianjin/GLIA-Net/out/predictions/