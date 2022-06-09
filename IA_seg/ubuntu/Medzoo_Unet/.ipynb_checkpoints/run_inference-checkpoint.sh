#python ./inference.py -c inference_ExA_config
#python ./inference.py -c inference_ExB_config
#python ./inference.py -c inference_CellPressTest
#python ./inference.py -c inference_XJTsN


python ./inference.py -c inference_XJTsN
python ./inference.py -c inference_XJ18
python ./inference.py -d cpu -c inference_XJTsN

python ./inference.py -c inference_XJ18_headcut


