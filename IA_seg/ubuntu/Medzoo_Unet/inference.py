import argparse
import os
import torch
import torch.nn.functional as F
import glob
import configs
# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
from lib.medloaders import medical_image_process as img_loader
import lib.medzoo as medzoo
from lib.losses3D import DiceLoss
from lib.visual3D_temp import viz
import logging
import datetime



def main():
    args = get_arguments()
    if not os.path.exists(args.save):
        utils.make_dirs(args.save)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))
    utils.save_config(os.path.join(args.save, 'inference_config.yaml'), config)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.save, 'log_inference_' + datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S') + '.txt'))
    logging.info(str(args))
    logging.info('config loaded:\n%s', config)


    test_loader = medical_loaders.generate_datasets(args, config)

    model, optimizer = medzoo.create_model(args,config)
    model.restore_checkpoint(config['inference']['pretrained'])
    logging.debug('pretrain model:\n{}'.format(config['inference']['pretrained']))
    
    if not args.device == 'cpu':
        model = model.cuda()
        print ('use cuda')
    
    brain_flag = ''
    if config['inference']['just_head']:
        brain_flag = '_headcut_'
        
    figure7_timespend_txt = []
    for img_info,name in test_loader:
        
        print (img_info['id'])
        output_file = args.save + '/' + name
        if os.path.exists(os.path.join(args.save,'pred',name +'.nii.gz')):
            print (img_info['id'],' exists!')
            continue
        elif os.path.exists(os.path.join(args.save,name +'_refined.nii.gz')):
            print (img_info['id'],' exists!')
            continue
        try:
            timespend = viz.inference_every_case(args, config, img_info, output_file, model)
            figure7_timespend_txt.append(timespend)
            
        except:
            logging.info('{} error !!!!!!!!'.format(name))
    f1 = open(os.path.join(args.save,'figure7timeSpend.txt'),'w')
    for line in figure7_timespend_txt:
        f1.write(line + '\n')
    f1.close()




def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=False, default='inference_ExA_config',
                        help='config name. default: \'default\'')
    parser.add_argument('-d', '--device', type=str, required=False, default='0',
                        help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
    parser.add_argument('--mode', type=str, default='inference')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--augmentation', action='store_true', default=False)

    args = parser.parse_args()
    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))

    args.save = os.path.dirname(config['inference']['pretrained']) + '/' + os.path.basename(config['data']['dataset'])

    return args


if __name__ == '__main__':
    main()
