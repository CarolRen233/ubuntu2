# Python libraries

import argparse
import os
import logging
import datetime
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
import yaml

seed = 1777777


def main():
    args = get_arguments()
    utils.reproducibility(args, seed)


    if not os.path.exists(args.save):
        utils.make_dirs(args.save)

    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))
    utils.save_config(os.path.join(args.save, 'train_config.yaml'), config)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.save, 'log_train_' + datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S') + '.txt'))
    logging.info(str(args))
    logging.info('config loaded:\n%s', config)

    training_generator, val_generator = medical_loaders.generate_datasets(args, config)

    model, optimizer = medzoo.create_model(args,config)

    if args.resume:
        model.restore_checkpoint(args.resume)
        logging.info('resume mode checkpoint:'.format(args.resume))
    criterion = DiceLoss(classes=config['train']['classes'])

    if args.cuda:
        model = model.cuda()

    trainer = train.Trainer(args, config, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator)
    print ('kaishixunlian')
    trainer.training()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False, default='CellPress1338_defult',
                        help='config name. default: \'default\'')
    parser.add_argument('-n', '--exp_id', type=int, required=True, default=100,
                        help='to identify different exp ids.')
    parser.add_argument('-f', '--fold', type=int, required=True, default=1,
                        help='cross fold,can be 12345, 1 is same to ')
    parser.add_argument('--fullCTA', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


    args = parser.parse_args()

    config = utils.load_config(os.path.join('./configs', args.config + '.yaml'))
    args.save = os.path.join(config['train']['train_save_dir'],config['data']['dataset'] + str(config['data']['data_num']),'fold','fold_' + str(args.fold), 'exp_' +str(args.exp_id))
    args.log_dir = args.save + '/log/'

    return args

if __name__ == '__main__':
    main()