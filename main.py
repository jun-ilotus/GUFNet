#!/usr/bin/env python
# encoding: utf-8
import os
import yaml
import torch
import datetime
import importlib
import numpy as np
import pytorch_lightning as pl

from easydict import EasyDict
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from pytorch_lightning.callbacks import LearningRateMonitor

import warnings
warnings.filterwarnings("ignore")


def load_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)
            config = yaml.load(f)
    return config


def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')
    parser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--SWA', action='store_true', default=False, help='StochasticWeightAveraging')
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')
    parser.add_argument('--iou_log_path', type=str, default=None, help='path to save per class IoU log file')
    # testing
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')
    parser.add_argument('--num_vote', type=int, default=1, help='number of voting in the test')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')
    parser.add_argument('--test_split', type=str, default='auto', choices=['auto', 'val', 'test'],
                        help='which split to evaluate in --test mode: auto follows --submit_to_server, or force val/test')
    parser.add_argument('--save_test_predictions', action='store_true', default=False,
                        help='save test predictions to submission files (disabled by default)')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    parser.add_argument('--mask_dir', type=str, default=None, help='directory of precomputed .npz masks for stratified eval')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    if args.test:
        config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote
    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)


def build_loader(config):
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    train_config = config['dataset_params']['train_data_loader']
    val_config = config['dataset_params']['val_data_loader']
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None

    def should_use_test_split():
        split_mode = str(config.get('test_split', 'auto')).lower()
        if split_mode == 'test':
            return True
        if split_mode == 'val':
            return False
        return bool(config['submit_to_server'])

    if not config['test']:
        train_pt_dataset = pc_dataset(config, data_path=train_config['data_path'], imageset='train')
        val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val')
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(train_pt_dataset, config, train_config),
            batch_size=train_config["batch_size"],
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),
            shuffle=train_config["shuffle"],
            num_workers=train_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        # config['dataset_params']['training_size'] = len(train_dataset_loader) * len(configs.gpu)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=1),
            batch_size=val_config["batch_size"],
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),
            shuffle=val_config["shuffle"],
            pin_memory=True,
            num_workers=val_config["num_workers"]
        )
    else:
        if should_use_test_split():
            test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test', num_vote=val_config["batch_size"])
            test_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )
        else:
            val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val', num_vote=val_config["batch_size"])
            val_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )

    return train_dataset_loader, val_dataset_loader, test_dataset_loader


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    print(configs)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    num_gpu = len(configs.gpu)

    # output path
    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)
    if not configs.test:
        wandb_logger = WandbLogger(
            project=configs['dataset_params']['Wandb_Project'],
            name=configs.log_dir,
            save_dir="./wandb_logs/"+log_folder,
            config=dict(configs)
        )
    os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)
    profiler = SimpleProfiler(output_filename=f'{log_folder}/{configs.log_dir}/profiler.txt')
    np.set_printoptions(precision=4, suppress=True)

    # save the backup files
    backup_dir = os.path.join(log_folder, configs.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    if not configs['test']:
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp main.py {}'.format(backup_dir))
        os.system('cp dataloader/dataset.py {}'.format(backup_dir))
        os.system('cp dataloader/pc_dataset.py {}'.format(backup_dir))
        os.system('cp {} {}'.format(configs.config_path, backup_dir))
        os.system('cp network/base_model.py {}'.format(backup_dir))
        os.system('cp network/SPVCNN_LGC.py {}'.format(backup_dir))
        os.system('cp {}.py {}'.format('network/' + configs['model_params']['model_architecture'], backup_dir))

    # reproducibility
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(configs.seed)
    config_path = configs.config_path

    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)

    pl.seed_everything(configs.seed)
    checkpoint_callback = ModelCheckpoint(
        monitor=configs.monitor,
        mode='max',
        save_last=True,
        save_top_k=configs.save_top_k)

    if configs.checkpoint is not None:
        print('load pre-trained model...')
        if configs.fine_tune or configs.test or configs.pretrain2d:
            # my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=(not configs.pretrain2d))
            my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs,
                                                     strict=False)
        else:
            # continue last training — 用当前 yaml config 覆盖，确保新增配置项（如 puaf_vis）生效
            my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=False)

    if configs.SWA:
        swa = [StochasticWeightAveraging(swa_epoch_start=configs.train_params.swa_epoch_start, annealing_epochs=1)]
    else:
        swa = []

    if not configs.test:
        # init trainer
        print('Start training...')
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='ddp',
                             max_epochs=configs['train_params']['max_num_epochs'],
                             resume_from_checkpoint=configs.checkpoint if not configs.fine_tune and not configs.pretrain2d else None,
                             callbacks=[checkpoint_callback,
                                        LearningRateMonitor(logging_interval='step'),
                                        EarlyStopping(monitor=configs.monitor,
                                                      patience=configs.stop_patience,
                                                      mode='max',
                                                      verbose=True),
                                        ] + swa,
                             logger=[tb_logger, wandb_logger],
                             profiler=profiler,
                             check_val_every_n_epoch=configs.check_val_every_n_epoch,
                             gradient_clip_val=1,
                             accumulate_grad_batches=1
                             )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)

    else:
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='ddp',
                             logger=[tb_logger],
                             profiler=profiler)
        eval_loader = test_dataset_loader if test_dataset_loader is not None else val_dataset_loader
        trainer.test(my_model, eval_loader)