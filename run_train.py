# coding: utf-8
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# run file for model training

import os
import sys
import logging
import argparse
import torch
import random
import json
import numpy as np

sys.path.append(os.path.dirname(__file__))
from deepnet.train import TrainModel
from deepnet.utils.torch_utils import get_gpus_mem_info
from deepnet.utils.file_utils import merge_config

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--lr_bert', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_length_sen', type=int, default=256)
parser.add_argument('--max_checkpoint', type=int, default=4096)
parser.add_argument('--sampled_num', type=int, default=1200)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--per_checkpoint', type=int, default=256)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu_mode', type=int, default=1)
parser.add_argument('--dp_mode_gpu_num', type=int, default=-1)
parser.add_argument('--language', type=str, default="en", choices=["zh", "en"])
parser.add_argument('--n_gram', type=int, default=3)
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--lambda2', type=float, default=1)
parser.add_argument('--model_type', type=str, default="BertConCap",
                    choices=["BertSeq", "BertCNN", "BertConCap"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "Adagrad"])
parser.add_argument('--segment_type', type=str, default="char", choices=['char', 'word'])
parser.add_argument('--data_dir', type=str, default='./data/sample')
parser.add_argument('--has_valid', type=int, default=1)
parser.add_argument('--auto_clean_model_mode', type=int, default=1)
parser.add_argument('--breakpoint', type=int, default=-1)
parser.add_argument('--name_model', type=str, default='sample')
config_input = parser.parse_args()
config_input.config_path = './config/model-%s.conf' % config_input.language
config_from_file = argparse.Namespace(**json.load(open(config_input.config_path, 'r')))
config = merge_config(config_input, config_from_file)

if __name__ == "__main__":
    # step 0: config log file, random seed
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    log_file_path = os.path.join(os.path.dirname(__file__), 'log')
    config.data_dir = os.path.join(os.path.dirname(__file__), config.data_dir)
    logging.basicConfig(
        filename=os.path.join(log_file_path, '%s.log' % config.name_model),
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('train')
    # step 1: select the good gpu/cpu
    config.multi_gpu = None
    if torch.cuda.is_available() and config.gpu_mode > 0:
        torch.cuda.manual_seed(config.seed)
        assert config.gpu_mode <= 3, 'Please set correct gpu_mode'
        if config.gpu_mode == 3:
            # use DistributedDataParallel GPU mode
            device = torch.device('cuda')
            print('Using distributed GPUs.')
            logger.info('Using distributed GPUs.')
        elif config.gpu_mode == 1:
            # use single GPU
            # 此处最好不指定idx，使用CUDA_VISIBLE_DEVICES在外部指定
            device = torch.device('cuda')
            print('Using one GPU: %s.' % device)
            logger.info('Using one GPU: %s.' % device)
        else:
            assert config.gpu_mode == 2
            # use DataParallel GPU mode, auto select max_memory GPUs
            assert config.dp_mode_gpu_num > 0, 'Please set GPU number!'
            idx_ranked = get_gpus_mem_info(config.dp_mode_gpu_num)[0]
            device = torch.device('cuda', idx_ranked[0])
            config.multi_gpu = idx_ranked[:config.dp_mode_gpu_num]
            print('Using main GPU: %s, and multi GPU: %s.' % (device, config.multi_gpu))
            logger.info('Using main GPU: %s, and multi GPU: %s.' % (device, config.multi_gpu))
    else:
        config.gpu_mode = 0
        device = torch.device('cpu')
        print('Using device: cpu.')
        logger.info('Using device: cpu.')
    print(config)
    # step 3: build folder for trained model
    folder_path = os.path.join(os.path.dirname(__file__), 'saved_model', config.name_model)
    os.mkdir(folder_path) if not os.path.isdir(folder_path) else None
    # step 4: train model
    train_model = TrainModel(config, device=device)
    train_model.run(config)
