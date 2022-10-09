# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# Basic data operation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import torch
import subprocess
import numpy as np



def merge_config(config_high_priority, config_low_priority):
    for k, v in vars(config_high_priority).items():
        vars(config_low_priority)[k] = v
    return config_low_priority


def delete_pretrained_model(folder_path, num_keep=20, filename_pre='model', file_extension='pth'):
    list_idx = []
    for name in os.listdir(folder_path):
        if not name.startswith(filename_pre):
            continue
        str_idx = name[len(filename_pre):-len(file_extension) - 1]
        if str_idx.isdigit():
            list_idx.append(int(str_idx))
    idx_ranked = np.argsort(list_idx).tolist()

    for idx in idx_ranked[:-num_keep]:
        num = list_idx[idx]
        str_command = 'rm %s' % os.path.join(folder_path, '%s%s.%s' % (filename_pre, num, file_extension))
        rc, gopath = subprocess.getstatusoutput(str_command)

        if rc != 0:
            print('Command %s fails.' % str_command)


class BestPerformanceRecord(object):
    def __init__(self, dict_evaluation_index, saved_model_folder_path, logger, auto_clean_model_mode,
                 valid_dataset_name='valid'):
        self.dict_evaluation_index = dict_evaluation_index
        self.f1_best = dict()
        self.idx_best = dict()
        self.valid_dataset_name = valid_dataset_name
        self.auto_clean_model_mode = auto_clean_model_mode
        self.saved_model_folder_path = saved_model_folder_path
        self.logger = logger
        for evaluation_index_name, mode in dict_evaluation_index.items():
            value = 0 if mode == 'high' else 1e9
            self.f1_best[evaluation_index_name] = {'valid': value, 'test': value, 'train': value}
            self.idx_best[evaluation_index_name] = -1
        return

    def record_kernel(self, f1_step, name, current_iter, model):
        if f1_step[name][self.valid_dataset_name] > self.f1_best[name][self.valid_dataset_name]:
            if self.dict_evaluation_index[name] == 'low':
                return self.f1_best[name]['test'], self.idx_best[name]
        else:
            if self.dict_evaluation_index[name] == 'high':
                return self.f1_best[name]['test'], self.idx_best[name]
        self.f1_best[name]['valid'] = f1_step[name]['valid']
        self.f1_best[name]['test'], idx_best = f1_step[name]['test'], current_iter
        self.idx_best[name] = idx_best
        torch.save(model.state_dict(),
                   os.path.join(self.saved_model_folder_path, 'model-%s-%s.pth' % (name, idx_best)))
        torch.save(model.state_dict(), os.path.join(self.saved_model_folder_path, 'best-%s.pth' % name))
        if self.auto_clean_model_mode:
            delete_pretrained_model(self.saved_model_folder_path, filename_pre='model-%s-' % name)
        return self.f1_best[name]['test'], self.idx_best[name]

    def record(self, dict_f1_step, current_iter, model):
        for evaluation_index_name in self.dict_evaluation_index:
            f1_best, idx_best = self.record_kernel(dict_f1_step, evaluation_index_name, current_iter, model)
            self.logger.info('The best %s util now is %.4f, at step %s' % (evaluation_index_name, f1_best, idx_best))
        return
