# coding: utf-8
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# train kernel

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import logging
import numpy as np
import torch

from .data.dataset_text import IterTextDataset
from .data.data_utils import text_collate_fn, build_data_for_eva
from .recognition import RecognitionModel
from .evaluation import eva_classifier
from .utils.file_utils import BestPerformanceRecord

logger = logging.getLogger('train')

class TrainModel(object):
    def __init__(self, config, device=torch.device('cpu')):
        self.dataset_name_eva = ('train', 'valid', 'test') if config.has_valid else ('train', 'test')
        self.config = config
        self.num_loss = config.n_loss
        logger.info('Model parameters: %s' % str(config))
        data_loader_num_worker = 4 if config.gpu_mode > 0 else 0
        logger.info('Num of worker in data loader is: %s' % data_loader_num_worker)
        config_label = dict()
        config.tasks_label = json.load(open(config.label_path, 'r'))
        logger.info('task detail: %s' % config.tasks_label)
        config_label['detail'] = config.tasks_label
        config_label['n_task'] = len(config_label['detail'])
        self.config_label = config_label
        self.model = RecognitionModel(config, device=device)
        self.data = {}
        self.data_train = IterTextDataset(
            os.path.join(config.data_dir, 'train.txt'),
            chunk_size=1024 * 1024 * 256,
            config=config,
            config_label=config_label,
            tackle_data=text_collate_fn,
            num_workers=data_loader_num_worker,
            shuffle=True,
            use_distributed=True if config.gpu_mode == 3 else False,
        )
        for tmp in self.dataset_name_eva:
            with open(os.path.join(config.data_dir, '%s.txt' % tmp), 'r') as f:
                self.data[tmp] = build_data_for_eva(
                    f.readlines(1024 * 1024 * 64),
                    max_length_text=config.max_length_sen,
                    config_label=config_label,
                    sampled_num=config.sampled_num,
                )
            logger.info('Dataset Statictis(only for evaluation): %s: %s' % (tmp, len(self.data[tmp])))

        saved_model_folder_path = os.path.join(os.path.dirname(__file__), os.pardir, 'saved_model', config.name_model)
        self.dict_evaluation_index = {'f1_macro': 'high', 'f1_micro': 'high'}
        self.best_performance_recorder = BestPerformanceRecord(
            dict_evaluation_index=self.dict_evaluation_index,
            saved_model_folder_path=saved_model_folder_path,
            logger=logger,
            auto_clean_model_mode=config.auto_clean_model_mode,
            valid_dataset_name='valid' if config.has_valid else 'test'
        )

    def run(self, config):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"summary/{config.name_model}", flush_secs=6)
        if config.breakpoint > 0:
            model_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                      'saved_model', config.name_model, 'model-f1_macro-%s.pth' % config.breakpoint)
            self.model.load_model(model_path)
            logger.info('Load model %s' % model_path)
        start_iter = 0 if config.breakpoint < 0 else (config.breakpoint * config.per_checkpoint)

        loss_step, time_step = np.ones((self.num_loss, )), 0
        start_time = time.time()
        for step in range(start_iter, config.max_checkpoint * config.per_checkpoint):
            if step % config.per_checkpoint == 0:
                if self.config.gpu_mode == 3:
                    if torch.distributed.get_rank() > 0:
                        continue
                show = lambda a: '[%s]' % (' '.join(['%.4f' % x for x in a]))
                n_iter = int(step / config.per_checkpoint)
                time_step = time.time() - start_time
                logger.info("----------------------------------------------------------------------------------")
                logger.info('Time of iter training %.2f s' % time_step)
                logger.info("On iter step %s:, global step %d Loss-step %s" % (n_iter, step, show(loss_step)))
                f1_step = {tmp: dict() for tmp in self.dict_evaluation_index}
                for name in self.dataset_name_eva:
                    loss, dict_eva_all, dict_rep = TrainModel.evaluate(
                        self.model, self.data[name], config.batch_size, self.num_loss, self.config_label)
                    # output the representations
                    with open('./reps/%s.txt' % (name), 'w') as f:
                        for k, v in dict_rep.items():
                            f.write(k + '\n')
                            f.write('\n'.join([str(tmp) for tmp in v]))
                            f.write('\n')
                    writer.add_scalar(f'Loss/{name}', loss[0], n_iter)
                    writer.add_scalar(f'Loss/cla/{name}', loss[1], n_iter)
                    writer.add_scalar(f'Loss/con/{name}', loss[2], n_iter)
                    for evaluation_index_name in self.dict_evaluation_index:
                        f1_step[evaluation_index_name][name] = \
                            np.mean([v[evaluation_index_name] for k, v in dict_eva_all.items()])
                    logger.info(f'In dataset {name}: Loss is {show(loss)}')
                    for id_task in dict_eva_all.keys():
                        task = 'task-%s' % id_task
                        dict_eva = dict_eva_all[id_task]
                        writer.add_scalar(f'Acc/{task}/{name}', dict_eva['acc'], n_iter)
                        writer.add_scalar(f'F1/{task}/{name}', dict_eva['f1'], n_iter)
                        writer.add_scalar(f'F1-macro/{task}/{name}', dict_eva['f1_macro'], n_iter)
                        writer.add_scalar(f'F1-micro/{task}/{name}', dict_eva['f1_micro'], n_iter)

                        logger.info(f"{task}\t\tAcc is {dict_eva['acc']:.4f}, F1 is {dict_eva['f1']:.4f}")
                        logger.info(f"{task}\t\tF1-macro is {dict_eva['f1_macro']:.4f}, F1-micro is {dict_eva['f1_micro']:.4f}")
                        logger.info(f"{task}\t\tPre is {dict_eva['pre']:.4f}, Rec is {dict_eva['rec']:.4f}")
                        logger.info(f"{task}\t\tC_M is \n{dict_eva['c_m']}")
                self.best_performance_recorder.record(f1_step, n_iter, self.model)
                start_time = time.time()
                loss_step = np.zeros((self.num_loss,))
                if config.breakpoint < 0:
                    exit()
            loss_step = loss_step + TrainModel.train(self.model, self.data_train) / config.per_checkpoint

    @staticmethod
    def train(model, data_train):
        batched_data = data_train.get_data()
        loss = model.step_train(batched_data)
        return loss

    @staticmethod
    def evaluate(model, data, batch_size, num_loss, config_label):
        loss = np.zeros((num_loss,))
        st, ed, times = 0, batch_size, 0
        dict_all_pred = {key: [] for key in config_label['detail'].keys()}
        dict_all_label = {key: [] for key in config_label['detail'].keys()}
        dict_all_mask = {key: [] for key in config_label['detail'].keys()}
        dict_rep = {'anchor': [], 'positive': [], 'negative': []}
        while st < len(data):
            selected_data = data[st:ed]
            batched_data = text_collate_fn(selected_data)
            _loss, dict_prob, reps = model.predict(batched_data)
            for k, v in reps.items():
                dict_rep[k].extend(v.cpu().numpy().tolist())
            for task in dict_prob.keys():
                label_tmp_pred = [tmp.item() for tmp in torch.argmax(dict_prob[task], dim=-1)]
                dict_all_pred[task].extend(label_tmp_pred)
                label_tmp = [tmp.item() for tmp in batched_data['anchor']['labels'][task]]
                dict_all_label[task].extend(label_tmp)
                mask_tmp = [tmp.item() for tmp in batched_data['anchor']['mask'][int(task)]]
                dict_all_mask[task].extend(mask_tmp)

            loss = loss + _loss
            st, ed = ed, ed + batch_size
            times += 1
        loss = loss / times

        dict_eva = dict()
        for task in dict_all_label.keys():
            if not config_label['detail'][task]['is_valid']:
                continue
            dict_eva[task] = eva_classifier(dict_all_label[task], dict_all_pred[task], mask=dict_all_mask[task],
                                            average='micro')
        dict_rep['anchor'] = merge_matrix(dict_rep['anchor'], dict_all_label, config_label)
        return loss, dict_eva, dict_rep


def merge_matrix(list_rep, dict_all_label, config_label):
    matrix_str = []
    for i, rep in enumerate(list_rep):
        str_tmp = str(rep)
        for k, v in config_label['detail'].items():
            if v['is_valid']:
                str_tmp += ',' + str(dict_all_label[k][i])

        matrix_str.append(str_tmp)
    return matrix_str
