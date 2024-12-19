# main model bridge

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from .model.basic_model import Lang
from .data.word_vec import WordVectorManagement
from .model.models import BertConCap
from .model.comcap import BertContrastiveCapsule
from .word_bert.dataset_wordbert import WordBertDataset
from .utils.const import CONST


class RecognitionModel(nn.Module):
    def __init__(self, config, ignore_idx=-1, device=torch.device('cpu')):
        '''
        :param config:
        :param ignore_idx:
        :param device:
        :param use_pretrain_bert: should True when training, but False for inference
        '''
        super(RecognitionModel, self).__init__()
        self.max_length_sen = config.max_length_sen
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.model_type = config.model_type
        self.gpu_mode = config.gpu_mode
        self.ignore_idx = ignore_idx
        self.lambda1 = config.lambda1

        # using word vector
        if 'use_word_vec' in config and config.use_word_vec:
            self.vocab_label = ['<UNK>'] + WordVectorManagement.load_vocab(config.label_path)
            self.lang_label = Lang(self.vocab_label)
            config.embed = WordVectorManagement.load_word_vec(self.vocab_label, dim_word=config.dim_bert)

        self.data_loader_bert = WordBertDataset(
            vocab_file=os.path.join(CONST.APP_ROOT_PATH, config.bert_vocab_path),
            max_seq_len=511,
            is_array=config.bert_is_array
        )

        assert config.model_type in globals() and issubclass(
            globals()[config.model_type], nn.Module), 'Unknown model name: %s' % config.model_type

        self.model = globals()[config.model_type](config)
        if config.gpu_mode == 3:
            # ddp auto set gpu index, and gpu index has been re-mapped.
            # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0,
            #                                      world_size=1)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            ddp_device = torch.device("cuda", local_rank)
            self.device = ddp_device
            self.model.to(ddp_device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   find_unused_parameters=True)
            model_wrapper = self.model.module
        if config.gpu_mode == 2:
            self.device = device
            self.model = torch.nn.DataParallel(self.model, device_ids=config.multi_gpu)
            self.model.to(device)
            model_wrapper = self.model.module
        if (config.gpu_mode == 1) or (config.gpu_mode == 0):
            self.device = device
            # avoid the gpu 0 used unwanted caused by pin_memory, the solution is torch.cuda.set_device(local_rank)
            # another solution is to use CUDA_VISIBLE_DEVICES (suggest)
            self.model.to(device)
            model_wrapper = self.model
        if 'bert' in self.model_type.lower():
            self.optimizer = getattr(optim, config.optim_type)(
                [{'params': model_wrapper.base_params, 'weight_decay': config.weight_decay, 'amsgrad': True},
                 {'params': model_wrapper.bert.parameters(), 'lr': config.lr_bert, 'weight_decay': 0}],
                lr=self.learning_rate)
        else:
            self.optimizer = getattr(optim, config.optim_type)(
                [{'params': model_wrapper.base_params, 'weight_decay': config.weight_decay, 'amsgrad': True},
                 {'params': model_wrapper.embed.parameters(), 'lr': config.lr_word_vector, 'weight_decay': 0,
                  'amsgrad': True}], lr=self.learning_rate)

    def get_batch_data(self, batched_data):

        def get_batch_data_one(input):
            # dict_data['label'] = torch.LongTensor(batched_data['labels']).to(self.device)
            list_tokens, list_lens = self.data_loader_bert.get_batched_data(input['bert_text'], self.device)
            list_lens_mgpu = pad_sequence(
                [torch.tensor(tmp, dtype=torch.long, device=self.device) for tmp in list_lens],
                batch_first=True, padding_value=-1)
            labels = {k: v.to(self.device) for k, v in input['labels'].items()}
            masks = input['mask'].to(self.device)
            len_text = torch.LongTensor(input['len_sen']).to(self.device)
            return list_tokens, list_lens_mgpu, len_text, labels, masks

        batched_data_torch = dict()
        for key, value in batched_data.items():
            batched_data_torch[key] = get_batch_data_one(value)
        return batched_data_torch

    def predict(self, batched_data):
        self.model.eval()
        batched_data_torch = self.get_batch_data(batched_data)
        with torch.no_grad():
            loss_general, loss_cont, dict_probs, reps = self.model(batched_data_torch, self.ignore_idx, is_train=False)
            loss = loss_general + (self.lambda1 * loss_cont if self.lambda1 > 0 else 0)
        if self.gpu_mode == 2:
            loss = loss.mean()
        return np.array([loss.item(), loss_general.item(), loss_cont.item()]).reshape(3), dict_probs, reps

    def step_train(self, batched_data, inference=False):
        # Turn on training mode which enables dropout.
        self.model.eval() if inference else self.model.train()
        if not inference:
            # zero the parameter gradients
            self.optimizer.zero_grad()
        batched_data_torch = self.get_batch_data(batched_data)
        loss_general, loss_cont, dict_probs, _ = self.model(batched_data_torch, self.ignore_idx, is_train=True)
        loss = loss_general + (self.lambda1 * loss_cont if self.lambda1 > 0 else 0)
        if self.gpu_mode == 2:
            loss = loss.mean()
        # prob = self.model(list_tokens, list_lens)
        # prob = prob.clamp(min=1e-12)
        # loss = F.nll_loss(torch.log(prob.transpose(1, 2)), b_data_label, ignore_index=self.ignore_idx)
        # if torch.isnan(loss):
        #     self.save_model("%s/%s" % ("./model", self.name_model), '-1')
        #     assert ~torch.isnan(loss)
        if not inference:
            loss.backward()
            self.optimizer.step()
        # debug mode, do not delete!!!
        # if not inference:
        #     with autograd.detect_anomaly():
        #         # self.model.test.mean().backward()
        #         # prob.mean().backward()
        #         loss.backward()
        #         self.optimizer.step()

        # when use multi-GPU, no matter dataparrallel of distributeddataparrallel,
        # loss is the list of all gpu loss.
        return np.array([loss.item(), loss_general.item(), loss_cont.item()]).reshape(3)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.gpu_mode <= 1:
                name = k.replace('module.', '')  # remove `module.`
            else:
                name = k if 'module.' in k else k[:5] + '.module' + k[5:]  # add `module.`
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict)
