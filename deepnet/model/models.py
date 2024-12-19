# model description

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import unicode_literals, print_function, division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsule import Capsule
from ..word_bert.word_bert import WordBertNet
from ..utils.const import CONST


class BertConCap(nn.Module):
    '''
    Decoding the sentences using words
    Inout: sentences
    Output: prob of words in sentence
    '''

    def __init__(self, config):
        super(BertConCap, self).__init__()
        self.add_module('bert', WordBertNet(
            bert_config_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_config_path),
            pretrained_model_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_pretrained_path),
            max_num_word=config.max_length_sen,
            use_pretrain_bert=config.use_pretrain_bert,
            )
        )

        self.add_module('dropout', nn.Dropout(config.linear_dropout_rate))
        for task, label_item in config.tasks_label.items():
            if not label_item['is_valid']:
                continue
            n_label = label_item['value']
            for i in range(n_label):
                self.add_module('cap_%s_%s' % (task, i), Capsule(config.dim_bert, config.linear_dropout_rate))
        self.tasks_label = config.tasks_label
        self.anchor = config.anchor
        self.lambda2 = config.lambda2
        # self.add_module('linear', nn.Linear(config.dim_bert, 5, bias=False))
        params_bert = list(map(id, self.bert.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in params_bert, self.parameters()))

    def predict(self, batched_data, is_train: bool = True):
        # cal representations
        reps = dict()
        output_pad_words = dict()
        for k, v in batched_data.items():
            # if (not is_train) and k != self.anchor:
            #     continue
            list_tokens, list_lens = v[0], v[1]
            rep_text, output_tensor_pad = self.bert(list_tokens, list_lens)
            reps[k] = rep_text
            output_pad_words[k] = output_tensor_pad

        # cal capsule
        dict_probs, dict_sims = dict(), dict()
        for task, label_item in self.tasks_label.items():
            if not label_item['is_valid']:
                continue
            n_label = label_item['value']
            list_prob, list_r_s = [], []
            for i in range(n_label):
                prob_tmp, r_s_tmp = getattr(self, 'cap_%s_%s' % (task, i))(
                    output_pad_words[self.anchor], batched_data[self.anchor][2]
                )
                list_prob.append(prob_tmp)
                list_r_s.append(r_s_tmp)
            list_r_s = torch.stack(list_r_s)
            list_sim = torch.sum(reps[self.anchor] * list_r_s, 2).t()
            prob = torch.stack(list_prob).squeeze(-1).t()
            prob = prob.clamp(min=1e-9, max=1.0)
            dict_probs[task] = prob
            dict_sims[task] = list_sim

        return dict_probs, dict_sims, reps

    @staticmethod
    def cal_hinge_loss(dict_pred, ground_truth, mask, tasks_detail):
        loss_total_task = 0
        for task, prob in dict_pred.items():
            mask_hinge = mask[int(task)].float()
            batch_loss_unmasked = F.multi_margin_loss(prob, ground_truth[task], reduction='none')
            batch_loss = torch.sum(torch.mul(batch_loss_unmasked, mask_hinge)) / (torch.sum(mask_hinge) + CONST.EPS)
            contribution = tasks_detail[task]['contribution'] if 'contribution' in tasks_detail[task] else 1
            loss_total_task = loss_total_task + contribution * batch_loss
        return loss_total_task

    def forward(self, batched_data, ignore_idx, is_train: bool = True):
        dict_probs, dict_sims, reps = self.predict(batched_data, is_train)
        mask = batched_data[self.anchor][-1]
        ground_truth = batched_data[self.anchor][3]
        loss_hinge_classify = self.cal_hinge_loss(dict_probs, ground_truth, mask, self.tasks_label)
        loss_hinge_sim = self.cal_hinge_loss(dict_sims, ground_truth, mask, self.tasks_label)
        loss_general = loss_hinge_classify + (self.lambda2 * loss_hinge_sim if self.lambda2 > 0 else 0)

        # contrastive learning
        if True:
            loss_entropy_1 = 1.0 - F.cosine_similarity(reps[self.anchor], reps['positive'], dim=-1).mean()
            margin = 0
            loss_entropy_2 = max(0, F.cosine_similarity(reps[self.anchor], reps['negative'], dim=-1).mean() - margin)
            loss_contrastive = loss_entropy_1 + loss_entropy_2
        else:
            loss_contrastive = torch.tensor(0)
        return loss_general, loss_contrastive, dict_probs, reps


class BertSeq(nn.Module):
    '''
    Decoding the sentences using words
    Inout: sentences
    Output: prob of words in sentence
    '''

    def __init__(self, config, use_pretrain_bert=True):
        super(BertSeq, self).__init__()
        self.add_module('bert', WordBertNet(
            bert_config_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_config_path),
            pretrained_model_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_pretrained_path),
            max_num_word=config.max_length_sen,
            use_pretrain_bert=use_pretrain_bert,
            )
        )

        self.add_module('dropout', nn.Dropout(config.linear_dropout_rate))
        self.add_module('linear', nn.Linear(config.dim_bert, config.n_label, bias=False))
        params_bert = list(map(id, self.bert.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in params_bert, self.parameters()))

    def predict(self, list_tokens, list_lens):
        # encode text(sentence)
        rep_text, output_tensor_pad = self.bert(list_tokens, list_lens)
        # matrix_in = output_tensor_pad.unsqueeze(1)
        rep = self.dropout(output_tensor_pad)
        prob = torch.softmax(self.linear(rep), dim=-1)
        return prob

    def forward(self, list_tokens, list_lens, labels, ignore_idx, is_train: bool = True):
        prob = self.predict(list_tokens, list_lens)
        # prob = prob.view(prob.size(0) * prob.size(1), prob.size(2))
        prob = prob.clamp(min=1e-9, max=1.0)
        loss = F.nll_loss(torch.log(prob.transpose(1, 2)), labels, ignore_index=ignore_idx)
        return loss, prob


class BertCNN(nn.Module):
    '''
    Decoding the sentences using words
    Inout: sentences
    Output: prob of words in sentence
    '''
    def __init__(self, config, use_pretrain_bert=True):
        super(BertCNN, self).__init__()
        self.add_module('bert', WordBertNet(
            bert_config_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_config_path),
            pretrained_model_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_pretrained_path),
            max_num_word=config.max_length_sen,
            use_pretrain_bert=use_pretrain_bert,
            )
        )

        self.add_module('conv_1_gram', nn.Sequential(
            nn.Conv2d(1, config.dim_hidden, (1, config.dim_bert)),
            nn.ReLU(True),
            )
        )

        self.add_module('dropout', nn.Dropout(config.linear_dropout_rate))
        self.add_module('linear', nn.Linear(config.dim_hidden, config.n_label, bias=False))
        params_bert = list(map(id, self.bert.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in params_bert, self.parameters()))

    def predict(self, list_tokens, list_lens):
        # encode text(sentence)
        rep_text, output_tensor_pad = self.bert(list_tokens, list_lens)
        matrix_in = output_tensor_pad.unsqueeze(1)
        # Convolution
        c_1 = self.conv_1_gram(matrix_in)
        r_1 = c_1.squeeze(-1).transpose(-1, -2)
        rep = self.dropout(r_1)
        prob = torch.softmax(self.linear(rep), dim=-1)
        return prob

    def forward(self, list_tokens, list_lens, labels, ignore_idx, is_train: bool = True):
        prob = self.predict(list_tokens, list_lens)
        prob = prob.clamp(min=1e-9, max=1.0)
        loss = F.nll_loss(torch.log(prob.transpose(1, 2)), labels, ignore_index=ignore_idx)
        return loss, prob

