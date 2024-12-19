# model description

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..word_bert.word_bert import WordBertNet
from ..utils.const import CONST
from ..model.attention import AttentionPair


class Capsule(nn.Module):
    def __init__(self, config, index):
        super(Capsule, self).__init__()
        self.idx = index
        self.add_module('dropout', nn.Dropout(config.linear_dropout_rate))
        self.add_module('linear', nn.Linear(config.dim_bert, config.n_label, bias=False))
        # self.add_module('attn', AttentionPair(config.dim_bert, config.dim_bert, flag_bid=False))
        self.v_kernel = Parameter(torch.FloatTensor(1, config.dim_bert))
        self.reset_parameters(config.dim_bert)

    def reset_parameters(self, dim_hidden):
        stdv = 1.0 / math.sqrt(dim_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, output_tensor_pad, list_lens, labels, ignore_idx):
        # transfer labels, to do
        unmask = ((labels <= 0) + (labels == 2 * self.idx + 1) + (labels == 2 * self.idx + 2)).int()
        labels_capsule = labels * unmask

        rep = self.dropout(output_tensor_pad)
        prob = torch.softmax(self.linear(rep), dim=-1)
        prob = prob.clamp(min=1e-9, max=1.0)
        prob_cap = prob * unmask.unsqueeze(-1).expand_as(prob)
        loss = F.nll_loss(torch.log(prob.transpose(1, 2)), labels_capsule, ignore_index=ignore_idx)
        # rep_here, attn_sen = self.attn(self.v_kernel, output_tensor_pad, list_lens)
        rep_here = []
        return prob_cap, loss, rep_here


class BertContrastiveCapsule(nn.Module):
    '''
    Decoding the sentences using words
    Inout: sentences
    Output: prob of words in sentence
    '''
    def __init__(self, config, use_pretrain_bert=True):
        super(BertContrastiveCapsule, self).__init__()
        self.add_module('bert', WordBertNet(
            bert_config_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_config_path),
            pretrained_model_path=os.path.join(CONST.APP_ROOT_PATH, config.bert_pretrained_path),
            max_num_word=config.max_length_sen,
            use_pretrain_bert=use_pretrain_bert,
            )
        )

        self.add_module('dropout', nn.Dropout(config.linear_dropout_rate))
        self.add_module('linear', nn.Linear(config.dim_bert, config.n_label, bias=False))

        self.count_capsule = 3

        for i in range(self.count_capsule):
            self.add_module('capsule_%s' % i, Capsule(config, i))

        params_bert = list(map(id, self.bert.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in params_bert, self.parameters()))

    def predict(self, list_tokens, list_lens):
        # encode text(sentence)
        rep_text, output_tensor_pad = self.cal_bert(list_tokens, list_lens)
        prob = self.cal_prob(output_tensor_pad)
        return prob

    def cal_bert(self, list_tokens, list_lens):
        rep_text, output_tensor_pad = self.bert(list_tokens, list_lens)
        return rep_text, output_tensor_pad

    def cal_prob(self, output_tensor_pad):
        rep = self.dropout(output_tensor_pad)
        prob = torch.softmax(self.linear(rep), dim=-1)
        return prob

    def forward(self, list_tokens, list_lens, labels, ignore_idx, is_train: bool = True,
                list_tokens_discourse=None, list_lens_discourse=None):
        # encode text(sentence)
        rep_text, output_tensor_pad = self.cal_bert(list_tokens, list_lens)
        prob_1 = self.cal_prob(output_tensor_pad)
        loss_assign_capsule = F.nll_loss(torch.log(prob_1.transpose(1, 2)), labels, ignore_index=ignore_idx)
        labels_pred = torch.argmax(prob_1, dim=-1)
        labels_pred = labels_pred * (list_lens != -1) - (list_lens == -1).int()

        list_rep = []
        loss_capsule = 0
        prob = 0
        for i in range(self.count_capsule):
            prob_tmp, loss_tmp, rep_tmp = getattr(self, 'capsule_%s' % i)(
                output_tensor_pad, list_lens, labels if is_train else labels_pred, ignore_idx)
            list_rep.append(rep_tmp)
            loss_capsule = loss_capsule + loss_tmp
            prob = prob + prob_tmp

        def cal_hard_attention(prob, list_lens, discourse_related: bool = True):
            mask = (list_lens > -1).int()
            if discourse_related:
                tmp = (torch.argmax(prob, dim=-1) > 0)
            else:
                tmp = (torch.argmax(prob, dim=-1) <= 0)
            att = mask * tmp
            att_sums = att.sum(dim=1, keepdim=True)  # sums per sequence
            attentions = att.div(att_sums + 1e-9)
            return attentions

        def get_rep(att, tensor_in):
            weighted = torch.mul(tensor_in, att.unsqueeze(-1).expand_as(tensor_in))
            rep = weighted.sum(dim=1)
            return rep

        # compute loss
        # loss_seq_label = F.nll_loss(torch.log(prob.transpose(1, 2)), labels, ignore_index=ignore_idx)
        if list_tokens_discourse is not None:
            att = cal_hard_attention(prob, list_lens)
            att_other = cal_hard_attention(prob, list_lens, discourse_related=False)
            rep_discourse_related = get_rep(att, output_tensor_pad)
            rep_discourse_unrelated = get_rep(att_other, output_tensor_pad)
            rep_discourse_ori, _ = self.cal_bert(list_tokens_discourse, list_lens_discourse)
            loss_entropy_1 = 1.0 - F.cosine_similarity(rep_discourse_ori, rep_discourse_related, dim=-1).mean()
            margin = 0
            loss_entropy_2 = max(0, F.cosine_similarity(rep_discourse_ori, rep_discourse_unrelated, dim=-1).mean() - margin)
            loss_entropy = loss_entropy_1 + loss_entropy_2
            loss = loss_assign_capsule + loss_capsule + loss_entropy
        else:
            loss = loss_assign_capsule + loss_capsule
        return loss, prob
