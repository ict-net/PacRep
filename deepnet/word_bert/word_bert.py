# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# word bert module

import os
import torch
import torch.nn as nn

from .bert_models import BertForSequence


class WordBertNet(nn.Module):
    def __init__(self, bert_config_path=None, pretrained_model_path=None, max_num_word=128, use_pretrain_bert=True):
        super(WordBertNet, self).__init__()
        self.max_num_word = max_num_word

        dir_config_bert = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        if bert_config_path is None:
            bert_config_path = os.path.join(dir_config_bert, 'config', 'chinese_L-12_H-768_A-12', 'bert_config.json')
        if pretrained_model_path is None:
            pretrained_model_path = os.path.join(dir_config_bert, 'saved_model', 'chinese_bert_base.bin')
        self.bert = BertForSequence.get_bert_encoder(
            bert_config_path=bert_config_path,
            pretrained_model_path=pretrained_model_path,
            use_pretrain=use_pretrain_bert,
        )

    def forward(self, list_tokens, list_lens):
        tokens, segment_ids, attn_masks = list_tokens['tokens'], list_tokens['segment_ids'], list_tokens['attn_masks']
        rep_text, rep_sub_words = self.bert(tokens, segment_ids, attn_masks)
        output_tensor = []
        # list_lens = list_lens if isinstance(list_lens, np.ndarray) else list_lens.data.cpu().numpy()

        def del_padding(np_list, pad_value=-1):
            for idx, ele in enumerate(np_list):
                if ele == pad_value:
                    return np_list[:idx]
            return np_list

        list_lens_del_pad = [del_padding(tmp) for tmp in list_lens]
        for tensor_item, len_item in zip(rep_sub_words, list_lens_del_pad):
            output_tensor.append(get_word_rep_from_subword(tensor_item[1:], len_item))
        # output_tensor.append(output_tensor[0].data.new(*(self.max_num_word, self.bert.config.hidden_size)).fill_(-1))
        # output_tensor_pad = pad_sequence(output_tensor, batch_first=True)[:-1]
        output_tensor_pad = pad_sequence_with_max_len(output_tensor, batch_first=True, max_len=-1)
        # output_tensor_pad = pad_sequence(output_tensor, batch_first=True)
        return rep_text, output_tensor_pad


def pad_sequence_with_max_len(sequences, batch_first=False, padding_value=0, max_len=-1):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences]) if max_len < 0 else max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def get_word_rep_from_subword(vectors, lengths):
    embedded_ = []
    idx_begin, idx_end = 0, 0
    for len_current in lengths:
        idx_begin, idx_end = idx_end, idx_end + len_current.item()
        embedded_tmp = vectors[idx_begin: idx_end]
        # fix the bug that some word are ignored by bert, then the len_current is 0, then cause nan error
        # Please notice that had better use those when the input of Bert is array.
        if len(embedded_tmp) < 1:
            embedded_tmp = vectors[idx_begin: idx_begin + 1]
        embedded_.append(torch.mean(embedded_tmp, dim=0))
    return torch.stack(embedded_)
