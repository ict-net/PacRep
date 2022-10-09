# -*- coding:utf8 -*-
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# common operations for extraction

from typing import List
from deepnet.utils.const import CONST


def get_element(labels: List[str]):
    dict_main = {}

    def reset(flag_find: bool = False, tmp_label: str = '', tmp_idx_start: int = -1):
        return flag_find, tmp_label, tmp_idx_start

    flag_find, tmp_label, tmp_idx_start = reset()

    for idx, label in enumerate(labels):
        if label == CONST.IGNORE:
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])
            break
        if label.startswith('B-'):
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])

            tmp_label = label[2:]
            if tmp_label not in dict_main:
                dict_main[tmp_label] = []
            flag_find = True
            tmp_idx_start = idx
        elif label.startswith('I-'):
            if flag_find and (label[2:] == tmp_label):
                continue
            else:
                # avoid the pattern such as, 'B-source', 'I-content'
                if flag_find:
                    dict_main[tmp_label].append([tmp_idx_start, idx])

                flag_find, tmp_label, tmp_idx_start = reset()
        else:
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])

            flag_find, tmp_label, tmp_idx_start = reset()
    else:
        if flag_find:
            dict_main[tmp_label].append([tmp_idx_start, len(labels)])
    return dict_main


def get_spans(token: List[str], dict_idx: dict, lang: str):
    dict_span = dict()
    for k, v in dict_idx.items():
        if k not in dict_span:
            dict_span[k] = []
        for start, end in v:
            tmp = '' if lang == 'zh' else ' '
            dict_span[k].append(tmp.join(token[start: end]))
    return dict_span


if __name__ == '__main__':
    label_test = ["B-source", "I-cue", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
                 "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
                 "O", "O", "O", "O", "O", "O", "O", "B-content", "I-content", "<IGN>"]

    dict_label = get_element(label_test)
    print(dict_label)
    print(get_spans(label_test, dict_label, 'en'))
