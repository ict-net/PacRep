# coding: utf-8
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# model evaluation

from __future__ import unicode_literals, print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import List

from deepnet.utils.extraction import get_element, get_spans


def eva_classifier(list_t_in, list_p_in, labels=None, mask=None, average='binary'):
    if mask is not None:
        assert len(mask) == len(list_t_in) == len(list_p_in)
        list_t, list_p = [], []
        for t1, p1, flag in zip(list_t_in, list_p_in, mask):
            if flag > 0:
                list_t.append(t1)
                list_p.append(p1)
    else:
        list_t, list_p = list_t_in, list_p_in
    c_m = confusion_matrix(list_t, list_p, labels=labels)
    acc = accuracy_score(list_t, list_p)
    rec = recall_score(list_t, list_p, labels=labels, average=average)
    pre = precision_score(list_t, list_p, labels=labels, average=average)
    f1 = f1_score(list_t, list_p, labels=labels, average=average)
    f1_micro = f1_score(list_t, list_p, labels=labels, average='micro')
    f1_macro = f1_score(list_t, list_p, labels=labels, average='macro')

    return {'c_m': c_m, 'acc': acc, 'f1': f1, 'pre': pre, 'rec': rec, 'f1_macro': f1_macro, 'f1_micro': f1_micro}