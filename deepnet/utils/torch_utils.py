# basic torch operations

from __future__ import unicode_literals, print_function, division

import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence


def get_gpus_mem_info(n_gpu):
    import pynvml
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(pynvml.nvmlDeviceGetCount())]
    gpus_free = [int(pynvml.nvmlDeviceGetMemoryInfo(handle).free/1024/1024) for handle in handles]
    gpus_idx = np.argsort(gpus_free)[::-1].tolist()[:n_gpu]
    gpus_free = [gpus_free[idx] for idx in gpus_idx]
    return gpus_idx, gpus_free


def vectors2padsequence(vectors, lengths):
    '''

    :param vectors:
    :param lengths: torch.LongTensor
    :return:
    '''
    embedded_ = []
    idx_begin, idx_end = 0, 0
    for len_current in lengths:
        idx_begin, idx_end = idx_end, idx_end + len_current.item()
        embedded_tmp = vectors[idx_begin: idx_end]
        embedded_.append(embedded_tmp)
    return pad_sequence(embedded_, batch_first=True)


def dynamic_softmax(input, input_lengths):
    """ Forward pass.
    # Arguments:
        inputs (Torch.Variable): Tensor of input matrix
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        attentions: dynamic softmax results
    """
    mask = mask_gen(input_lengths)

    # apply mask and renormalize attention scores (weights)
    masked_weights = input * mask
    att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
    dyn_softmax = masked_weights.div(att_sums + 1e-12)

    return dyn_softmax


def mask_gen(input_lengths):
    """ Forward pass.
    # Arguments:
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        mask: mask results
    """
    max_len = torch.max(input_lengths)
    indices = torch.arange(0, max_len, device=input_lengths.device).unsqueeze(0)
    # mask = Variable((indices < input_lengths.unsqueeze(1)).float())
    mask = (indices < input_lengths.unsqueeze(1)).float()

    return mask

