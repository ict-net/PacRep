# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# attention module

from __future__ import unicode_literals, print_function, division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ..utils.torch_utils import dynamic_softmax


class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.attention_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(inputs.device)).unsqueeze(0)
        mask = (idxes < input_lengths.unsqueeze(1)).float()

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, attentions



class AttentionPair(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_vect, dim_attn, flag_bid):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(AttentionPair, self).__init__()
        dim_attn_bid = dim_attn * (2 if flag_bid else 1)
        self.add_module('linear_vec', nn.Linear(dim_vect, dim_attn, bias=False))
        self.add_module('linear_mat', nn.Linear(dim_attn_bid, dim_attn, bias=False))
        self.add_module('linear_attn', nn.Linear(dim_attn, 1, bias=False))

    def forward(self, vector, matrix, input_lengths):
        """ Forward pass.
        # Arguments:
            vect (Torch.Variable): Tensor of input vector
            matrix (Torch.Variable): Tensor of input matrix
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        t1 = self.linear_vec(vector)
        t2 = self.linear_mat(matrix)
        t3 = F.relu(t1.unsqueeze(1) + t2)
        logits = self.linear_attn(t3).squeeze(-1)
        unnorm_ai = (logits - logits.max()).exp()

        attentions = dynamic_softmax(unnorm_ai, input_lengths)

        # apply attention weights
        weighted = torch.mul(matrix, attentions.unsqueeze(-1).expand_as(matrix))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return representations, attentions
