# basic model

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_rnn import DynamicRNN
from ..utils.torch_utils import vectors2padsequence


class Lang:
    def __init__(self, vocab):
        self.index2word = {-1: '<IGN>'}
        self.word2index = {'<IGN>': -1}
        self.__len__ = len(vocab)
        for i in range(len(vocab)):
            self.index2word[i] = vocab[i]
            self.word2index[vocab[i]] = i

    def idx_from_sentence(self, sentence, flag_list=True):
        list_ = sentence if flag_list else sentence.lower().split()
        list_idx = []
        for word in list_:
            if word not in self.word2index:
                print(word)
            list_idx.append(self.word2index[word] if word in self.word2index else self.word2index['<UNK>'])
        return list_idx

    def variables_from_sentences(self, sentences, flag_list=True):
        '''
        if sentence is a list of word, flag_list should be True in the training 
        '''
        indexes = [self.idx_from_sentence(sen, flag_list) for sen in sentences]
        return torch.LongTensor(indexes)


class CharLangModel(nn.Module):
    """docstring for CharLangModel"""
    def __init__(self, dim_char, dim_hidden_char, n_vocab_char, n_vocab, list_embed_char=None, n_layers=1, bias=True, batch_first=True, dropout=0, 
                        bidirectional=False, rnn_type='LSTM'):
        super(CharLangModel, self).__init__()
        self.rnn_type = rnn_type

        self.add_module('embed_char', nn.Embedding(n_vocab_char, dim_char))
        self.add_module('dropout', nn.Dropout(dropout))
        self.add_module('rnn', DynamicRNN(dim_char, dim_hidden_char, n_layers, 
                                            bidirectional=bidirectional, rnn_type=rnn_type))
        self.add_module('linear_lm', nn.Linear(dim_hidden_char * (2 if bidirectional else 1), n_vocab, bias=False))

        if list_embed_char != None:
            self.embed_char.weight.data.copy_(torch.from_numpy(list_embed_char))

    def forward(self, input_char, length_word, length_sen, label_lm):
        e_char = self.embed_char(input_char)
        e_char = self.dropout(e_char)

        # encode word by characters
        output_pad_word, hidden_word = self.rnn(e_char, lengths=length_word, flag_ranked=False)

        h_out_word = hidden_word[0] if self.rnn_type == 'LSTM' else hidden_word
        v_word = torch.cat((h_out_word[-2], h_out_word[-1]), dim=1)

        # format the word encoded from RNN
        embedded_word_from_character = self.dropout(vectors2padsequence(v_word, length_sen))
        prob_log = F.log_softmax(self.linear_lm(embedded_word_from_character), dim=-1)

        loss_batch = []
        for p, g, l in zip(prob_log, label_lm, length_sen):
            loss_batch.append(F.nll_loss(p[: l], g[: l]))
        loss_char_lm = torch.mean(torch.stack(loss_batch))

        return embedded_word_from_character, loss_char_lm


class BERTLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
