# word vector management

import os
import logging
import numpy as np
from typing import List


from deepnet.utils.const import CONST

logger = logging.getLogger('word_vec')


class WordVectorManagement(object):
    def __init__(self, config):
        self.voc_size = config.voc_size
        self.dim_word = config.dim_word
        self.max_length = config.max_length
        self.n_label = config.n_label

    @staticmethod
    def load_vocab(filepath_vocab):
        with open(os.path.join(CONST.APP_ROOT_PATH, filepath_vocab), 'r') as f:
            vocab = [line.strip() for line in f.readlines()]
        return vocab

    def load_data(self, path, fname):
        with open('%s/%s' % (path, fname)) as f:
            lines = [line.strip() for line in f.readlines()]
        data = []
        for line in lines:
            dict_tmp = eval(line)
            dict_tmp['sentence'] = dict_tmp['sentence'].lower().split()
            data.append(dict_tmp)
        return data

    def build_vocab(self, path, data):
        print("Creating vocabulary...")
        vocab = {}
        for pair in data:
            for token in pair['sentence']:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.voc_size:
            vocab_list = vocab_list[:self.voc_size]
        vocab_list.append('<unk>')

    @staticmethod
    def load_word_vec(vocab: List[str], filepath_word_vec: str = None, dim_word: int = 300):
        logger.info("Loading word vectors...")
        vectors = dict()
        if filepath_word_vec is not None:
            with open(filepath_word_vec) as f:
                for line in f:
                    s = line.strip()
                    word = s[:s.find(' ')]
                    vector = s[s.find(' ') + 1:]
                    vectors[word] = vector
        else:
            logger.warning('all word vec is inited by random')

        embed = []
        num_not_found, num_found = 0, 0
        for word in vocab:
            if word in vectors:
                vector = map(float, vectors[word].split())
                num_found = num_found + 1
            else:
                num_not_found = num_not_found + 1
                u = 1 / (np.sqrt(dim_word) + 1e-9)
                vector = np.random.uniform(-u, u, dim_word)
            embed.append(vector)
        logger.info('%s words found in vocab' % num_found)
        logger.info('%s words not found in vocab' % num_not_found)
        embed = np.array(embed, dtype=np.float32)
        return embed

    def gen_batched_data(self, data, flag_label_respresentation=2):
        '''
        flag_label_respresentation
        0, scalar output
        1, vector output, negative idx is 0, for cross entropy
        2, vector output, negative idx is -1, for hinge margin loss
        '''
        max_len_ = max([len(item['sentence']) for item in data])
        max_len = self.max_length if max_len_ > self.max_length else max_len_
        sentence, sentence_length, labels = [], [], []

        def padding(sent, l):
            return sent + ['_PAD'] * (l - len(sent))

        def scalar2vect(num, n_label):
            if flag_label_respresentation == 0:
                return num
            vect_re = [-1] * n_label if flag_label_respresentation == 2 else [0] * n_label
            vect_re[num] = 1
            return vect_re

        for item in data:
            if len(item['sentence']) < 1:
                print(item)
                exit()
            if len(item['sentence']) > max_len:
                sentence.append(item['sentence'][:max_len])
                sentence_length.append(max_len)
                labels.append(scalar2vect(item['label'], self.n_label))
            else:
                sentence.append(padding(item['sentence'], max_len))
                sentence_length.append(len(item['sentence']))
                labels.append(scalar2vect(item['label'], self.n_label))

        # sort by the length of sentence
        idx = np.argsort(sentence_length)[::-1]
        sentence = np.array(sentence)[idx]
        labels = np.array(labels)[idx]
        sentence_length = np.array(sentence_length)[idx]

        batched_data = {'sentence': sentence, 'labels': labels,
                        'sentence_length': sentence_length}
        return batched_data

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = []
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))
