# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# inherit tokenization, and advanced

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np

from .huggingface.tokenization import FullTokenizer, BasicTokenizer, WordpieceTokenizer
from .huggingface.tokenization import convert_to_unicode, whitespace_tokenize, load_vocab
from .huggingface.tokenization import _is_control, _is_whitespace


class WordFullTokenizer(FullTokenizer):
    def __init__(self, vocab_file, is_array, do_lower_case=True):
        super().__init__(vocab_file, do_lower_case)
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = WordBasicTokenizer(do_lower_case=do_lower_case, is_array=is_array)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize_with_length(self, text):
        split_tokens = []
        tokens, len_tokens = self.basic_tokenizer.tokenize(text)
        len_tokens_shift = len_tokens.copy()

        for idx, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                split_tokens.append(sub_token)
            if len(sub_tokens) > 1:
                def find_idx(lens, num):
                    for i, len in enumerate(lens):
                        num = num - len
                        if num < 0:
                            return i
                    return -1
                idx_len = find_idx(len_tokens, idx)
                len_tokens_shift[idx_len] = len_tokens_shift[idx_len] + len(sub_tokens) - 1
        # assert len(len_tokens_shift) == len(text.split())
        if len(split_tokens) != np.sum(len_tokens_shift):
            print(text)
            print(split_tokens)
            print(len_tokens_shift, np.sum(len_tokens_shift), len(split_tokens))
        assert len(split_tokens) == np.sum(len_tokens_shift)
        return split_tokens, len_tokens_shift


class WordBasicTokenizer(BasicTokenizer):
    def __init__(self, do_lower_case, is_array: bool):
        super().__init__(do_lower_case)
        self.is_array = is_array

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        if not self.is_array:
            text = convert_to_unicode(text)
            text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text, lens_word = self._word_tokenize_chinese_chars(text)
        len_tokens_shift = lens_word.copy()
        # text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for idx, token in enumerate(orig_tokens):
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            tokens_split_punc = self._run_split_on_punc(token)
            if len(tokens_split_punc) > 1:
                def find_idx(lens, num):
                    for i, len_tmp in enumerate(lens):
                        num = num - len_tmp
                        if num < 0:
                            return i
                    return -1
                split_tokens.extend(tokens_split_punc)
                idx_len = find_idx(lens_word, idx)
                # len_tokens_shift[idx_len] = lens_word[idx_len] + len(tokens_split_punc) - 1
                len_tokens_shift[idx_len] = len_tokens_shift[idx_len] + len(tokens_split_punc) - 1
            elif len(tokens_split_punc) == 1:
                split_tokens.extend(tokens_split_punc)
            else:
                split_tokens.extend(['[UNK]'])

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        assert len(split_tokens) == np.sum(len_tokens_shift)
        return output_tokens, len_tokens_shift

    def _word_tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        if self.is_array:
            list_word = text
        else:
            list_word = text.split(' ')
        list_splited_word = []
        for i, word in enumerate(list_word):
            word_tmp = []
            for char in word:
                cp = ord(char)
                if self._is_chinese_char(cp):
                    word_tmp.append(f' {char} ')
                else:
                    word_tmp.append(char)
            list_splited_word.append(''.join(word_tmp))
        return ' '.join(list_splited_word), np.array([len(tmp.split()) for tmp in list_splited_word])

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        str_cleaned = "".join(output)
        # clean the VARIATION SELECTOR, which exist(s) after emoji
        re_VARIATION_SELECTOR = re.compile(
            u'['
            u'\ufe00-\ufe0f]+',
            re.UNICODE)
        return re_VARIATION_SELECTOR.sub('', str_cleaned)