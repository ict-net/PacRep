# dataset operations for word bert

import numpy as np
import torch
from .tokenization_word import WordFullTokenizer


def text_classification_collate_fn(tokens, padding=0, device=torch.device('cpu')):
    """The collate function for text classification dataset.
    Args:
        batch: the list of different instance
        padding: padding token index
    Return:
        The dict contains bert inputs and labels for text classification, each
        element is batched.
        :param device:
    """
    segment_ids = [[0] * len(token) for token in tokens]
    attn_masks = [[1] * len(token) for token in tokens]
    max_len = max([len(token) for token in tokens])
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long, device=device)
    return {"tokens": tokens, "segment_ids": segment_ids, "attn_masks": attn_masks}


class WordBertDataset():
    def __init__(self, vocab_file, max_seq_len, is_array, do_lower_case=True, *inputs, **kwargs):
        super(WordBertDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = WordFullTokenizer(
            vocab_file=vocab_file,
            is_array=is_array,
            do_lower_case=do_lower_case
        )
        self.max_seq_len = max_seq_len

    def get_idx(self, text):
        """
        Data structure:
        w1 w2 w3 wn
        """
        tokens, lens = self.tokenizer.tokenize_with_length(text)
        tokens = tokens[: self.max_seq_len]
        tokens = ["[CLS]"] + tokens
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # trunk lens
        sum_len = 1
        lens_trunk = []
        for i, len_item in enumerate(lens):
            if sum_len <= (len(tokens) - len_item):
                lens_trunk.append(len_item)
            else:
                # only trunk full word, if not, add the next one
                # lens_trunk.append(len(tokens) - sum_len)
                break
            sum_len = sum_len + len_item
        lens_trunk = np.array(lens_trunk)
        tokens = tokens[:sum_len]
        assert len(tokens) == np.sum(lens_trunk) + 1
        return tokens, lens_trunk

    def get_batched_data(self, list_text, device):
        list_tokens, list_lens = [], []
        for text in list_text:
            tokens, lens = self.get_idx(text)
            list_tokens.append(tokens)
            list_lens.append(lens)
        # list_tokens = [self.get_idx(text) for text in list_text]
        dict_re = text_classification_collate_fn(list_tokens, device=device)
        return dict_re, list_lens
