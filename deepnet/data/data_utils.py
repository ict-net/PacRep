# Basic data operation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data_from_memory(lines, max_length_text, config_label):
    def tackle_one(input: dict):
        # init label
        label = {k: 0 for k in config_label['detail'].keys()}
        mask = np.zeros(config_label['n_task'])
        for k, v in input['label'].items():
            if v is None:
                continue
            if k in label:
                mask[int(k)] = 1
                label[k] = v
            else:
                pass
        dict_inst = {
            'tokens': input['text'][:max_length_text],
            'labels': label,
            'mask': mask
        }
        return dict_inst

    data = []
    for line in lines:
        try:
            line = json.loads(line.strip())
        except json.decoder.JSONDecodeError:
            pass
        else:
            data_one = {key: tackle_one(value) for key, value in line.items()}
            data.append(data_one)
    return data


def build_data_for_eva(list_text, max_length_text, config_label, sampled_num: int = -1):
    data = load_data_from_memory(list_text, max_length_text, config_label)
    sampled_num = len(list_text) if sampled_num == -1 else sampled_num
    data_with_label = [build_single_instance(tmp, config_label) for tmp in data]
    # data_with_label = {
    #     key: [build_single_instance_kernel(tmp) for tmp in value] for key, value in data.items()
    # }
    return data_with_label[:sampled_num]


def build_single_instance_kernel(item):
    item_data = {
        'bert_text': item['tokens'],
        'len_sen': len(item['tokens']),
        'labels': {key: torch.tensor(value, dtype=torch.long) for key, value in item['labels'].items()},
        'mask': torch.tensor(item['mask'], dtype=torch.long),
    }
    return item_data


def build_single_instance(input, config_label):
    data_with_label = {
        key: build_single_instance_kernel(value) for key, value in input.items()
    }
    return data_with_label


def text_collate_fn(data):
    """
    batched_data = {
        # 'sentences': pad_sequence([tmp['sentences'] for tmp in data], batch_first=True, padding_value=0),
        'bert_text': [tmp['bert_text'] for tmp in data],
        'len_sen': [tmp['len_sen'] for tmp in data],
        'labels': torch.stack([tmp['labels'] for tmp in data]),
        # 'labels': pad_sequence([tmp['labels'] for tmp in data], batch_first=True, padding_value=0),
    }
    """
    batched_data = dict()
    assert len(data) > 0, 'Please check data_utils.py'

    def collate_fn_one(input):
        batched_data_one = dict()
        for name in input[0].keys():
            if name != 'labels':
                batched_data_one[name] = [tmp[name] for tmp in input]
                if name == 'mask':
                    batched_data_one[name] = pad_sequence(batched_data_one[name])
            else:
                batched_data_one[name] = {
                    tmp: torch.stack([_tmp['labels'][tmp] for _tmp in input]) for tmp in input[0]['labels'].keys()
                }
        return batched_data_one

    dict_data = {k: [tmp[k] for tmp in data] for k, v in data[0].items()}

    for k, v in dict_data.items():
        batched_data[k] = collate_fn_one(v)
    return batched_data
