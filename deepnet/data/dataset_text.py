# Basic data operation

import torch
import logging
from torch.utils.data.distributed import DistributedSampler
from .data_utils import load_data_from_memory, build_single_instance


logger = logging.getLogger(__name__)


class TextDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching."""
    def __init__(self, filepath, chunk_size, max_len_text, config_label):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.file_object = open(self.filepath, 'r')
        self.max_len_text = max_len_text
        self.config_label = config_label
        self.read_count = 1
        self.__reset__()

    def __getitem__(self, index):
        _item = self.data_chunk[index]
        item = build_single_instance(_item, self.config_label)
        return item

    def __len__(self):
        return len(self.data_chunk)

    def __reset__(self):
        data = self.file_object.readlines(self.chunk_size)
        if not data:
            self.file_object.close()
            self.file_object = open(self.filepath, 'r')
            data = self.file_object.readlines(self.chunk_size)
            self.read_count = self.read_count + 1
            logger.info('The count of reading file %s is %s.' % (self.filepath, self.read_count))
        self.data_chunk = load_data_from_memory(data, self.max_len_text, self.config_label)


class IterTextDataset:
    def __init__(self, filepath, chunk_size, config, config_label, tackle_data=None, num_workers=4, shuffle=True,
                 use_distributed=True):
        self.dataset = TextDataset(filepath, chunk_size, config.max_length_sen, config_label)
        self.batch_size = config.batch_size
        self.tackle_data = tackle_data
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.use_distributed = use_distributed
        self.__reset__()
        if self.use_distributed:
            logger.info('Use distributed data sampler.')
        else:
            logger.info('Use general data sampler.')

    def __reset__(self):
        '''
        if the data in training data is not shuffled, make the parameter shuffle True
        '''
        # self.count_inst = self.dataset.__len__()
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=self.shuffle if not self.use_distributed else None,
            collate_fn=self.tackle_data,
            sampler=DistributedSampler(self.dataset) if self.use_distributed else None,
        )
        # self.data_loader_iter = iter(self.data_loader)
        self.data_loader_iter = self.data_loader.__iter__()

    def get_data(self):
        try:
            data = self.data_loader_iter.next()
        except StopIteration:
            self.dataset.__reset__()
            self.__reset__()
            data = self.data_loader_iter.next()
            logger.info('Read next training chunk, data size is %s' % self.dataset.__len__())
            # logger.info('The 1st: %s' % str(self.dataset.data_chunk[0]))
        return data

    def reload_dataset(self):
        self.dataset.__reset__()
        self.__reset__()
