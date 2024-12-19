# bert model packaging

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .huggingface.modeling import BertModel
from .huggingface.modeling import BertConfig
from .huggingface.modeling import BERTLayerNorm


class BertModelExtension(BertModel):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output, sequence_output


class PretrainedBertModel(nn.Module):
    def __init__(self, config, *unused_arg, **unused_kwargs):
        super(PretrainedBertModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, BERTLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def load_pretrained_bert_model(cls, bert_config_path, pretrained_model_path, *inputs, **kwargs):
        config = BertConfig.from_json_file(bert_config_path)
        model = cls(config, *inputs, **kwargs)
        # pretrained_model_weights = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        pretrained_model_weights = torch.load(pretrained_model_path)
        model.bert.load_state_dict(pretrained_model_weights)
        return model


class BertForSequence(PretrainedBertModel):
    def __init__(self, config):
        super(BertForSequence, self).__init__(config)
        self.bert = BertModelExtension(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output, sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        return pooled_output, sequence_output,

    @staticmethod
    def get_bert_encoder(bert_config_path, pretrained_model_path=None, use_pretrain=True):
        if use_pretrain:
            bert_model = BertForSequence.load_pretrained_bert_model(
                bert_config_path=bert_config_path,
                pretrained_model_path=pretrained_model_path,
            )
        else:
            bert_model = BertForSequence(
                BertConfig.from_json_file(bert_config_path))

        return bert_model
