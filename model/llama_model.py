from model.llama_config import LlamaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union
import math
from torch.nn import CrossEntropyLoss, MSELoss
from module.ffn_module import SwiGLUFFN
from module.attention_module import GraphQueryAttention
from module.embedding_module import RotaryPositionalEmbedding
from module.norm_module import RMSNorm


class LlamaBlock(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = GraphQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
            self,
            x,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.input_layernorm(x)
        attn_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = x + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.ffn(hidden_states)
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [LlamaBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )

        return output


class LlamaForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.model = LlamaModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = hidden_states[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output


class LlamaForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.model = LlamaModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output
