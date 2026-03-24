import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention_module import MultiHeadAttention
from module.embedding_module import TransformerEmbedding
from module.ffn_module import TransformerFFN


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        self.ffn = TransformerFFN(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_dropout = nn.Dropout(config.resid_dropout)

        self.LN_mode = config.LN_mode

    def forward(self, hidden_states, padding_mask):
        # 默认用Post-LN，不在选项里也用Post-LN
        if self.LN_mode == 'Pre-LN':
            residual_states = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attention(hidden_states, attention_mask=padding_mask)
            hidden_states = self.attn_dropout(hidden_states) + residual_states

            residual_states = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            output = self.ffn_dropout(hidden_states) + residual_states

        elif self.LN_mode == 'Sandwich-LN':
            residual_states = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attention(hidden_states, attention_mask=padding_mask)
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = self.attn_norm(hidden_states + residual_states)

            residual_states = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            hidden_states = self.ffn_dropout(hidden_states)
            output = self.ffn_norm(hidden_states + residual_states)
        else:
            residual_states = hidden_states
            hidden_states = self.attention(hidden_states, attention_mask=padding_mask)
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = self.attn_norm(hidden_states + residual_states)

            residual_states = hidden_states
            hidden_states = self.ffn(hidden_states)
            hidden_states = self.ffn_dropout(hidden_states)
            output = self.ffn_norm(hidden_states + residual_states)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embedding = TransformerEmbedding(config)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )

    def forward(self, input, padding_mask):
        h_state = self.embedding(input)

        for layer in self.layers:
            h_state = layer(h_state, padding_mask)

        return h_state
