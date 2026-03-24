import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention_module import MultiHeadAttention
from module.embedding_module import TransformerEmbedding
from module.ffn_module import TransformerFFN

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        self.cross_attention = MultiHeadAttention(config)
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.cross_attn_dropout = nn.Dropout(config.resid_dropout)

        self.ffn = TransformerFFN(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, enc_input, dec_input, padding_mask, tri_mask):
        residual_input = dec_input
        h_state = self.attn_norm(self.attn_dropout(self.attention(dec_input, attention_mask=tri_mask)) + residual_input)

        if enc_input is not None:
            residual_input = h_state
            h_state = self.cross_attn_norm(self.cross_attn_dropout(self.cross_attention(h_state,enc_input,enc_input,padding_mask)) + residual_input)

        residual_input = h_state
        output = self.ffn_norm(self.ffn_dropout(self.ffn(h_state)) + residual_input)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.embedding = TransformerEmbedding(config)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_encoder_layers)]
        )

    def forward(self, enc_input, dec_input, padding_mask, tri_mask):
        h_state = self.embedding(dec_input)

        for layer in self.layers:
            h_state = layer(enc_input,h_state,padding_mask,tri_mask)

        return h_state