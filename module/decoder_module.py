import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention_module import MultiHeadAttention, GraphQueryAttention
from module.embedding_module import TransformerEmbedding
from module.ffn_module import TransformerFFN
from module.relative_position_bias import RelativePositionBias
from model.t5_config import T5Config

class TransformerDecoderLayer(nn.Module):
    """
    通用的 Transformer 解码器层
    
    支持多种注意力机制（MHA、GQA）和多种层归一化模式（Pre-LN、Post-LN、Sandwich-LN）。
    通过配置可以灵活地适配不同的模型架构（包括 T5）。
    
    设计特点：
    - 可配置的注意力机制
    - 支持相对位置偏置
    - 多种层归一化模式
    - 自注意力 + 交叉注意力的双注意力结构
    """
    def __init__(self, config, attention_type='mha'):
        super(TransformerDecoderLayer, self).__init__()
        
        # 确定隐藏层维度名称
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(config, 'd_model', None)
        assert hidden_size is not None, "config must have hidden_size or d_model"
        
        # 确定 dropout 率
        attention_dropout = getattr(config, 'attention_dropout', None)
        if attention_dropout is None:
            attention_dropout = getattr(config, 'attention_probs_dropout_prob', None)
        if attention_dropout is None:
            attention_dropout = getattr(config, 'dropout_rate', 0.1)
        
        resid_dropout = getattr(config, 'resid_dropout', None)
        if resid_dropout is None:
            resid_dropout = getattr(config, 'hidden_dropout_prob', None)
        if resid_dropout is None:
            resid_dropout = getattr(config, 'dropout_rate', 0.1)
        
        # 确定层归一化 epsilon
        layer_norm_eps = getattr(config, 'layer_norm_eps', None)
        if layer_norm_eps is None:
            layer_norm_eps = getattr(config, 'layer_norm_epsilon', 1e-6)
        
        # 选择注意力机制 - 自注意力
        if attention_type == 'gqa':
            self.attention = GraphQueryAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
        # 选择注意力机制 - 交叉注意力
        if attention_type == 'gqa':
            self.cross_attention = GraphQueryAttention(config)
        else:
            self.cross_attention = MultiHeadAttention(config)
        
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.cross_attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn_dropout = nn.Dropout(resid_dropout)

        self.ffn = TransformerFFN(config)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_dropout = nn.Dropout(resid_dropout)

        self.LN_mode = getattr(config, 'LN_mode', 'Post-LN')

    def forward(
        self, 
        dec_input, 
        enc_input=None, 
        self_attention_mask=None, 
        cross_attention_mask=None,
        self_attention_bias=None,
        cross_attention_bias=None
    ):
        """
        Transformer 解码器层前向传播
        
        参数：
        - dec_input: 解码器输入隐藏状态 [batch_size, seq_length, hidden_size]
        - enc_input: 编码器输出隐藏状态 [batch_size, enc_seq_length, hidden_size]
        - self_attention_mask: 自注意力掩码
        - cross_attention_mask: 交叉注意力掩码
        - self_attention_bias: 自注意力偏置（如相对位置偏置）
        - cross_attention_bias: 交叉注意力偏置
        
        返回：
        - 输出隐藏状态 [batch_size, seq_length, hidden_size]
        """
        # 默认用Post-LN，不在选项里也用Post-LN
        if self.LN_mode == 'Pre-LN':
            # Self-Attention
            residual_input = dec_input
            hidden_states = self.attn_norm(dec_input)
            hidden_states = self.attention(
                Q_input=hidden_states,
                attention_mask=self_attention_mask,
                attention_bias=self_attention_bias
            )
            h_state = self.attn_dropout(hidden_states) + residual_input

            # Cross-Attention
            if enc_input is not None:
                residual_input = h_state
                hidden_states = self.cross_attn_norm(h_state)
                hidden_states = self.cross_attention(
                    Q_input=hidden_states,
                    K_input=enc_input,
                    V_input=enc_input,
                    attention_mask=cross_attention_mask,
                    attention_bias=cross_attention_bias
                )
                h_state = self.cross_attn_dropout(hidden_states) + residual_input

            # FFN
            residual_input = h_state
            hidden_states = self.ffn_norm(h_state)
            hidden_states = self.ffn(hidden_states)
            output = self.ffn_dropout(hidden_states) + residual_input

        elif self.LN_mode == 'Sandwich-LN':
            # Self-Attention
            residual_input = dec_input
            hidden_states = self.attn_norm(dec_input)
            hidden_states = self.attention(
                Q_input=hidden_states,
                attention_mask=self_attention_mask,
                attention_bias=self_attention_bias
            )
            hidden_states = self.attn_dropout(hidden_states)
            h_state = self.attn_norm(hidden_states + residual_input)

            # Cross-Attention
            if enc_input is not None:
                residual_input = h_state
                hidden_states = self.cross_attn_norm(h_state)
                hidden_states = self.cross_attention(
                    Q_input=hidden_states,
                    K_input=enc_input,
                    V_input=enc_input,
                    attention_mask=cross_attention_mask,
                    attention_bias=cross_attention_bias
                )
                hidden_states = self.cross_attn_dropout(hidden_states)
                h_state = self.cross_attn_norm(hidden_states + residual_input)

            # FFN
            residual_input = h_state
            hidden_states = self.ffn_norm(h_state)
            hidden_states = self.ffn(hidden_states)
            hidden_states = self.ffn_dropout(hidden_states)
            output = self.ffn_norm(hidden_states + residual_input)
        else:
            # Self-Attention (Post-LN)
            residual_input = dec_input
            hidden_states = self.attention(
                Q_input=dec_input,
                attention_mask=self_attention_mask,
                attention_bias=self_attention_bias
            )
            hidden_states = self.attn_dropout(hidden_states)
            h_state = self.attn_norm(hidden_states + residual_input)

            # Cross-Attention
            if enc_input is not None:
                residual_input = h_state
                hidden_states = self.cross_attention(
                    Q_input=h_state,
                    K_input=enc_input,
                    V_input=enc_input,
                    attention_mask=cross_attention_mask,
                    attention_bias=cross_attention_bias
                )
                hidden_states = self.cross_attn_dropout(hidden_states)
                h_state = self.cross_attn_norm(hidden_states + residual_input)

            # FFN
            residual_input = h_state
            hidden_states = self.ffn(h_state)
            hidden_states = self.ffn_dropout(hidden_states)
            output = self.ffn_norm(hidden_states + residual_input)

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
            h_state = layer(h_state, enc_input, self_attention_mask=tri_mask, cross_attention_mask=padding_mask)

        return h_state


class T5Decoder(nn.Module):
    """
    T5 解码器
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        original_ln_mode = getattr(config, 'LN_mode', None)
        config.LN_mode = 'Pre-LN'
        
        num_layers = getattr(config, 'num_layers', 6)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, attention_type='gqa')
            for _ in range(num_layers)
        ])
        
        if original_ln_mode is not None:
            config.LN_mode = original_ln_mode
        
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        if hasattr(config, "relative_attention_num_buckets"):
            self.relative_position_bias = RelativePositionBias(
                num_buckets=config.relative_attention_num_buckets,
                max_distance=config.relative_attention_max_distance,
                num_heads=config.num_heads,
            )
        else:
            self.relative_position_bias = None

    def forward(self, input_ids, encoder_hidden_states, self_attention_mask, cross_attention_mask):
        h_state = self.embed_tokens(input_ids)
        h_state = self.dropout(h_state)
        
        _, seq_length = input_ids.size()
        enc_seq_length = encoder_hidden_states.size(1)

        for i, layer in enumerate(self.layers):
            self_position_bias = None
            cross_position_bias = None
            if i == 0 and self.relative_position_bias is not None:
                self_position_bias = self.relative_position_bias(
                    seq_length, seq_length, bidirectional=False, device=h_state.device
                )
                cross_position_bias = self.relative_position_bias(
                    seq_length, enc_seq_length, bidirectional=True, device=h_state.device
                )
            h_state = layer(
                dec_input=h_state,
                enc_input=encoder_hidden_states,
                self_attention_mask=extended_self_mask,
                cross_attention_mask=extended_cross_mask,
                self_attention_bias=self_position_bias,
                cross_attention_bias=cross_position_bias
            )
        
        h_state = self.final_layer_norm(h_state)
        h_state = self.dropout(h_state)
        
        return h_state