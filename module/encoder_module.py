import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention_module import MultiHeadAttention, GraphQueryAttention
from module.embedding_module import TransformerEmbedding
from module.ffn_module import TransformerFFN
from module.relative_position_bias import RelativePositionBias
from model.t5_config import T5Config


class TransformerEncoderLayer(nn.Module):
    """
    通用的 Transformer 编码器层
    
    支持多种注意力机制（MHA、GQA）和多种层归一化模式（Pre-LN、Post-LN、Sandwich-LN）。
    通过配置可以灵活地适配不同的模型架构（包括 T5）。
    
    设计特点：
    - 可配置的注意力机制
    - 支持相对位置偏置
    - 多种层归一化模式
    """
    def __init__(self, config, attention_type='mha'):
        super(TransformerEncoderLayer, self).__init__()
        
        # 选择注意力机制
        if attention_type == 'gqa':
            self.attention = GraphQueryAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
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
        
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.ffn = TransformerFFN(config)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_dropout = nn.Dropout(resid_dropout)

        self.LN_mode = getattr(config, 'LN_mode', 'Post-LN')

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        """
        Transformer 编码器层前向传播
        
        参数：
        - hidden_states: 输入隐藏状态 [batch_size, seq_length, hidden_size]
        - attention_mask: 注意力掩码 [batch_size, seq_length] 或扩展后的 4D 掩码
        - attention_bias: 注意力偏置（如相对位置偏置）
        
        返回：
        - 输出隐藏状态 [batch_size, seq_length, hidden_size]
        """
        # 默认用Post-LN，不在选项里也用Post-LN
        if self.LN_mode == 'Pre-LN':
            residual_states = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attention(
                Q_input=hidden_states,
                attention_mask=attention_mask,
                attention_bias=attention_bias
            )
            hidden_states = self.attn_dropout(hidden_states) + residual_states

            residual_states = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            output = self.ffn_dropout(hidden_states) + residual_states

        elif self.LN_mode == 'Sandwich-LN':
            residual_states = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attention(
                Q_input=hidden_states,
                attention_mask=attention_mask,
                attention_bias=attention_bias
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = self.attn_norm(hidden_states + residual_states)

            residual_states = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            hidden_states = self.ffn_dropout(hidden_states)
            output = self.ffn_norm(hidden_states + residual_states)
        else:
            residual_states = hidden_states
            hidden_states = self.attention(
                Q_input=hidden_states,
                attention_mask=attention_mask,
                attention_bias=attention_bias
            )
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
            h_state = layer(h_state, attention_mask=padding_mask)

        return h_state


class T5Encoder(nn.Module):
    """
    T5 编码器
    
    简洁实现，仅支持 input_ids 输入，参考 TransformerEncoder 的设计。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        original_ln_mode = getattr(config, 'LN_mode', None)
        config.LN_mode = 'Pre-LN'
        
        num_layers = getattr(config, 'num_layers', 6)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config, attention_type='gqa')
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

    def forward(self, input_ids, attention_mask):
        h_state = self.embed_tokens(input_ids)
        h_state = self.dropout(h_state)
        
        _, seq_length = input_ids.size()
        
        for i, layer in enumerate(self.layers):
            position_bias = None
            if i == 0 and self.relative_position_bias is not None:
                position_bias = self.relative_position_bias(seq_length, seq_length, bidirectional=True, device=h_state.device)
            h_state = layer(h_state, attention_mask=attention_mask, attention_bias=position_bias)
        
        h_state = self.final_layer_norm(h_state)
        h_state = self.dropout(h_state)
        
        return h_state