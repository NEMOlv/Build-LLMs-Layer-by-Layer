import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention_module import MultiHeadAttention, GraphQueryAttention
from module.embedding_module import TransformerEmbedding, BartEmbedding, RelativePositionEmbedding
from module.ffn_module import TransformerFFN
from model.t5_config import T5Config


class UniversalDecoder(nn.Module):
    """
    通用解码器基类
    
    支持多种配置的解码器实现，可以通过参数配置：
    - 嵌入层（可自定义）
    - 注意力类型（mha/gqa）
    - 层数
    - 是否使用相对位置偏置
    - Pre-LN 架构（强制）
    - final_layer_norm 和 dropout
    """
    def __init__(
        self,
        config,
        embedding_layer,
        attention_type='mha',
        num_layers_key='num_layers',
        use_relative_position_bias=False,
        dropout_after_embedding=False
    ):
        """
        Args:
            config: 模型配置对象
            embedding_layer: 嵌入层模块（如 nn.Embedding 或自定义 Embedding）
            attention_type: 注意力类型，'mha' 或 'gqa'
            num_layers_key: 从 config 中获取层数的键名
            use_relative_position_bias: 是否使用相对位置偏置
            dropout_after_embedding: 是否在嵌入后立即应用 dropout
        """
        super().__init__()
        self.config = config
        self.embedding = embedding_layer
        self.dropout_after_embedding = dropout_after_embedding
        
        original_ln_mode = getattr(config, 'LN_mode', None)
        config.LN_mode = 'Pre-LN'
        
        num_layers = getattr(config, num_layers_key, 6)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, attention_type=attention_type)
            for _ in range(num_layers)
        ])
        
        if original_ln_mode is not None:
            config.LN_mode = original_ln_mode
        
        hidden_size = getattr(config, 'd_model', None)
        if hidden_size is None:
            hidden_size = getattr(config, 'hidden_size', None)
        
        layer_norm_eps = getattr(config, 'layer_norm_eps', None)
        if layer_norm_eps is None:
            layer_norm_eps = getattr(config, 'layer_norm_epsilon', 1e-6)
        
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
        
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.use_relative_position_bias = use_relative_position_bias
        if use_relative_position_bias and hasattr(config, "relative_attention_num_buckets"):
            self.relative_position_bias = RelativePositionEmbedding(
                num_buckets=config.relative_attention_num_buckets,
                max_distance=config.relative_attention_max_distance,
                num_heads=getattr(config, 'num_heads', config.num_attention_heads),
            )
        else:
            self.relative_position_bias = None
    
    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        self_attention_mask=None,
        cross_attention_mask=None
    ):
        """
        Args:
            input_ids: 解码器输入 token id 序列，shape [batch_size, seq_len]
            encoder_hidden_states: 编码器输出，shape [batch_size, enc_seq_len, hidden_size]
            self_attention_mask: 自注意力掩码，shape [batch_size, seq_len]
            cross_attention_mask: 交叉注意力掩码，shape [batch_size, enc_seq_len]
        Returns:
            decoder_hidden_states: 解码器输出，shape [batch_size, seq_len, hidden_size]
        """
        h_state = self.embedding(input_ids)
        
        if self.dropout_after_embedding:
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
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                self_attention_bias=self_position_bias,
                cross_attention_bias=cross_position_bias
            )
        
        h_state = self.final_layer_norm(h_state)
        h_state = self.dropout(h_state)
        
        return h_state

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


class T5Decoder(UniversalDecoder):
    """
    T5 解码器
    
    继承自 UniversalDecoder，使用 GQA 注意力和相对位置编码。
    """
    def __init__(self, config):
        embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        super().__init__(
            config=config,
            embedding_layer=embed_tokens,
            attention_type='gqa',
            num_layers_key='num_layers',
            use_relative_position_bias=True,
            dropout_after_embedding=True
        )

class BartDecoder(UniversalDecoder):
    """
    BART 解码器
    
    继承自 UniversalDecoder，使用 MHA 注意力和绝对位置编码。
    """
    def __init__(self, config):
        embedding = BartEmbedding(config)
        super().__init__(
            config=config,
            embedding_layer=embedding,
            attention_type='mha',
            num_layers_key='num_decoder_layers',
            use_relative_position_bias=False,
            dropout_after_embedding=False
        )