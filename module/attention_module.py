import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    selfAttnV1 公式版自注意力机制，实际上不会这样实现
    :param hidden_dim: dimension of hidden layer
"""
class selfAttnV1(nn.Module):
    def __init__(self, hidden_dim, config):
        super(selfAttnV1, self).__init__()

        self.h_dim = hidden_dim
        self.q_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.k_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.v_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.sqrt_h_dim = math.sqrt(self.h_dim)

    def forward(self, input):
        # input shape: [b_size, s_len, h_dim]
        # Q K V shape: [b_size, s_len, h_dim]
        Q = self.q_proj(input)
        K = self.k_proj(input)
        V = self.v_proj(input)

        # 转置:
        # 最后两个维度交换, K.transpose(-1, -2) == K.transpose(-2, -1)
        # 经过softmax的自注意力得分
        output = F.softmax(Q @ K.transpose(-1, -2) / self.sqrt_h_dim, dim=-1) @ V

        return output


"""
    selfAttnV2 带有mask和dropout的最基础实现
    :param hidden_dim: dimension of hidden layer
"""
class SelfAttnV2(nn.Module):
    def __init__(self, hidden_dim, config) -> None:
        super().__init__()

        self.h_dim = hidden_dim
        self.q_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.k_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.v_proj = nn.Linear(self.h_dim, self.h_dim, bias=False)

        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(config.attn_dropout_prob)

    def forward(self, input, attention_mask=None):
        # input shape: [b_size, s_len, h_dim]
        # Q K V shape: [b_size, s_len, h_dim]
        Q = self.q_proj(input)
        K = self.k_proj(input)
        V = self.v_proj(input)

        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor

        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # 对attn_scores做dropout似乎是一种不太稳定的做法, 但主流模型都在这么做
        output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V

        return output


"""
    SelfAttnVX 基于QKV大矩阵的工程优化
    优点: 将Q、K、V三个矩阵合并成一个大矩阵, 加速训练过程中的矩阵计算
    缺点: 在MQA、GQA、MLA中Q和K、V矩阵的数量不一致，进而导致维度大小不一致。
    综合: 因此在现在这个时间点,已经不太能算作是优化了
    :param hidden_dim: dimension of hidden layer
"""
class SelfAttnV3(nn.Module):
    def __init__(self, hidden_dim, config) -> None:
        super().__init__()

        self.h_dim = hidden_dim
        self.proj = nn.Linear(self.h_dim, self.h_dim * 3, bias=False)

        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(config.attn_dropout_prob)

    def forward(self, input, attention_mask=None):
        # input shape: [b_size, s_len, h_dim]
        # QKV shape: [b_size, s_len, h_dim * 3]
        QKV = self.proj(input)

        # Q, K, V shape:  [b_size, s_len, h_dim]
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        attn_scores = Q @ K.transpose(-1, -2) / self.scale_factor

        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # 对attn_scores做dropout似乎是一种不太稳定的做法, 但主流模型都在这么做
        output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V

        return output

"""
    MultiHeadAttention 多头注意力机制
    :param hidden_dim: dimension of hidden layer
"""
# MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nums_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.h_dim = config.hidden_size
        self.scale_factor = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.h_dim, self.head_dim * config.num_attention_heads, bias=False)
        self.k_proj = nn.Linear(self.h_dim, self.head_dim * config.num_attention_heads, bias=False)
        self.v_proj = nn.Linear(self.h_dim, self.head_dim * config.num_attention_heads, bias=False)
        self.o_proj = nn.Linear(self.head_dim * config.num_attention_heads, self.h_dim)
        nn.MultiheadAttention

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None):
        if K_input is None:
            K_input = Q_input
        if V_input is None:
            V_input = K_input
        # input shape: [b_size, s_len, h_dim]
        # attention_mask shape: [b_size, s_len]
        b_size, s_len, _ = Q_input.shape

        # input映射成QKV
        # Q, K, V shape:  [b_size, s_len, nums_head * head_dim]
        Q = self.q_proj(Q_input)
        K = self.k_proj(K_input)
        V = self.v_proj(V_input)

        # QKV重组成多头模式
        # view(b_size, s_len, self.nums_head, self.h_dim) shape: [b_size, s_len, nums_head, head_dim]
        # view(b_size, s_len, self.nums_head, self.h_dim).transpose(1,2) shape: [b_size, nums_head, s_len, head_dim]
        # 相当于每一个head，处理s_len/nums_head个token
        Q = Q.view(b_size, s_len, self.nums_head, self.head_dim).transpose(1,2)
        K = K.view(b_size, s_len, self.nums_head, self.head_dim).transpose(1,2)
        V = V.view(b_size, s_len, self.nums_head, self.head_dim).transpose(1,2)

        # attn_scores shape: [b_size, nums_head, s_len, s_len]
        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # self.attn_drop(torch.softmax(attn_scores, dim=-1)) shape: [b_size, nums_head, s_len, s_len]
        # V shape: [b_size, nums_head, s_len, head_dim]
        # head_output shape: [b_size, nums_head, s_len, head_dim]
        head_output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V
        # head_output.transpose(1, 2) shape: [b_size, s_len, nums_head, head_dim]
        # head_output.transpose(1, 2).reshape(b_size, s_len, -1) shape: [b_size, s_len, nums_head * head_dim] = [b_size, s_len, nums_head * head_dim]
        output = head_output.transpose(1, 2).reshape(b_size, s_len, -1)
        # output shape: [b_size, s_len, h_dim]
        output = self.o_proj(output)

        return output

"""
    MutilQueryAttention 多查询注意力机制（N个Q，1个K、V）
    :param hidden_dim: dimension of hidden layer
"""
# MQA
class MutilQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )

        assert config.num_attention_heads % config.num_key_value_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_dim // self.num_attention_heads
        self.n_rep = self.num_attention_heads
        self.scale_factor = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_dim, bias=False)

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None):
        if K_input is None:
            K_input = Q_input
        if V_input is None:
            V_input = K_input

        b_size, s_len, _ = Q_input.shape
        # qkv projection
        Q = self.q_proj(Q_input)  # （batch, seq, hidden_dim)
        K = self.k_proj(K_input)
        V = self.v_proj(V_input)

        Q = Q.view(b_size, s_len, self.num_attention_heads, self.head_dim)
        K = K.view(b_size, s_len, 1, self.head_dim)
        V = V.view(b_size, s_len, 1, self.head_dim)

        # 中间要加位置编码

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # k v repeat； （广播操作）
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # attn_scores shape: [b_size, nums_head, s_len, s_len]
        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # self.attn_drop(torch.softmax(attn_scores, dim=-1)) shape: [b_size, nums_head, s_len, s_len]
        # V shape: [b_size, nums_head, s_len, head_dim]
        # head_output shape: [b_size, nums_head, s_len, head_dim]
        head_output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V
        # head_output.transpose(1, 2) shape: [b_size, s_len, nums_head, head_dim]
        # head_output.transpose(1, 2).reshape(b_size, s_len, -1) shape: [b_size, s_len, nums_head * head_dim] = [b_size, s_len, nums_head * head_dim]
        output = head_output.transpose(1, 2).reshape(b_size, s_len, -1)
        # output shape: [b_size, s_len, h_dim]
        output = self.o_proj(output)

        return output

"""
    GraphQueryAttention 分组查询注意力机制（N个Q，M个K、V）
    M=1时为，MQA
    N=M时为，MHA
    :param hidden_dim: dimension of hidden layer
"""
# GQA
class GraphQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        hidden_dim = getattr(config, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(config, "hidden_size", None)
        assert hidden_dim is not None, "config must have hidden_dim or hidden_size"

        assert config.num_attention_heads % self.num_key_value_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = hidden_dim // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.scale_factor = 1 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_dim, bias=False)
        
        self.attn_dropout = getattr(config, "attention_dropout", 0.1)
        self.attn_drop = nn.Dropout(self.attn_dropout)

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None, attention_bias=None):
        """
        Args:
            Q_input: Query 输入，shape [batch_size, seq_len, hidden_dim]
            K_input: Key 输入，shape [batch_size, seq_len, hidden_dim]
            V_input: Value 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
            attention_bias: 注意力偏置（如相对位置偏置），shape [num_heads, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
        """
        if K_input is None:
            K_input = Q_input
        if V_input is None:
            V_input = K_input

        b_size, s_len, _ = Q_input.shape
        # qkv projection
        Q = self.q_proj(Q_input)  # （batch, seq, hidden_dim)
        K = self.k_proj(K_input)
        V = self.v_proj(V_input)

        Q = Q.view(b_size, s_len, self.num_attention_heads, self.head_dim)
        K = K.view(b_size, s_len, self.num_key_value_heads, self.head_dim)
        V = V.view(b_size, s_len, self.num_key_value_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # k v repeat； （广播操作）
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # attn_scores shape: [b_size, nums_head, s_len, s_len]
        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor
        
        # 添加注意力偏置（如相对位置偏置）
        if attention_bias is not None:
            if attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(0)
            attn_scores = attn_scores + attention_bias
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # self.attn_drop(torch.softmax(attn_scores, dim=-1)) shape: [b_size, nums_head, s_len, s_len]
        # V shape: [b_size, nums_head, s_len, head_dim]
        # head_output shape: [b_size, nums_head, s_len, head_dim]
        head_output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V
        # head_output.transpose(1, 2) shape: [b_size, s_len, nums_head, head_dim]
        # head_output.transpose(1, 2).reshape(b_size, s_len, -1) shape: [b_size, s_len, nums_head * head_dim] = [b_size, s_len, nums_head * head_dim]
        output = head_output.transpose(1, 2).reshape(b_size, s_len, -1)
        # output shape: [b_size, s_len, h_dim]
        output = self.o_proj(output)

        return output

