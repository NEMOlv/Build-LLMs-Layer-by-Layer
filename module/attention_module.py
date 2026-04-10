import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.embedding_module import RotaryPositionalEmbedding


"""
    selfAttnV1 公式版自注意力机制，实际上不会这样实现
    :param hidden_dim: dimension of hidden layer
"""
class selfAttnV1(nn.Module):
    def __init__(self, hidden_dim):
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
    SelfAttnV2 带有mask和dropout的最基础实现
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

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        
        # 初始化旋转位置编码
        self.RotaryPositionalEmbedding = None
        if getattr(config, "isRotaryPositional", False):
            max_pos_emb = getattr(config, "max_position_embeddings", 4096)
            rope_base = getattr(config, "rope_base", 10000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            rotate_type = getattr(config, "rotate_type", "rotate_interval")
            self.RotaryPositionalEmbedding = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_pos_emb,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
                rotate_type=rotate_type
            )

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None):
        """
        Args:
            Q_input: Query 输入，shape [batch_size, seq_len, hidden_dim]
            K_input: Key 输入，shape [batch_size, seq_len, hidden_dim]
            V_input: Value 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
        """
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
        Q = Q.view(b_size, s_len, self.nums_head, self.head_dim)
        K = K.view(b_size, s_len, self.nums_head, self.head_dim)
        V = V.view(b_size, s_len, self.nums_head, self.head_dim)

        # 应用旋转位置编码（如果初始化了）
        if self.RotaryPositionalEmbedding is not None:
            Q, K = self.RotaryPositionalEmbedding(Q, K, unsqueeze_dim=1)

        # view(b_size, s_len, self.nums_head, self.h_dim).transpose(1,2) shape: [b_size, nums_head, s_len, head_dim]
        # 相当于每一个head，处理s_len/nums_head个token
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

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
        
        self.attn_dropout = getattr(config, "attention_dropout", 0.1)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        
        # 初始化旋转位置编码
        self.RotaryPositionalEmbedding = None
        if getattr(config, "isRotaryPositional", False):
            max_pos_emb = getattr(config, "max_position_embeddings", 4096)
            rope_base = getattr(config, "rope_base", 10000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            rotate_type = getattr(config, "rotate_type", "rotate_interval")
            self.RotaryPositionalEmbedding = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_pos_emb,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
                rotate_type=rotate_type
            )

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None):
        """
        Args:
            Q_input: Query 输入，shape [batch_size, seq_len, hidden_dim]
            K_input: Key 输入，shape [batch_size, seq_len, hidden_dim]
            V_input: Value 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
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
        K = K.view(b_size, s_len, 1, self.head_dim)
        V = V.view(b_size, s_len, 1, self.head_dim)

        # 应用旋转位置编码（如果初始化了）
        if self.RotaryPositionalEmbedding is not None:
            Q, K = self.RotaryPositionalEmbedding(Q, K, unsqueeze_dim=1)

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
        
        # 初始化旋转位置编码
        self.RotaryPositionalEmbedding = None
        if getattr(config, "isRotaryPositional", False):
            max_pos_emb = getattr(config, "max_position_embeddings", 4096)
            rope_base = getattr(config, "rope_base", 10000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            rotate_type = getattr(config, "rotate_type", "rotate_interval")
            self.RotaryPositionalEmbedding = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_pos_emb,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
                rotate_type=rotate_type
            )

    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None, attention_bias=None, past_key_values=None, use_cache=False):
        """
        Args:
            Q_input: Query 输入，shape [batch_size, seq_len, hidden_dim]
            K_input: Key 输入，shape [batch_size, seq_len, hidden_dim]
            V_input: Value 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
            attention_bias: 注意力偏置（如相对位置偏置），shape [num_heads, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
            past_key_values: 之前的 KV 缓存，tuple 形式 (past_key, past_value)
            use_cache: 是否使用 KV 缓存
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

        # 应用旋转位置编码（如果初始化了）
        if self.RotaryPositionalEmbedding is not None:
            # 如果有 past_key_values，需要计算当前 token 的位置
            if past_key_values is not None:
                past_len = past_key_values[0].shape[2]
                Q, K = self.RotaryPositionalEmbedding(Q, K, unsqueeze_dim=1, offset=past_len)
            else:
                Q, K = self.RotaryPositionalEmbedding(Q, K, unsqueeze_dim=1)

        # 拼接 KV 缓存
        if past_key_values is not None:
            past_key, past_value = past_key_values
            # past_key shape: [batch, num_kv_heads, past_len, head_dim]
            # K shape: [batch, curr_len, num_kv_heads, head_dim] -> 转换为 [batch, num_kv_heads, curr_len, head_dim]
            K = torch.cat([past_key, K.transpose(1, 2)], dim=2)
            V = torch.cat([past_value, V.transpose(1, 2)], dim=2)
        else:
            # 没有缓存，直接转置
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

        # 保存新的 KV 缓存
        present_key_values = None
        if use_cache:
            present_key_values = (K, V)

        Q = Q.transpose(1, 2)

        # k v repeat； （广播操作）
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # attn_scores shape: [b_size, nums_head, s_len, total_len]
        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor
        
        # 添加注意力偏置（如相对位置偏置）
        if attention_bias is not None:
            if attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(0)
            # 如果有 past_key_values，需要截取或扩展 attention_bias
            if past_key_values is not None:
                total_len = K.shape[-2]
                # 注意：这里需要根据具体情况处理 attention_bias 的维度
                # 简单起见，只使用与当前序列长度匹配的部分
                pass
            attn_scores = attn_scores + attention_bias
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # self.attn_drop(torch.softmax(attn_scores, dim=-1)) shape: [b_size, nums_head, s_len, total_len]
        # V shape: [b_size, nums_head, total_len, head_dim]
        # head_output shape: [b_size, nums_head, s_len, head_dim]
        head_output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V
        # head_output.transpose(1, 2) shape: [b_size, s_len, nums_head, head_dim]
        # head_output.transpose(1, 2).reshape(b_size, s_len, -1) shape: [b_size, s_len, nums_head * head_dim] = [b_size, s_len, nums_head * head_dim]
        output = head_output.transpose(1, 2).reshape(b_size, s_len, -1)
        # output shape: [b_size, s_len, h_dim]
        output = self.o_proj(output)

        if use_cache:
            return output, present_key_values
        else:
            return output


"""
    Multi-head Latent Attention (MLA) 多头潜在注意力机制
    DeepSeek V3 提出的创新注意力机制
    
    核心思想：
        1. 将 KV 缓存压缩到低维潜在空间
        2. 使用两个投影层：
           - kv_b_proj: 将原始 KV 压缩到潜在空间（降维）
           - kv_a_proj: 将潜在表示投影回注意力空间（升维）
        3. 大幅减少 KV 缓存的内存占用
        4. 保持良好的性能
    
    优势：
        - 减少 KV 缓存内存占用（通常压缩到 1/4 或更少）
        - 保持推理速度
        - 几乎没有性能损失
"""
# MLA
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 获取隐藏层维度
        hidden_dim = getattr(config, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(config, "hidden_size", None)
        assert hidden_dim is not None, "config must have hidden_dim or hidden_size"
        
        self.hidden_dim = hidden_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        
        # 确保 attention heads 能被 key-value heads 整除
        assert self.num_attention_heads % self.num_key_value_heads == 0
        
        # 潜在维度（latent dimension），通常是 hidden_dim 的 1/4
        self.latent_dim = getattr(config, "latent_dim", hidden_dim // 4)
        self.head_dim = hidden_dim // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        
        # Q 投影（保持不变）
        self.q_proj = nn.Linear(hidden_dim, self.num_attention_heads * self.head_dim, bias=False)
        
        # KV 投影到潜在空间（降维）
        self.kv_b_proj = nn.Linear(hidden_dim, 2 * self.latent_dim, bias=False)
        
        # 从潜在空间投影回注意力空间（升维）
        # 包含两个独立的投影：k_proj 和 v_proj
        self.kv_a_proj = nn.Linear(self.latent_dim, 2 * self.num_key_value_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_dim, bias=False)
        
        # 注意力 dropout
        self.attn_dropout = getattr(config, "attention_dropout", 0.1)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        
        # 初始化旋转位置编码
        self.RotaryPositionalEmbedding = None
        if getattr(config, "isRotaryPositional", False):
            max_pos_emb = getattr(config, "max_position_embeddings", 4096)
            rope_base = getattr(config, "rope_base", 10000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            rotate_type = getattr(config, "rotate_type", "rotate_interval")
            self.RotaryPositionalEmbedding = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_pos_emb,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
                rotate_type=rotate_type
            )
    
    def forward(self, Q_input, K_input=None, V_input=None, attention_mask=None, attention_bias=None, past_key_values=None, use_cache=False):
        """
        Args:
            Q_input: Query 输入，shape [batch_size, seq_len, hidden_dim]
            K_input: Key 输入，shape [batch_size, seq_len, hidden_dim]
            V_input: Value 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
            attention_bias: 注意力偏置（如相对位置偏置），shape [num_heads, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
            past_key_values: 之前的 KV 缓存，tuple 形式 (past_latent_k, past_latent_v)
            use_cache: 是否使用 KV 缓存
        """
        if K_input is None:
            K_input = Q_input
        if V_input is None:
            V_input = K_input
        
        b_size, s_len, _ = Q_input.shape
        
        # ============================================
        # 步骤 1: Q 投影（保持标准 GQA 方式）
        # ============================================
        Q = self.q_proj(Q_input)
        Q = Q.view(b_size, s_len, self.num_attention_heads, self.head_dim)
        
        # ============================================
        # 步骤 2: KV 压缩到潜在空间（降维）
        # ============================================
        # 将 K 和 V 一起投影到潜在空间
        kv_b = self.kv_b_proj(K_input)  # [batch, seq_len, 2 * latent_dim]
        
        # 分离 K 和 V 的潜在表示
        k_b, v_b = kv_b.chunk(2, dim=-1)
        # k_b, v_b shape: [batch, seq_len, latent_dim]
        
        # ============================================
        # 步骤 3: 处理 KV 缓存
        # ============================================
        if past_key_values is not None:
            past_latent_k, past_latent_v = past_key_values
            # past_latent_k shape: [batch, past_len, latent_dim]
            # k_b shape: [batch, curr_len, latent_dim]
            k_b = torch.cat([past_latent_k, k_b], dim=1)
            v_b = torch.cat([past_latent_v, v_b], dim=1)
        
        # 保存新的 KV 缓存（保存潜在表示）
        present_key_values = None
        if use_cache:
            present_key_values = (k_b, v_b)
        
        # ============================================
        # 步骤 4: 从潜在空间投影回注意力空间（升维）
        # ============================================
        k_a = self.kv_a_proj(k_b)  # [batch, total_len, 2 * num_kv_heads * head_dim]
        v_a = self.kv_a_proj(v_b)  # [batch, total_len, 2 * num_kv_heads * head_dim]
        
        # 分离 K 和 V
        K, _ = k_a.chunk(2, dim=-1)
        _, V = v_a.chunk(2, dim=-1)
        
        total_len = K.shape[1]
        # 重塑 K 和 V
        K = K.view(b_size, total_len, self.num_key_value_heads, self.head_dim)
        V = V.view(b_size, total_len, self.num_key_value_heads, self.head_dim)
        
        # ============================================
        # 步骤 5: 应用旋转位置编码（如果初始化了）
        # ============================================
        if self.RotaryPositionalEmbedding is not None:
            # 如果有 past_key_values，需要计算当前 token 的位置
            if past_key_values is not None:
                past_len = past_key_values[0].shape[1]
                # 只对当前 token 的 Q 和 K 应用位置编码
                # 首先获取当前 Q
                current_Q = Q
                # 获取当前 K（最后 s_len 个 token）
                current_K = K[:, -s_len:, :, :]
                # 应用位置编码
                current_Q, current_K = self.RotaryPositionalEmbedding(
                    current_Q, current_K, unsqueeze_dim=1, offset=past_len
                )
                # 替换 Q 和 K 中的当前部分
                Q = current_Q
                # K 保留完整的，因为 past K 已经有位置编码了
            else:
                Q, K = self.RotaryPositionalEmbedding(Q, K, unsqueeze_dim=1)
        
        # ============================================
        # 步骤 6: 标准注意力计算
        # ============================================
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)  # [batch, num_kv_heads, total_len, head_dim]
        V = V.transpose(1, 2)  # [batch, num_kv_heads, total_len, head_dim]
        
        # 重复 K 和 V 以匹配 Q 的头数
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)
        
        # 计算注意力分数
        attn_scores = Q @ K.transpose(-1, -2) * self.scale_factor
        
        # 添加注意力偏置（如相对位置偏置）
        if attention_bias is not None:
            if attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(0)
            attn_scores = attn_scores + attention_bias
        
        # 应用注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        
        # 计算注意力输出
        head_output = self.attn_drop(F.softmax(attn_scores, dim=-1)) @ V
        output = head_output.transpose(1, 2).reshape(b_size, s_len, -1)
        
        # 输出投影
        output = self.o_proj(output)
        
        if use_cache:
            return output, present_key_values
        else:
            return output
    
    def get_latent_kv(self, hidden_states):
        """
        单独获取潜在 KV 表示（用于 KV 缓存）
        
        在推理时，可以只保存潜在 KV，而不是完整的 KV，
        这样可以大幅减少内存占用
        
        Args:
            hidden_states: 输入 hidden states，shape [batch_size, seq_len, hidden_dim]
        
        Returns:
            latent_k, latent_v: 潜在 KV 表示，shape [batch_size, seq_len, latent_dim]
        """
        kv_b = self.kv_b_proj(hidden_states)
        latent_k, latent_v = kv_b.chunk(2, dim=-1)
        return latent_k, latent_v
    
    def from_latent_kv(self, latent_k, latent_v):
        """
        从潜在 KV 表示恢复完整 KV
        
        Args:
            latent_k: 潜在 K 表示，shape [batch_size, seq_len, latent_dim]
            latent_v: 潜在 V 表示，shape [batch_size, seq_len, latent_dim]
        
        Returns:
            K, V: 完整的 KV 表示，shape [batch_size, seq_len, num_kv_heads, head_dim]
        """
        b_size, s_len, _ = latent_k.shape
        
        k_a = self.kv_a_proj(latent_k)
        v_a = self.kv_a_proj(latent_v)
        
        K, _ = k_a.chunk(2, dim=-1)
        _, V = v_a.chunk(2, dim=-1)
        
        K = K.view(b_size, s_len, self.num_key_value_heads, self.head_dim)
        V = V.view(b_size, s_len, self.num_key_value_heads, self.head_dim)
        
        return K, V
