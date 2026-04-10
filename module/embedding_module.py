import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

"""
     symbols:
        b: batch
        s: sequence
        h: hidden
        q: query
        k: key
        v: value
        dim: dimension
        proj: projection
        attn: attention
"""

# sin/cos位置编码向量 (transformer论文称其为Sinusoidal Positional Embedding)
# 但实际上是偶数用正弦sin, 奇数用余弦cos
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, config):
        super(SinusoidalPositionEmbedding, self).__init__()
        # 创建位置编码矩阵
        self.encoding = torch.zeros(config.max_len, config.hidden_dim, device=config.device)
        self.encoding.requires_grad_(False)
        # 创建一个0~max_len-1的张量, 将其转为float tensor, 并扩展一个维度
        # pos shape: [1, max_len]
        # pos是token的下标数组
        pos = torch.arange(0, config.max_len, device=config.device).float().unsqueeze(1)
        # 2i是token特征的下标数组
        _2i = torch.arange(0, config.hidden_dim, 2, device=config.device)
        # 偶数用正弦
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / config.hidden_dim)))
        # 奇数用余弦
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / config.hidden_dim)))

    def forward(self, input):
        # 返回s_len长度为input.shape[1]的位置编码
        return self.encoding[:input.shape[1], :]


class LearnableAbsolutePositionEmbedding(nn.Module):
    """
    可学习的绝对位置编码 (Learnable Absolute Position Embedding) 模块

    核心思想：
    - 为每个位置学习独立的嵌入向量，通过反向传播优化
    - 简单直观，被 BERT、GPT-1/2 等经典模型采用
    - 注意：Transformer 原论文使用的是正弦/余弦编码，
            但 BERT/GPT 等实际实现改用可学习编码，效果更好

    关键特性：
    - 可学习参数：位置嵌入矩阵会在训练过程中更新
    - 支持变长序列：自动截取或扩展到输入序列长度
    - 易于集成：可以直接与 token embedding 相加
    - 训练稳定：绝对位置编码提供明确的位置信号

    数学表示：
        position_embedding[pos] = learnable_parameter[pos]
        output = token_embedding + position_embedding

    参考文献：
    - BERT 论文: https://arxiv.org/abs/1810.04805
    - GPT 论文: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Transformer 原论文（正弦编码）: https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            hidden_dim: int,
            max_position_embeddings: int = 512,
            padding_idx: Optional[int] = None,
    ):
        """
        初始化可学习的绝对位置编码模块

        Args:
            hidden_dim: 嵌入维度，需要与 token embedding 维度保持一致
            max_position_embeddings: 最大支持的序列长度，默认 512
            padding_idx: 填充 token 的索引，该位置的梯度不会被更新（可选）
                - 如果提供，padding_idx，该位置的嵌入向量不会被训练
                - 通常用于处理 padding token 的位置
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings
        self.padding_idx = padding_idx

        # 创建可学习的位置嵌入参数矩阵
        # shape: [max_position_embeddings, hidden_dim]
        # 为每个位置学习独立的嵌入向量
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx
        )

    def forward(
            self,
            input_or_shape,
            position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：生成位置嵌入

        支持两种使用方式：
        1. 通过 input_or_shape 自动生成位置索引
        2. 直接传入 position_ids 指定位置索引

        Args:
            input_or_shape: 输入张量或输入形状
                - 如果是 Tensor: shape [batch_size, seq_len] 或 [batch_size, seq_len, hidden_dim]
                - 如果是 tuple: (batch_size, seq_len)
            position_ids: 位置索引，shape [batch_size, seq_len]，None 表示从 0 开始自动生成
                - 用于自定义位置（如 padding 后的位置）

        Returns:
            position_embeddings: 位置嵌入张量，shape [batch_size, seq_len, hidden_dim]
        """
        # 确定输入序列长度和 batch size
        if isinstance(input_or_shape, torch.Tensor):
            if input_or_shape.dim() == 3:
                batch_size, seq_len = input_or_shape.shape[0], input_or_shape.shape[1]
            else:
                batch_size, seq_len = input_or_shape.shape[0], input_or_shape.shape[1]
        else:
            batch_size, seq_len = input_or_shape

        # 生成位置索引
        if position_ids is None:
            # 自动生成从 0 到 seq_len-1 的位置索引
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.position_embeddings.weight.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # 确保位置索引不超过最大位置嵌入数
        # 如果输入序列长度超过 max_position_embeddings，截断
        position_ids = torch.clamp(position_ids, 0, self.max_position_embeddings - 1)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        return position_embeddings

    def extra_repr(self) -> str:
        """返回模块的字符串表示，用于打印和调试"""
        return (f'hidden_dim={self.hidden_dim}, '
                f'max_position_embeddings={self.max_position_embeddings}, '
                f'padding_idx={self.padding_idx}')


class AttentionWithLinearBiases(nn.Module):
    """
    Attention with Linear Biases (ALiBi) 模块

    核心思想：
    - 不在嵌入层添加位置编码，而是在注意力分数上添加线性偏置
    - 每个注意力头学习不同的斜率参数，控制对相对位置的敏感度
    - 支持训练时使用短序列、推理时扩展到更长序列，无需重新训练

    关键特性：
    - 无显式位置嵌入：不增加 embedding 维度的参数量
    - 长度外推能力：训练 512，推理 4k+ 效果良好
    - 代码简洁：只需在 attention score 上添加偏置
    - 训练稳定：无需担心位置编码的优化问题

    数学表示：
        attention_score = q @ k^T / sqrt(d_k)
        attention_score = attention_score + bias
        where bias[i, j] = -m * |i - j|
        m 是每个头学习的斜率参数

    参考文献：
    - ALiBi 论文: https://arxiv.org/abs/2108.12409
    """

    def __init__(
            self,
            num_heads: int,
            max_positions: int = 4096,
            train_max_positions: Optional[int] = None,
    ):
        """
        初始化 ALiBi 模块

        Args:
            num_heads: 注意力头的数量
            max_positions: 最大支持的序列长度，默认 4096
            train_max_positions: 训练时的最大序列长度（用于外推场景），
                如果为 None 则等于 max_positions
                - 例如：训练时用 512，推理时用 4096
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_positions
        self.train_max_positions = train_max_positions or max_positions

        # 为每个注意力头计算斜率 m
        # 按照 ALiBi 论文的建议：
        # m = 2^(-8/num_heads * (i+1))  for i in 0..num_heads-1
        # 这种几何级数分配让不同头对不同距离的敏感度不同
        m = torch.tensor(
            [2 ** (-8 / num_heads * (i + 1)) for i in range(num_heads)],
            dtype=torch.float32
        )
        
        # 注册为 buffer（不需要梯度，但会随模型保存）
        self.register_buffer("m", m)

        # 预计算最大长度的偏置矩阵（训练时可以只计算需要的长度）
        self._precompute_bias(max_positions)

    def _precompute_bias(self, max_len: int) -> None:
        """
        预计算偏置矩阵

        Args:
            max_len: 预计算的最大长度
        """
        device = self.m.device
        
        # 生成位置索引
        pos = torch.arange(max_len, device=device)
        
        # 计算相对位置矩阵 [max_len, max_len]
        # rel_pos[i, j] = |i - j|
        rel_pos = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
        
        # 计算偏置矩阵 [num_heads, max_len, max_len]
        # bias[h, i, j] = -m[h] * |i - j|
        bias = -self.m.view(-1, 1, 1) * rel_pos.unsqueeze(0)
        
        # 注册为 buffer
        if hasattr(self, "bias"):
            self.bias = bias
        else:
            self.register_buffer("bias", bias)

    def forward(
            self,
            attention_scores: torch.Tensor,
            key_len: Optional[int] = None,
            query_len: Optional[int] = None,
            use_cache: bool = False,
    ) -> torch.Tensor:
        """
        前向传播：将 ALiBi 偏置添加到注意力分数上

        支持两种使用方式：
        1. 标准模式：传入完整的 attention_scores，形状 [batch, num_heads, query_len, key_len]
        2. 增量解码模式：use_cache=True，支持自回归生成

        Args:
            attention_scores: 注意力分数张量
                形状: [batch_size, num_heads, query_length, key_length]
            key_len: key 的长度（可选，自动推断）
            query_len: query 的长度（可选，自动推断）
            use_cache: 是否使用缓存（用于增量解码）

        Returns:
            attention_scores_with_bias: 添加了 ALiBi 偏置的注意力分数
                形状与输入相同
        """
        batch_size, num_heads, q_len, k_len = attention_scores.shape
        
        # 验证头数量匹配
        assert num_heads == self.num_heads, \
            f"注意力头数量不匹配: 期望 {self.num_heads}, 实际 {num_heads}"
        
        # 确定需要的偏置长度
        required_len = max(q_len, k_len)
        
        # 如果需要的长度超过预计算的，重新计算
        if required_len > self.bias.shape[1]:
            self._precompute_bias(required_len)
        
        # 截取需要的偏置部分
        # 对于自回归生成，query_len=1，key_len=past+current
        bias = self.bias[:, :q_len, :k_len]
        
        # 扩展 batch 维度并添加偏置
        # bias: [num_heads, q_len, k_len] -> [1, num_heads, q_len, k_len]
        attention_scores = attention_scores + bias.unsqueeze(0)
        
        return attention_scores

    def get_alibi_slopes(self) -> torch.Tensor:
        """
        获取每个注意力头的 ALiBi 斜率参数

        Returns:
            slopes: 斜率张量，shape [num_heads]
        """
        return self.m.clone()

    def extra_repr(self) -> str:
        """返回模块的字符串表示，用于打印和调试"""
        return (f'num_heads={self.num_heads}, '
                f'max_positions={self.max_positions}, '
                f'train_max_positions={self.train_max_positions}')
                

class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码

    核心思想：
    - 考虑了 token 之间的相对距离，使模型能够更好地处理长序列和泛化到训练时未见过的序列长度
    - 使用对数分桶（logarithmic bucketing）技术减少参数数量

    工程特性：
    - 支持双向（编码器）和单向（解码器）模式
    - 可以独立于特定的注意力机制使用
    - 可与 MHA、MQA、GQA 等多种注意力机制配合使用

    数学公式：
        attention_score = q @ k^T / sqrt(d_k) + relative_position_bias

    参考文献：
    - T5 论文: https://arxiv.org/abs/1910.10683
    """

    def __init__(
            self,
            num_buckets: int = 32,
            max_distance: int = 128,
            num_heads: int = 8,
    ):
        """
        初始化相对位置偏置模块
        
        Args:
            num_buckets: 相对位置的 bucket 数量，默认 32
                - 更多的 bucket 可以提供更精细的位置表示，但会增加参数量
                - T5 论文中使用 32 个 bucket
            max_distance: 最大距离，超过这个距离的都会被放到同一个 bucket，默认 128
                - 超过 max_distance 的相对位置不会被区分
                - 这样可以限制参数量，同时保持对长距离的一定感知能力
            num_heads: 注意力头的数量，默认 8
                - 每个注意力头会学习独立的相对位置偏置
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        
        # Embedding 层，用于存储每个 bucket 的偏置值
        # shape: [num_buckets, num_heads]
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
            relative_position: torch.Tensor,
            bidirectional: bool = True,
            num_buckets: int = 32,
            max_distance: int = 128
    ) -> torch.Tensor:
        """
        将相对位置转换为 bucket 索引
        
        这是 T5 相对位置编码的核心算法，使用对数分桶技术：
        
        算法原理：
            1. 对于小距离（小于 max_exact），使用精确的整数表示
            2. 对于大距离（大于等于 max_exact），使用对数尺度
            3. 这样可以在保持精细局部信息的同时，减少对长距离的参数需求
        
        数学表示：
            if distance < max_exact:
                bucket = distance
            else:
                bucket = max_exact + log(distance / max_exact) * (num_buckets - max_exact) / log(max_distance / max_exact)
        
        Args:
            relative_position: 相对位置张量，shape [query_len, key_len]
                每个元素表示 key_position - query_position
                正值表示 key 在 query 后面
                负值表示 key 在 query 前面
            bidirectional: 是否为双向（编码器使用双向，解码器使用单向）
                - 双向：区分正负相对位置（前面和后面）
                - 单向：只考虑非负相对位置（解码器自注意力）
            num_buckets: bucket 数量
            max_distance: 最大距离
            
        Returns:
            bucket 索引，shape [query_len, key_len]
            每个元素是一个整数，表示对应相对位置所属的 bucket
        """
        relative_buckets = 0
        
        # 步骤 1：处理双向模式
        if bidirectional:
            # 双向模式下，将 bucket 分成两半
            # 前一半用于负相对位置（key 在 query 前面）
            # 后一半用于正相对位置（key 在 query 后面）
            num_buckets //= 2
            # 标记正相对位置
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 取绝对值，后续只需要处理距离大小
            relative_position = torch.abs(relative_position)
        else:
            # 单向模式下（解码器），只考虑非负相对位置
            # 将负值裁剪为 0（因为解码器只能看到前面的 token）
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # 步骤 2：计算精确表示的最大距离
        # max_exact 是我们使用精确整数表示的最大距离
        max_exact = num_buckets // 2
        
        # 标记哪些相对位置是小距离（可以精确表示）
        is_small = relative_position < max_exact
        
        # 步骤 3：处理大距离（使用对数分桶）
        # 对于大于等于 max_exact 的距离，使用对数尺度
        # 这样可以在保持参数数量可控的同时，仍能感知长距离关系
        relative_position_if_large = max_exact + (
                # 对数计算：log(distance / max_exact)
                torch.log(relative_position.float() / max_exact)
                # 归一化到 [0, 1] 范围（相对于 max_distance）
                / math.log(max_distance / max_exact)
                # 映射到剩余的 bucket 数量
                * (num_buckets - max_exact)
        ).to(torch.long)
        
        # 确保不超出 bucket 范围
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        # 步骤 4：合并小距离和大距离的 bucket 索引
        # 对于小距离，直接使用相对位置值
        # 对于大距离，使用对数分桶后的值
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        
        return relative_buckets

    def forward(
            self,
            query_length: int,
            key_length: int,
            bidirectional: bool = True,
            device: torch.device = None,
    ) -> torch.Tensor:
        """
        计算相对位置偏置
        
        这是模块的前向传播函数，执行以下步骤：
            1. 生成所有 query 和 key 的位置索引
            2. 计算每对 (query, key) 之间的相对位置
            3. 将相对位置映射到 bucket 索引
            4. 通过 embedding 层获取每个 bucket 的偏置值
            5. 调整维度顺序以适应注意力计算
        
        Args:
            query_length: query 的长度（自注意力中等于序列长度）
            key_length: key 的长度（自注意力中等于序列长度）
            bidirectional: 是否为双向
                - 编码器：True（可以看到前后所有 token）
                - 解码器自注意力：False（只能看到前面的 token）
                - 解码器交叉注意力：True（可以看到编码器的所有 token）
            device: 计算设备
                - 如果为 None，使用 relative_attention_bias.weight.device
            
        Returns:
            相对位置偏置，shape [num_heads, query_length, key_length]
            可以直接加到 attention score 上：
                attention_score = attention_score + relative_position_bias
        """
        # 步骤 1：确定计算设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        # 步骤 2：生成 query 和 key 的位置索引
        # context_position: [query_length, 1] - query 的位置索引，扩展为列向量
        # 例如，query_length=3 时：[[0], [1], [2]]
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        
        # memory_position: [1, key_length] - key 的位置索引，扩展为行向量
        # 例如，key_length=3 时：[[0, 1, 2]]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        # 步骤 3：计算相对位置
        # relative_position = memory_position - context_position
        # 使用广播机制，得到 [query_length, key_length] 的矩阵
        # 每个元素 (i, j) 表示 memory_position[j] - context_position[i]
        # 正值表示 key j 在 query i 后面
        # 负值表示 key j 在 query i 前面
        # 0 表示同一个位置
        #
        # 示例：
        # context_position = [[0], [1], [2]]
        # memory_position = [[0, 1, 2]]
        # relative_position = [[0-0, 1-0, 2-0],
        #                     [0-1, 1-1, 2-1],
        #                     [0-2, 1-2, 2-2]]
        #                   = [[0, 1, 2],
        #                      [-1, 0, 1],
        #                      [-2, -1, 0]]
        relative_position = memory_position - context_position
        
        # 步骤 4：将相对位置映射到 bucket 索引
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        # 确保 bucket 索引在正确的设备上
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        
        # 步骤 5：通过 embedding 层获取偏置值
        # values shape: [query_length, key_length, num_heads]
        values = self.relative_attention_bias(relative_position_bucket)
        
        # 步骤 6：调整维度顺序
        # 从 [query_length, key_length, num_heads]
        # 转换为 [num_heads, query_length, key_length]
        # 这样可以直接与 attention score 相加（attention score 通常是 [batch, num_heads, query_len, key_len]）
        values = values.permute([2, 0, 1])
        
        return values


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Positional Embedding, RoPE) 模块

    核心思想：
    - 可以直接在查询和键向量上应用旋转操作，而不需要显式的位置嵌入

    关键特性：
    - 支持两种旋转方式：rotate_interval（相邻对旋转）和 rotate_half（前后半旋转）
    - 支持可选的 RoPE 缩放（rope_scaling），用于扩展上下文长度
    - 预计算频率，提高推理效率
    - 可与 MHA、MQA、GQA 等多种注意力机制配合使用

    数学公式：
        q_rotated = q * cos + rotate(q) * sin
        k_rotated = k * cos + rotate(k) * sin

    参考文献：
    - RoPE 论文: https://arxiv.org/abs/2104.09864
    - Llama 论文: https://arxiv.org/abs/2302.13971
    """

    def __init__(
            self,
            dim: int,
            max_position_embeddings: int = 4096,
            rope_base: float = 10000.0,
            rope_scaling: Optional[dict] = None,
            rotate_type: str = "rotate_interval",
    ):
        """
        初始化旋转位置编码模块
        
        Args:
            dim: 每个注意力头的维度（head_dim）
            max_position_embeddings: 最大位置嵌入数，默认 4096
            rope_base: RoPE 的基础频率，默认 10000.0
            rope_scaling: RoPE 缩放配置，用于扩展上下文长度
                示例: {
                    "original_max_position_embeddings": 2048,
                    "factor": 4,
                    "beta_fast": 4.0,
                    "beta_slow": 1.0
                }
            rotate_type: 旋转类型，"rotate_interval"（相邻对旋转）或 "rotate_half"（前后半旋转）
                - rotate_interval: 将相邻两个向量的第一个当作实部，第二个当作虚部
                - rotate_half: 将前半部分向量当作实部，后半部分向量当作虚部
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.rotate_type = rotate_type
        
        # 预计算频率
        cos, sin = self._precompute_freqs(
            dim=dim,
            end=max_position_embeddings,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            rotate_type=rotate_type
        )
        
        # 注册为 buffer，这样可以随模型保存和加载
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _precompute_freqs(
            dim: int,
            end: int = 32 * 1024,
            rope_base: float = 10000.0,
            rope_scaling: Optional[dict] = None,
            rotate_type: str = "rotate_interval"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预计算旋转位置编码的频率
        
        Args:
            dim: 每个注意力头的维度
            end: 最大位置
            rope_base: RoPE 的基础频率
            rope_scaling: RoPE 缩放配置
            rotate_type: 旋转类型
            
        Returns:
            (cos, sin): 预计算的余弦和正弦值，shape [end, dim]
        """
        freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        
        if rope_scaling is not None:
            original_max, factor, beta_fast, beta_slow = (
                rope_scaling.get("original_max_position_embeddings", 2048),
                rope_scaling.get("factor", 4),
                rope_scaling.get("beta_fast", 4.0),
                rope_scaling.get("beta_slow", 1.0),
            )

            if end / original_max > 1.0:
                corr_dim = [i if 2 * math.pi / freqs[i] > original_max else dim // 2 for i in range(dim // 2)]
                power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
                beta = beta_slow + (beta_fast - beta_slow) * power
                scale = torch.where(
                    torch.arange(dim // 2, device=freqs.device) < corr_dim,
                    (beta * factor - beta + 1) / (beta * factor),
                    1.0 / factor,
                )

                freqs = freqs * scale

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        
        if rotate_type == "rotate_half":
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()
            sin = freqs.sin()
        else:
            cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
            sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        旋转前后部分
        
        1. 分割张量：将张量分割为前半部分和后半部分，并对后半部分取负
        2. 旋转操作：将后半部分拼接至前半部分的前方
        
        数学含义：将前半部分向量当作实部，后半部分向量当作虚部
        
        Args:
            x: 输入张量，shape [..., dim]
            
        Returns:
            旋转后的张量，shape [..., dim]
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    @staticmethod
    def _rotate_interval(x: torch.Tensor) -> torch.Tensor:
        """
        旋转相邻对
        
        1. 分割张量：将张量分割为奇数部分和偶数部分，并对奇数部分取负
        2. 旋转操作：将奇数部分交错插入偶数部分
        
        数学含义：将相邻两个向量的第一个当作实部，第二个当作虚部
        
        Args:
            x: 输入张量，shape [..., dim]
            
        Returns:
            旋转后的张量，shape [..., dim]
        """
        return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None,
            unsqueeze_dim: int = 1,
            offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到查询和键向量
        
        Args:
            q: 查询向量，shape [batch_size, seq_len, num_heads, head_dim] 或 [batch_size, num_heads, seq_len, head_dim]
            k: 键向量，shape 与 q 相同
            position_ids: 位置索引，shape [batch_size, seq_len]，None 表示从 0 开始
            unsqueeze_dim: 扩展维度的位置，用于匹配 q/k 的形状
            offset: 位置偏移量（用于 KV Cache 场景）
            
        Returns:
            (q_rotated, k_rotated): 旋转后的查询和键向量，shape 与输入相同
        """
        # 确定旋转函数
        if self.rotate_type == "rotate_half":
            rotate = self._rotate_half
        else:
            rotate = self._rotate_interval
        
        seq_len = q.shape[1] if q.dim() == 4 else q.shape[2]
        
        # 获取预计算的 cos 和 sin（考虑 offset）
        cos = self.cos[offset : offset + seq_len]
        sin = self.sin[offset : offset + seq_len]
        
        # 调整维度以匹配 q/k 的形状
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        
        # 应用旋转位置编码
        q_rotated = (q * cos) + (rotate(q) * sin)
        k_rotated = (k * cos) + (rotate(k) * sin)
        
        return q_rotated, k_rotated


class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.padding_idx
        )
        self.position_embedding = SinusoidalPositionEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input):
        tok_embedding = self.token_embedding(input)
        pos_embedding = self.position_embedding(input)
        output = self.dropout(tok_embedding + pos_embedding)

        return output


class BertEmbedding(nn.Module):
    """
    严格对齐结构图命名的BERT嵌入层
    变量名完全匹配图中标识：Token Embeddings/Segment Embeddings/Position Embeddings
    """
    def __init__(self, config):
        super().__init__()

        # 1. Token Embeddings（对应图中黄色Token Embeddings行）
        self.Token_Embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.padding_idx
        )

        # 2. Segment Embeddings（对应图中绿色Segment Embeddings行，E_A/E_B）
        self.Segment_Embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )

        # 3. Position Embeddings（对应图中白色Position Embeddings行，E0~E10）
        self.Position_Embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )

        # 图中相加后的归一化和Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor = None):
        """
        Args:
            input_ids: 输入token的id序列，对应图中输入token列，shape [batch_size, seq_len]
            segment_ids: 句子段id，对应图中E_A/E_B，shape [batch_size, seq_len]
                         不传则默认全0（单句子场景）
        Returns:
            embeddings: 最终融合嵌入，shape [batch_size, seq_len, hidden_size]
        """
        # 获取序列长度和设备信息
        seq_len = input_ids.size(1)
        device = input_ids.device

        # ---------------------- 生成Position IDs（对应图中E0~Eseq_len-1） ----------------------
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # ---------------------- 处理Segment IDs（对应图中E_A/E_B） ----------------------
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        # ---------------------- 计算三种嵌入（完全匹配图中命名） ----------------------
        Token_Emb = self.Token_Embeddings(input_ids)          # Token Embeddings输出
        Segment_Emb = self.Segment_Embeddings(segment_ids)    # Segment Embeddings输出
        Position_Emb = self.Position_Embeddings(position_ids) # Position Embeddings输出

        # ---------------------- 三种嵌入相加（对应图中+号） ----------------------
        Embeddings = Token_Emb + Segment_Emb + Position_Emb

        # ---------------------- 归一化和Dropout（对应图中后续处理） ----------------------
        Embeddings = self.LayerNorm(Embeddings)
        Embeddings = self.Dropout(Embeddings)

        return Embeddings


class BartEmbedding(nn.Module):
    """
    BART 嵌入层，使用绝对位置编码
    
    BART 使用标准的绝对位置编码，与 BERT 类似但不需要 segment 嵌入。
    仅包含 token 嵌入和位置嵌入。
    """
    def __init__(self, config):
        super().__init__()
        
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx
        )
        
        self.embed_positions = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.d_model
        )
        
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: 输入 token id 序列，shape [batch_size, seq_len]
        Returns:
            embeddings: 嵌入向量，shape [batch_size, seq_len, d_model]
        """
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_emb = self.embed_tokens(input_ids)
        position_emb = self.embed_positions(position_ids)
        
        embeddings = token_emb + position_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.Dropout(embeddings)
        
        return embeddings