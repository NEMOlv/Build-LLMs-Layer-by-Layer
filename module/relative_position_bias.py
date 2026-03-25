"""
T5 风格相对位置偏置模块

本模块实现了 T5（Text-to-Text Transfer Transformer）模型中使用的相对位置编码机制。
与传统的绝对位置编码不同，相对位置编码考虑了 token 之间的相对距离，
使模型能够更好地处理长序列和泛化到训练时未见过的序列长度。

关键特性：
- 使用对数分桶（logarithmic bucketing）技术减少参数数量
- 支持双向（编码器）和单向（解码器）模式
- 可以独立于特定的注意力机制使用
- 可与 MHA、MQA、GQA 等多种注意力机制配合使用

参考文献：
- T5 论文: https://arxiv.org/abs/1910.10683
"""

import math
import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    相对位置偏置模块，用于 T5 风格的相对位置编码
    
    这个模块可以独立使用，计算相对位置偏置并返回，
    可以被任何注意力机制使用，包括 MHA、MQA、GQA 等。
    
    核心思想：
        1. 计算每对 token 之间的相对位置
        2. 将相对位置映射到有限数量的 bucket 中（使用对数分桶）
        3. 为每个 bucket 学习可训练的偏置值
        4. 将偏置值添加到 attention score 中
    
    数学公式：
        attention_score = q @ k^T / sqrt(d_k) + relative_position_bias
        
    其中 relative_position_bias 是通过本模块学习得到的。
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
