import torch


class Masking:
    """
    统一的掩码策略类
    
    支持两种掩码模式：
    - 静态掩码 (BERT 风格): 首次生成后缓存，后续使用相同的掩码
    - 动态掩码 (RoBERTa 风格): 每次都重新生成掩码，不缓存
    """

    def __init__(self, tokenizer, mlm_probability=0.15, mask_probability=0.8, random_probability=0.1, strategy="dynamic"):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            mlm_probability: 整体掩码概率，默认 0.15
            mask_probability: 替换为 [MASK] 的概率，默认 0.8
            random_probability: 替换为随机 token 的概率，默认 0.1
            strategy: 掩码策略，"static" (BERT 风格) 或 "dynamic" (RoBERTa 风格)
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.keep_probability = 1.0 - mask_probability - random_probability
        self.strategy = strategy
        
        # 静态掩码缓存
        if strategy == "static":
            self.mask_cache = {}

    def mask(self, input_ids, special_tokens_mask, sample_index=None):
        """
        生成掩码
        
        Args:
            input_ids: 输入 token IDs
            special_tokens_mask: 特殊 token 掩码
            sample_index: 样本索引，用于静态掩码缓存
        
        Returns:
            masked_input_ids: 掩码后的输入
            labels: 标签 (-100 表示忽略)
        """
        # 静态掩码：使用缓存
        if self.strategy == "static" and sample_index is not None and sample_index in self.mask_cache:
            return self.mask_cache[sample_index]
        
        # 生成新的掩码
        labels = input_ids.clone()
        
        # 概率矩阵: 每个token被mask的概率为mlm_probability
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 屏蔽特殊token: 将特殊token的mask概率置为0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 采样需要掩码的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 非掩码位置标签设为 -100，CrossEntropyLoss 会自动忽略
        labels[~masked_indices] = -100
        
        # 每个token有self.mask_probability 概率被替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_probability)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 每个token有self.random_probability 概率被替换为随机 token
        indices_random = torch.bernoulli(
            torch.full(labels.shape, self.random_probability / (self.random_probability + self.keep_probability))
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            low=0,
            high=len(self.tokenizer),
            size=labels.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        
        # 静态掩码：缓存结果
        if self.strategy == "static" and sample_index is not None:
            self.mask_cache[sample_index] = (input_ids, labels)
        
        return input_ids, labels

    def clear_cache(self):
        """清空静态掩码缓存"""
        if self.strategy == "static" and hasattr(self, "mask_cache"):
            self.mask_cache.clear()
