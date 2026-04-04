import torch
from torch.utils.data import Dataset
import os
import random
from datasets import load_dataset
from dataset.masking import Masking

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ========================================
# 统一数据集 - 支持所有任务
# ========================================

class UnifiedDataset(Dataset):
    """
    统一的预训练数据集
    
    所有任务都在 token 级别进行处理，支持 8 个独立任务：
    1. token_masking (等价于 MLM) - Token 遮挡
    2. token_deletion - Token 删除
    3. text_infilling (等价于 Span Corruption) - 连续文本填空/片段损坏
    4. sentence_permutation - 句子打乱
    5. document_rotation - 文档旋转
    6. next_sentence_prediction (NSP) - 下一句预测
    7. sentence_order_prediction (SOP) - 句序预测
    8. replaced_token_detection (RTD) - 替换 token 检测 (ELECTRA)
    
    任务执行顺序：
    第一轮（无 mask 任务）：next_sentence_prediction, sentence_order_prediction, sentence_permutation, document_rotation
    第二轮（有 mask 任务）：token_masking, token_deletion, text_infilling, replaced_token_detection
    """
    
    # 所有支持的任务
    ALL_TASKS = [
        "token_masking",
        "token_deletion",
        "text_infilling",
        "sentence_permutation",
        "document_rotation",
        "next_sentence_prediction",
        "sentence_order_prediction",
        "replaced_token_detection"
    ]
    
    # 无 mask 的任务（第一轮执行）
    NO_MASK_TASKS = [
        "next_sentence_prediction",
        "sentence_order_prediction",
        "sentence_permutation",
        "document_rotation"
    ]
    
    # 有 mask 的任务（第二轮执行）
    MASK_TASKS = [
        "token_masking",
        "token_deletion",
        "text_infilling",
        "replaced_token_detection"
    ]
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=512,
        task_types=None,
        model_style="encoder_decoder",
        mlm_probability=0.15,
        mask_probability=0.8,
        random_probability=0.1,
        deletion_probability=0.5,
        infilling_span_probability=0.1,
        infilling_lambda=3.0,
        nsp_probability=0.5,
        sop_probability=0.5,
        rtd_probability=0.15,
        evaluation_enabled=True
    ):
        """
        Args:
            data_path: 预训练数据路径，JSONL 格式 {"text": "文本内容"}
            tokenizer: Hugging Face tokenizer
            max_length: 最大序列长度
            task_types: 要执行的任务列表，None 表示所有任务
            model_style: 模型风格，"encoder_only"、"decoder_only" 或 "encoder_decoder"
            mlm_probability: Token Masking 的整体掩码概率，默认 0.15
            mask_probability: 每个 mask_token 用 [MASK] 替换的概率，默认 0.8
            random_probability: 每个 mask_token 用随机 token 替换的概率，默认 0.1
            deletion_probability: Token Deletion 的删除概率，默认 0.5
            infilling_span_probability: Text Infilling 的片段选择概率，默认 0.1
            infilling_lambda: Text Infilling 的泊松分布 lambda 参数，默认 3.0
            nsp_probability: NSP (Next Sentence Prediction) 任务的负样本概率，默认 0.5
            sop_probability: SOP (Sentence Order Prediction) 任务的交换概率，默认 0.5
            rtd_probability: RTD (Replaced Token Detection) 任务的替换概率，默认 0.15
            evaluation_enabled: 是否启用数据质量评估
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_style = model_style
        self.mlm_probability = mlm_probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.deletion_probability = deletion_probability
        self.infilling_span_probability = infilling_span_probability
        self.infilling_lambda = infilling_lambda
        self.nsp_probability = nsp_probability
        self.sop_probability = sop_probability
        self.rtd_probability = rtd_probability
        self.evaluation_enabled = evaluation_enabled
        
        # 设置要执行的任务
        if task_types is None:
            self.task_types = self.ALL_TASKS
        else:
            # 验证任务类型有效
            for task in task_types:
                assert task in self.ALL_TASKS, f"无效的任务类型: {task}"
            self.task_types = task_types
        
        # 加载数据
        self.samples = load_dataset("json", data_files=data_path, split="train")
        
        # 预处理：将所有样本分词化并存储
        self.token_samples = self._preprocess_tokenize()
        
        # 数据质量评估指标
        self.evaluation_metrics = {
            "total_samples": 0,
            "task_counts": {task: 0 for task in self.ALL_TASKS},
            "avg_source_length": 0,
            "avg_target_length": 0
        }
    
    def _preprocess_tokenize(self):
        """
        预处理函数：将所有文本数据分词化
        
        Returns:
            list: 包含所有样本的 token 列表
        """
        token_samples = []
        for sample in self.samples:
            text = sample["text"]
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            tokens = tokenized["input_ids"].squeeze(0).tolist()
            token_samples.append({
                "text": text,
                "tokens": tokens
            })
        return token_samples
    
    def _token_masking(self, tokens):
        """
        Token 遮挡任务（等价于 MLM）
        
        随机选择 15% 的 token 进行遮挡：
        - 80% 替换为 [MASK]
        - 10% 替换为随机 token
        - 10% 保持不变
        """
        if not tokens:
            return tokens, tokens
        
        tokens = tokens.copy()
        labels = tokens.copy()
        num_tokens = len(tokens)
        
        num_to_mask = max(1, int(num_tokens * self.mlm_probability))
        mask_indices = random.sample(range(num_tokens), num_to_mask)
        
        for idx in mask_indices:
            rand = random.random()
            if rand < self.mask_probability:
                tokens[idx] = self.tokenizer.mask_token_id
            elif rand < self.mask_probability + self.random_probability:
                random_token = random.randint(0, self.tokenizer.vocab_size - 1)
                tokens[idx] = random_token
        
        return tokens, labels
    
    def _token_deletion(self, tokens):
        """
        Token 删除任务
        
        以 50% 的概率随机删除 token，保留原始顺序
        """
        if not tokens:
            return tokens, tokens
        
        labels = tokens.copy()
        kept_tokens = []
        for token in tokens:
            if random.random() > self.deletion_probability:
                kept_tokens.append(token)
        
        if not kept_tokens:
            kept_tokens = [tokens[0]]
        
        return kept_tokens, labels
    
    def _text_infilling(self, tokens):
        """
        Text Infilling 任务（等价于 Span Corruption - 片段损坏）
        
        将输入文本分割成多个片段，其中 10% 的片段被替换为单个 [MASK]
        片段长度遵循泊松分布(λ=3)
        """
        if not tokens:
            return tokens, tokens
        
        tokens = tokens.copy()
        labels = tokens.copy()
        num_tokens = len(tokens)
        
        if num_tokens < 2:
            return tokens, labels
        
        result_tokens = []
        current_pos = 0
        
        while current_pos < num_tokens:
            if random.random() < self.infilling_span_probability and current_pos < num_tokens - 1:
                span_length = min(
                    max(1, int(random.expovariate(1.0 / self.infilling_lambda))),
                    num_tokens - current_pos
                )
                result_tokens.append(self.tokenizer.mask_token_id)
                current_pos += span_length
            else:
                result_tokens.append(tokens[current_pos])
                current_pos += 1
        
        return result_tokens, labels
    
    def _sentence_permutation(self, tokens):
        """
        Sentence Permutation 任务（句子打乱）
        
        将文档中的句子进行随机重排，保持句子内部词序不变
        按 [SEP] token 分割成片段，然后打乱片段顺序
        """
        if not tokens or len(tokens) <= 1:
            return tokens, tokens
        
        # 按 [SEP] 分割
        segments = self._split_by_sep(tokens)
        
        if len(segments) < 2:
            return tokens, tokens
        
        # 打乱片段顺序
        random.shuffle(segments)
        
        # 重组
        permuted_tokens = []
        for seg in segments:
            permuted_tokens.extend(seg)
        
        return permuted_tokens, tokens
    
    def _document_rotation(self, tokens):
        """
        Document Rotation 任务（文档旋转）
        
        随机选择文档中的一个 token 作为新起始点，将文档旋转
        """
        if not tokens or len(tokens) <= 1:
            return tokens, tokens
        
        rotation_point = random.randint(1, len(tokens) - 1)
        rotated_tokens = tokens[rotation_point:] + tokens[:rotation_point]
        
        return rotated_tokens, tokens
    
    def _split_by_sep(self, tokens):
        """
        根据 [SEP] 特殊 token 分割 tokens
        
        Returns:
            list: 分割后的 tokens 列表
        """
        sep_token_id = self.tokenizer.sep_token_id
        segments = []
        start = 0
        
        for i, token in enumerate(tokens):
            if token == sep_token_id:
                segments.append(tokens[start:i + 1])
                start = i + 1
        
        if start < len(tokens):
            segments.append(tokens[start:])
        
        return segments
    
    def _next_sentence_prediction(self, index, original_tokens):
        """
        NSP (Next Sentence Prediction) 任务（下一句预测）
        
        预测第二部分 token 是否是第一部分 token 的后续内容
        按 [SEP] token 分割，然后用随机样本的片段替换第二部分
        """
        # 按 [SEP] 分割
        segments = self._split_by_sep(original_tokens)
        
        if len(segments) < 2:
            return original_tokens, original_tokens, torch.tensor(0, dtype=torch.long)
        
        is_negative = random.random() < self.nsp_probability
        
        if not is_negative:
            return original_tokens, original_tokens, torch.tensor(0, dtype=torch.long)
        
        # 取第一部分
        first_segment = segments[0]
        
        # 从 self.token_samples 中获取随机样本
        random_index = random.randint(0, len(self.token_samples) - 1)
        while random_index == index and len(self.token_samples) > 1:
            random_index = random.randint(0, len(self.token_samples) - 1)
        
        random_tokens = self.token_samples[random_index]["tokens"]
        random_segments = self._split_by_sep(random_tokens)
        
        # 从随机样本中取一个片段
        if random_segments:
            second_segment = random_segments[random.randint(0, len(random_segments) - 1)]
        else:
            second_segment = random_tokens
        
        # 组合
        corrupted_tokens = first_segment + second_segment
        
        # 确保长度不超限
        if len(corrupted_tokens) > self.max_length:
            corrupted_tokens = corrupted_tokens[:self.max_length]
        
        return corrupted_tokens, original_tokens, torch.tensor(1, dtype=torch.long)
    
    def _sentence_order_prediction(self, index, original_tokens):
        """
        SOP (Sentence Order Prediction) 任务（句序预测）
        
        预测两部分 token 是否被交换了顺序
        按 [SEP] token 分割，然后交换前两个片段的顺序
        """
        # 按 [SEP] 分割
        segments = self._split_by_sep(original_tokens)
        
        if len(segments) < 2:
            return original_tokens, original_tokens, torch.tensor(0, dtype=torch.long)
        
        should_exchange = random.random() < self.sop_probability
        
        if not should_exchange:
            return original_tokens, original_tokens, torch.tensor(0, dtype=torch.long)
        
        # 交换前两个片段的顺序
        first_segment = segments[0]
        second_segment = segments[1]
        remaining_segments = segments[2:]
        
        corrupted_tokens = second_segment + first_segment
        for seg in remaining_segments:
            corrupted_tokens.extend(seg)
        
        return corrupted_tokens, original_tokens, torch.tensor(1, dtype=torch.long)
    
    def _replaced_token_detection(self, tokens):
        """
        RTD (Replaced Token Detection) 任务（替换 token 检测） - ELECTRA
        
        首先使用 MLM 方式随机选择一些 token 替换为随机 token，
        然后为每个 token 打标签：0 表示原始 token，1 表示被替换的 token
        """
        if not tokens:
            return tokens, tokens, []
        
        tokens = tokens.copy()
        original_tokens = tokens.copy()
        num_tokens = len(tokens)
        
        # 选择要替换的位置（类似 MLM 的 15% 概率）
        num_to_replace = max(1, int(num_tokens * self.rtd_probability))
        replace_indices = random.sample(range(num_tokens), num_to_replace)
        
        # 生成标签：0 表示未替换，1 表示被替换
        rtd_labels = [0] * num_tokens
        
        for idx in replace_indices:
            # 替换为随机 token
            random_token = random.randint(0, self.tokenizer.vocab_size - 1)
            tokens[idx] = random_token
            rtd_labels[idx] = 1
        
        return tokens, original_tokens, rtd_labels
    
    def __len__(self):
        return len(self.token_samples)
    
    def __getitem__(self, index):
        """
        获取单个样本并进行处理
        
        任务执行顺序：
        1. 先执行无 mask 的任务（nsp, sop, sentence_permutation, document_rotation）
        2. 再执行有 mask 的任务（token_masking, token_deletion, text_infilling）
        
        Returns:
            根据 encoder_style 返回不同格式的结果
        """
        sample = self.token_samples[index]
        original_text = sample["text"]
        original_tokens = sample["tokens"].copy()
        
        corrupted_tokens = original_tokens.copy()
        target_tokens = original_tokens.copy()
        task_labels = {}
        rtd_labels = None
        
        # 第一轮：执行无 mask 的任务
        for task in self.NO_MASK_TASKS:
            if task in self.task_types:
                if self.evaluation_enabled:
                    self.evaluation_metrics["task_counts"][task] += 1
                
                if task == "sentence_permutation":
                    corrupted_tokens, _ = self._sentence_permutation(corrupted_tokens)
                elif task == "document_rotation":
                    corrupted_tokens, _ = self._document_rotation(corrupted_tokens)
                elif task == "next_sentence_prediction":
                    corrupted_tokens, target_tokens, label = self._next_sentence_prediction(index, corrupted_tokens)
                    task_labels["next_sentence_prediction"] = label
                elif task == "sentence_order_prediction":
                    corrupted_tokens, target_tokens, label = self._sentence_order_prediction(index, corrupted_tokens)
                    task_labels["sentence_order_prediction"] = label
        
        # 第二轮：执行有 mask 的任务
        for task in self.MASK_TASKS:
            if task in self.task_types:
                if self.evaluation_enabled:
                    self.evaluation_metrics["task_counts"][task] += 1
                
                if task == "token_masking":
                    corrupted_tokens, _ = self._token_masking(corrupted_tokens)
                elif task == "token_deletion":
                    corrupted_tokens, _ = self._token_deletion(corrupted_tokens)
                elif task == "text_infilling":
                    corrupted_tokens, _ = self._text_infilling(corrupted_tokens)
                elif task == "replaced_token_detection":
                    corrupted_tokens, _, rtd_labels = self._replaced_token_detection(corrupted_tokens)
        
        # 更新评估指标
        if self.evaluation_enabled:
            self.evaluation_metrics["total_samples"] += 1
            self.evaluation_metrics["avg_source_length"] += len(corrupted_tokens)
            self.evaluation_metrics["avg_target_length"] += len(target_tokens)
        
        # 根据 model_style 返回不同格式
        def pad_and_truncate(seq, max_len, pad_value):
            if len(seq) > max_len:
                return seq[:max_len]
            return seq + [pad_value] * (max_len - len(seq))
        
        if self.model_style == "encoder_only":
            input_ids = [self.tokenizer.cls_token_id] + corrupted_tokens + [self.tokenizer.sep_token_id]
            labels = [-100] + target_tokens + [-100]
            attention_mask = [1] * len(input_ids)
            
            if rtd_labels is not None:
                padded_rtd_labels = [-100] + rtd_labels + [-100]
            else:
                padded_rtd_labels = None
            
            input_ids = pad_and_truncate(input_ids, self.max_length, self.tokenizer.pad_token_id)
            labels = pad_and_truncate(labels, self.max_length, -100)
            attention_mask = pad_and_truncate(attention_mask, self.max_length, 0)
            
            result = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "task_types": self.task_types
            }
            
            if padded_rtd_labels is not None:
                padded_rtd_labels = pad_and_truncate(padded_rtd_labels, self.max_length, -100)
                result["rtd_labels"] = torch.tensor(padded_rtd_labels, dtype=torch.long)
            
            for task, label in task_labels.items():
                result[f"{task}_label"] = label
            
            return result
        elif self.model_style == "decoder_only":
            input_ids = [self.tokenizer.bos_token_id] + corrupted_tokens + [self.tokenizer.eos_token_id]
            labels = [-100] + target_tokens + [-100]
            attention_mask = [1] * len(input_ids)
            
            if rtd_labels is not None:
                padded_rtd_labels = [-100] + rtd_labels + [-100]
            else:
                padded_rtd_labels = None
            
            input_ids = pad_and_truncate(input_ids, self.max_length, self.tokenizer.pad_token_id)
            labels = pad_and_truncate(labels, self.max_length, -100)
            attention_mask = pad_and_truncate(attention_mask, self.max_length, 0)
            
            result = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "task_types": self.task_types
            }
            
            if padded_rtd_labels is not None:
                padded_rtd_labels = pad_and_truncate(padded_rtd_labels, self.max_length, -100)
                result["rtd_labels"] = torch.tensor(padded_rtd_labels, dtype=torch.long)
            
            for task, label in task_labels.items():
                result[f"{task}_label"] = label
            
            return result
        else:
            encoder_input_ids = [self.tokenizer.bos_token_id] + corrupted_tokens + [self.tokenizer.eos_token_id]
            labels = [self.tokenizer.bos_token_id] + target_tokens + [self.tokenizer.eos_token_id]
            decoder_input_ids = [self.tokenizer.bos_token_id] + labels[:-1]
            
            encoder_attention_mask = [1] * len(encoder_input_ids)
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            if rtd_labels is not None:
                padded_rtd_labels = [-100] + rtd_labels + [-100]
            else:
                padded_rtd_labels = None
            
            encoder_input_ids = pad_and_truncate(encoder_input_ids, self.max_length, self.tokenizer.pad_token_id)
            decoder_input_ids = pad_and_truncate(decoder_input_ids, self.max_length, self.tokenizer.pad_token_id)
            labels = pad_and_truncate(labels, self.max_length, -100)
            encoder_attention_mask = pad_and_truncate(encoder_attention_mask, self.max_length, 0)
            decoder_attention_mask = pad_and_truncate(decoder_attention_mask, self.max_length, 0)
            
            result = {
                "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
                "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "encoder_attention_mask": torch.tensor(encoder_attention_mask, dtype=torch.long),
                "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long),
                "task_types": self.task_types
            }
            
            if padded_rtd_labels is not None:
                padded_rtd_labels = pad_and_truncate(padded_rtd_labels, self.max_length, -100)
                result["rtd_labels"] = torch.tensor(padded_rtd_labels, dtype=torch.long)
            
            for task, label in task_labels.items():
                result[f"{task}_label"] = label
            
            return result
    
    def get_evaluation_metrics(self):
        """获取数据质量评估指标"""
        if self.evaluation_metrics["total_samples"] == 0:
            return {}
        
        total = self.evaluation_metrics["total_samples"]
        
        task_coverage = {}
        for task in self.ALL_TASKS:
            count = self.evaluation_metrics["task_counts"][task]
            task_coverage[task] = {
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0
            }
        
        avg_source = self.evaluation_metrics["avg_source_length"] / total if total > 0 else 0
        avg_target = self.evaluation_metrics["avg_target_length"] / total if total > 0 else 0
        
        return {
            "total_samples": total,
            "task_coverage": task_coverage,
            "avg_source_length": avg_source,
            "avg_target_length": avg_target
        }


def create_unified_dataset(
    data_path,
    tokenizer,
    task_types=None,
    model_style="encoder_decoder",
    **kwargs
):
    """
    创建统一数据集的便捷函数
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        task_types: 要执行的任务列表，None 表示所有任务
        model_style: 模型风格，"encoder_only"、"decoder_only" 或 "encoder_decoder"
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=task_types,
        model_style=model_style,
        **kwargs
    )


def create_bert_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 BERT 预训练数据集的便捷函数
    
    BERT 预训练任务：MLM + NSP
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=["token_masking", "next_sentence_prediction"],
        model_style="encoder_only",
        **kwargs
    )


def create_albert_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 ALBERT 预训练数据集的便捷函数
    
    ALBERT 预训练任务：MLM + SOP
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=["token_masking", "sentence_order_prediction"],
        model_style="encoder_only",
        **kwargs
    )


def create_electra_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 ELECTRA 预训练数据集的便捷函数
    
    ELECTRA 预训练任务：RTD (Replaced Token Detection)
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=["replaced_token_detection"],
        model_style="encoder_only",
        **kwargs
    )


def create_roberta_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 RoBERTa 预训练数据集的便捷函数
    
    RoBERTa 预训练任务：仅 MLM
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=["token_masking"],
        model_style="encoder_only",
        **kwargs
    )


def create_t5_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 T5 预训练数据集的便捷函数
    
    T5 预训练任务：Span Corruption (Text Infilling)
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=["text_infilling"],
        model_style="encoder_decoder",
        **kwargs
    )


def create_bart_dataset(
    data_path,
    tokenizer,
    **kwargs
):
    """
    创建 BART 预训练数据集的便捷函数
    
    BART 预训练任务：5 种去噪任务
    (Token Masking, Token Deletion, Text Infilling, Sentence Permutation, Document Rotation)
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=[
            "token_masking",
            "token_deletion",
            "text_infilling",
            "sentence_permutation",
            "document_rotation"
        ],
        model_style="encoder_decoder",
        **kwargs
    )


def create_llama_dataset(
        data_path,
        tokenizer,
        task_types=None,
        **kwargs
):
    """
    创建 Llama 预训练数据集的便捷函数

    Llama 预训练任务：仅因果语言建模 (Causal Language Modeling)
    不需要特殊的掩码任务，只需要标准的 decoder-only 格式

    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        task_types: 要执行的任务列表，None 表示无特殊任务（仅 CLM）
        **kwargs: 其他参数

    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=task_types or [],
        model_style="decoder_only",
        **kwargs
    )


def create_qwen_dataset(
    data_path,
    tokenizer,
    task_types=None,
    **kwargs
):
    """
    创建 Qwen 预训练数据集的便捷函数
    
    Qwen 预训练任务：仅因果语言建模 (Causal Language Modeling)
    不需要特殊的掩码任务，只需要标准的 decoder-only 格式
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        task_types: 要执行的任务列表，None 表示无特殊任务（仅 CLM）
        **kwargs: 其他参数
    
    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=task_types or [],
        model_style="decoder_only",
        **kwargs
    )


def create_deepseek_dataset(
    data_path,
    tokenizer,
    task_types=None,
    **kwargs
):
    """
    创建 DeepSeek 预训练数据集的便捷函数

    DeepSeek 预训练任务：仅因果语言建模 (Causal Language Modeling)
    不需要特殊的掩码任务，只需要标准的 decoder-only 格式

    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        task_types: 要执行的任务列表，None 表示无特殊任务（仅 CLM）
        **kwargs: 其他参数

    Returns:
        UnifiedDataset 实例
    """
    return UnifiedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_types=task_types or [],
        model_style="decoder_only",
        **kwargs
    )

