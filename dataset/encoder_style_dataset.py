import torch
from torch.utils.data import Dataset
import os
import random
from datasets import load_dataset
from dataset.masking import Masking

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EncoderStyleDataset(Dataset):
    """
    统一的编码器风格预训练数据集
    
    支持不同的编码器风格：
    - BERT 风格: 使用静态掩码，特殊 token 为 [CLS] 和 [SEP]
    - RoBERTa 风格: 使用动态掩码，特殊 token 为 <s> 和 </s>
    - 其他编码器风格: 可通过参数配置
    """

    def __init__(
            self,
            data_path,
            tokenizer,
            max_length=512,
            mlm_probability=0.15,
            mask_probability=0.8,
            random_probability=0.1,
            special_tokens_mask=None,
            masking_strategy="dynamic",  # "static" (BERT) or "dynamic" (RoBERTa/ALBERT)
            encoder_style="roberta",  # "bert", "roberta", or "albert"
    ):
        """
        Args:
            data_path: 预训练数据路径，JSONL 格式 {"text": "文本内容"}
            tokenizer: Hugging Face tokenizer
            max_length: 最大序列长度
            mlm_probability: 整体掩码概率，默认 0.15
            mask_probability: 每个mask_token的用[MASK]遮掩的概率，默认 0.8
            random_probability: 每个mask_token的用随机token遮掩的概率，默认 0.1
            special_tokens_mask: 特殊 token 掩码
            masking_strategy: 掩码策略，"static" (BERT 风格) 或 "dynamic" (RoBERTa/ALBERT 风格)
            encoder_style: 编码器风格，"bert", "roberta", 或 "albert"
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.special_tokens_mask = special_tokens_mask
        self.masking_strategy = masking_strategy
        self.encoder_style = encoder_style
        self.samples = load_dataset("json", data_files=data_path, split="train")
        
        # 初始化掩码策略
        self.masker = Masking(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            mask_probability=mask_probability,
            random_probability=random_probability,
            strategy=masking_strategy
        )
        print(f"使用{masking_strategy}掩码策略 ({'BERT' if masking_strategy == 'static' else 'RoBERTa'} 风格)")
        print(f"编码器风格: {encoder_style}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个样本并进行处理
        
        Returns:
            input_ids: 输入 token IDs
            labels: 标签 token IDs (-100 表示忽略)
            attention_mask: 注意力掩码
        """
        sample = self.samples[index]

        # 分词处理，添加相应的特殊 token
        tokenized = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        special_tokens_mask = torch.tensor(tokenized["special_tokens_mask"], dtype=torch.bool)

        # 使用掩码策略生成掩码
        input_ids, labels = self.masker.mask(
            input_ids=input_ids,
            special_tokens_mask=special_tokens_mask,
            sample_index=index  # 静态掩码会使用这个索引进行缓存
        )

        return input_ids, labels, attention_mask



# 便捷工厂函数
def create_bert_dataset(data_path, tokenizer, **kwargs):
    """
    创建 BERT 风格的数据集
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        BERT 风格的数据集
    """
    return EncoderStyleDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        masking_strategy="static",  # BERT 使用静态掩码
        encoder_style="bert",
        **kwargs
    )

def create_roberta_dataset(data_path, tokenizer, **kwargs):
    """
    创建 RoBERTa 风格的数据集
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        RoBERTa 风格的数据集
    """
    return EncoderStyleDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        masking_strategy="dynamic",  # RoBERTa 使用动态掩码
        encoder_style="roberta",
        **kwargs
    )


def create_albert_dataset(data_path, tokenizer, **kwargs):
    """
    创建 ALBERT 风格的数据集
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        **kwargs: 其他参数
    
    Returns:
        ALBERT 风格的数据集
    """
    return EncoderStyleDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        masking_strategy="dynamic",  # ALBERT 使用动态掩码
        encoder_style="albert",
        **kwargs
    )


class ALLDataset(EncoderStyleDataset):
    """
    支持多种任务类型的数据集：
    - MLM: 仅进行掩码语言建模
    - MLMWithNSP: MLM + 下一句预测
    - MLMWithSOP: MLM + 句序预测
    
    使用 [CLS] 作为分类标识符，[SEP] 作为分隔标识符
    """

    def __init__(
            self,
            data_path,
            tokenizer,
            max_length=512,
            mlm_probability=0.15,
            mask_probability=0.8,
            random_probability=0.1,
            task_type="MLM",
            nsp_probability=0.5,
            sop_probability=0.5,
            masking_strategy="dynamic",
            encoder_style="roberta",
    ):
        super().__init__(
            data_path,
            tokenizer,
            max_length,
            mlm_probability,
            mask_probability,
            random_probability,
            masking_strategy=masking_strategy,
            encoder_style=encoder_style
        )
        self.task_type = task_type
        self.nsp_probability = nsp_probability
        self.sop_probability = sop_probability

    def _get_random_text(self, exclude_index):
        """
        从数据集中随机获取一个文本，排除指定索引
        
        Args:
            exclude_index: 需要排除的索引
            
        Returns:
            随机抽取的文本
        """
        random_index = random.randint(0, len(self.samples) - 1)
        while random_index == exclude_index and len(self.samples) > 1:
            random_index = random.randint(0, len(self.samples) - 1)
        return str(self.samples[random_index]["text"])

    def nsp_process(self, index, original_text):
        """
        处理 NSP 任务
        
        Args:
            index: 样本索引
            original_text: 原始文本
            
        Returns:
            processed_text: 处理后的文本
            nsp_label: NSP 标签
        """
        if "[SEP]" not in original_text:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        is_negative = random.random() < self.nsp_probability
        
        if not is_negative:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        text_parts = original_text.split("[SEP]")
        text_parts = [p.strip() for p in text_parts if p.strip()]
        
        if len(text_parts) < 2:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        first_sentence = text_parts[0]
        random_text = self._get_random_text(index)
        random_parts = random_text.split("[SEP]")
        random_parts = [p.strip() for p in random_parts if p.strip()]
        second_sentence = random_parts[0]
        
        processed_text = f"{first_sentence}[SEP]{second_sentence}"
        return processed_text, torch.tensor(1, dtype=torch.long)

    def sop_process(self, index, original_text):
        """
        处理 SOP 任务
        
        Args:
            index: 样本索引
            original_text: 原始文本
            
        Returns:
            processed_text: 处理后的文本
            sop_label: SOP 标签
        """
        if "[SEP]" not in original_text:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        should_exchange = random.random() < self.sop_probability
        
        if not should_exchange:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        text_parts = original_text.split("[SEP]")
        text_parts = [p.strip() for p in text_parts if p.strip()]
        
        if len(text_parts) < 2:
            return original_text, torch.tensor(0, dtype=torch.long)
        
        processed_text = f"{text_parts[1]}[SEP]{text_parts[0]}"
        return processed_text, torch.tensor(1, dtype=torch.long)

    def __getitem__(self, index):
        """
        获取单个样本，根据任务类型进行相应处理
        
        Returns:
            根据任务类型返回不同的结果：
            - MLM: (input_ids, labels, attention_mask)
            - MLMWithNSP: (input_ids, labels, attention_mask, nsp_label)
            - MLMWithSOP: (input_ids, labels, attention_mask, sop_label)
        """
        sample = self.samples[index]
        original_text = str(sample["text"])
        
        task_label = None
        
        if self.task_type == "MLM":
            processed_text = original_text
        elif self.task_type == "MLMWithNSP":
            processed_text, task_label = self.nsp_process(index, original_text)
        elif self.task_type == "MLMWithSOP":
            processed_text, task_label = self.sop_process(index, original_text)
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
        
        tokenized = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        special_tokens_mask = torch.tensor(tokenized["special_tokens_mask"], dtype=torch.bool)
        
        input_ids, labels = self.masker.mask(
            input_ids=input_ids,
            special_tokens_mask=special_tokens_mask,
            sample_index=index
        )
        
        if self.task_type == "MLM":
            return input_ids, labels, attention_mask
        else:
            return input_ids, labels, attention_mask, task_label


def create_all_dataset(data_path, tokenizer, task_type="MLM", **kwargs):
    """
    创建支持多种任务类型的数据集
    
    Args:
        data_path: 数据路径
        tokenizer: tokenizer
        task_type: 任务类型 ("MLM", "MLMWithNSP", 或 "MLMWithSOP"
        **kwargs: 其他参数
    
    Returns:
        ALLDataset 实例
    """
    return ALLDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        task_type=task_type,
        **kwargs
    )

