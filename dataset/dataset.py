from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
#  input：[BOS, t1, t2, t3, EOS, PAD, PAD]
#  label：[BOS, t1, t2, t3, EOS, -100, -100]
#  BOS 预测 t1
#  t1 预测 t2
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        input_ids = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
        )

        # Step 4：右移input_ids构造labels
        labels = input_ids[1:] + [self.tokenizer.pad_token_id]
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        # [BOS] + a + b +   c   + [EOS]
        #   a   + b + c + [EOS]
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # Step 5：张量化
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels
