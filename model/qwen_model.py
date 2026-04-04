from model.qwen_config import QwenConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union
import math
from torch.nn import CrossEntropyLoss, MSELoss
from module.ffn_module import SwiGLUFFN
from module.attention_module import GraphQueryAttention
from module.embedding_module import RotaryPositionalEmbedding
from module.norm_module import RMSNorm
from module.moe_module import MoELayer


class QwenBlock(nn.Module):
    """
    Qwen 统一 Transformer Block
    
    支持 FFN 和 MoE 两种模式，通过配置自动切换：
        - use_moe=False: 使用 SwiGLUFFN
        - use_moe=True: 使用 MoELayer
    """
    def __init__(self, layer_id: int, config: QwenConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        
        # 输入归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 自注意力机制 (GQA/MHA)
        self.self_attention = GraphQueryAttention(config)
        
        # 注意力后归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN 或 MoE (通过配置切换)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLUFFN(config)

    def forward(
            self,
            x,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. 输入归一化
        hidden_states = self.input_layernorm(x)
        
        # 2. 自注意力
        attn_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        
        # 3. 残差连接
        hidden_states = x + attn_output
        
        # 4. 注意力后归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 5. FFN 或 MoE + 残差连接
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states


class QwenModel(nn.Module):
    """
    Qwen 统一骨干模型
    
    支持 Qwen 1 到 Qwen 3.5 全系列，通过配置灵活切换
    """
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        
        # Token 嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Layers (支持 FFN/MoE 切换)
        self.layers = nn.ModuleList(
            [QwenBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        
        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 存储 MoE 辅助损失 (如果使用 MoE)
        self.aux_loss = 0.0

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # 1. Token 嵌入 + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 2. 流经所有 Transformer Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        
        # 3. 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 4. 收集 MoE 辅助损失 (如果使用 MoE)
        if self.config.use_moe:
            self.aux_loss = sum(
                layer.ffn.aux_loss for layer in self.layers 
                if hasattr(layer.ffn, 'aux_loss')
            )
        else:
            self.aux_loss = 0.0
        
        return hidden_states


class QwenForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Qwen 统一 Causal LM 模型
    
    支持预训练、推理、文本生成
    """
    config_class = QwenConfig

    def __init__(self, config: QwenConfig):
        super().__init__(config)
        self.config = config
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重共享 (Weight Tying)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # 1. 通过骨干模型
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # 2. 语言模型头
        logits = self.lm_head(hidden_states)

        # 3. 计算损失
        loss = None
        total_loss = None
        
        if labels is not None:
            # 主损失：Causal LM 损失
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # MoE 辅助损失 (如果使用 MoE)
            if self.config.use_moe and self.model.aux_loss > 0:
                total_loss = loss + self.model.aux_loss
            else:
                total_loss = loss

        output = CausalLMOutputWithPast(
            loss=total_loss if total_loss is not None else loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )

        return output


class QwenForSequenceClassification(nn.Module):
    """
    Qwen 统一序列分类/文本分类模型
    
    支持情感分析、文本分类等任务
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # 骨干模型
        self.model = QwenModel(config)
        
        # Dropout 和分类器
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        # 1. 通过骨干模型
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        # 2. 取最后一个 token 的输出作为句子表示
        pooled_output = hidden_states[:, -1, :]
        
        # 3. Dropout + 分类器
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 4. 计算损失
        loss = None
        total_loss = None
        
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 分类任务
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # MoE 辅助损失 (如果使用 MoE)
            if self.config.use_moe and self.model.aux_loss > 0:
                total_loss = loss + self.model.aux_loss
            else:
                total_loss = loss

        output = (logits,)
        if total_loss is not None:
            output = (total_loss,) + output
        elif loss is not None:
            output = (loss,) + output

        return output


class QwenForTokenClassification(nn.Module):
    """
    Qwen 统一 Token 分类/序列标注模型
    
    支持命名实体识别 (NER)、词性标注等任务
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # 骨干模型
        self.model = QwenModel(config)
        
        # Dropout 和分类器
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        # 1. 通过骨干模型
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        # 2. Dropout + 分类器 (对每个 token)
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)

        # 3. 计算损失
        loss = None
        total_loss = None
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # MoE 辅助损失 (如果使用 MoE)
            if self.config.use_moe and self.model.aux_loss > 0:
                total_loss = loss + self.model.aux_loss
            else:
                total_loss = loss

        output = (logits,)
        if total_loss is not None:
            output = (total_loss,) + output
        elif loss is not None:
            output = (loss,) + output

        return output


# 便捷工厂函数
def create_qwen_1_model(config_kwargs=None):
    """
    创建 Qwen 1.0 模型的便捷函数
    """
    config = QwenConfig.for_qwen_1(**(config_kwargs or {}))
    return QwenForCausalLM(config)


def create_qwen_1_5_model(config_kwargs=None):
    """
    创建 Qwen 1.5 模型的便捷函数
    """
    config = QwenConfig.for_qwen_1_5(**(config_kwargs or {}))
    return QwenForCausalLM(config)


def create_qwen_2_model(config_kwargs=None):
    """
    创建 Qwen 2.0 模型的便捷函数
    """
    config = QwenConfig.for_qwen_2(**(config_kwargs or {}))
    return QwenForCausalLM(config)


def create_qwen_2_5_model(config_kwargs=None):
    """
    创建 Qwen 2.5 模型的便捷函数
    """
    config = QwenConfig.for_qwen_2_5(**(config_kwargs or {}))
    return QwenForCausalLM(config)


def create_qwen_3_model(config_kwargs=None):
    """
    创建 Qwen 3.0 模型的便捷函数
    """
    config = QwenConfig.for_qwen_3(**(config_kwargs or {}))
    return QwenForCausalLM(config)


def create_qwen_3_5_model(config_kwargs=None):
    """
    创建 Qwen 3.5 模型的便捷函数
    """
    config = QwenConfig.for_qwen_3_5(**(config_kwargs or {}))
    return QwenForCausalLM(config)
