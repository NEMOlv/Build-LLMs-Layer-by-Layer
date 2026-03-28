import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from model.bart_config import BartConfig
from module.embedding_module import BartEmbedding
from module.encoder_module import BartEncoder
from module.decoder_module import BartDecoder


class BartModel(nn.Module):
    """
    完整的 BART 模型（Encoder-Decoder 结构，权重共享）
    
    BART (Bidirectional and Auto-Regressive Transformers) 是一个通用的序列到序列模型，
    结合了双向编码器和自回归解码器的特点。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.encoder.embedding.embed_tokens.weight = self.decoder.embedding.embed_tokens.weight
        self.lm_head.weight = self.encoder.embedding.embed_tokens.weight
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None
    ):
        """
        Args:
            encoder_input_ids: 编码器输入 token id 序列，shape [batch_size, enc_seq_len]
            decoder_input_ids: 解码器输入 token id 序列，shape [batch_size, dec_seq_len]
            encoder_attention_mask: 编码器注意力掩码，shape [batch_size, enc_seq_len]
            decoder_self_attention_mask: 解码器自注意力掩码，shape [batch_size, dec_seq_len]
            decoder_cross_attention_mask: 解码器交叉注意力掩码，shape [batch_size, enc_seq_len]
        Returns:
            lm_logits: 语言模型 logits，shape [batch_size, dec_seq_len, vocab_size]
        """
        encoder_hidden = self.encoder(encoder_input_ids, attention_mask=encoder_attention_mask)
        decoder_hidden = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            self_attention_mask=decoder_self_attention_mask,
            cross_attention_mask=decoder_cross_attention_mask
        )
        lm_logits = self.lm_head(decoder_hidden)
        
        return lm_logits


# ========================================
# 预训练任务头
# ========================================

class BartDenoisingHead(nn.Module):
    """
    BART 去噪自编码头
    
    BART 的预训练任务都是去噪自编码任务，包括：
    1. Token Masking - Token 遮挡
    2. Token Deletion - Token 删除
    3. Text Infilling - 连续文本填空
    4. Sentence Permutation - 句子打乱
    5. Document Rotation - 文档旋转
    
    所有这些任务都使用相同的模型架构：从损坏的输入重构原始序列
    
    设计特点：
    - 输入：解码器序列输出 [batch, seq_len, d_model]
    - 输出：词汇表 logits [batch, seq_len, vocab_size]
    - 与词嵌入权重共享
    """
    def __init__(self, config, word_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.decoder.weight = word_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: 解码器输出，shape [batch, seq_len, d_model]
        Returns:
            logits: 重构 logits，shape [batch, seq_len, vocab_size]
        """
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        logits = self.decoder(x) + self.bias
        return logits


class BartPreTrainingHeads(nn.Module):
    """
    BART 预训练任务头集合
    
    BART 的预训练是去噪自编码任务，只需要一个序列重构头即可。
    各种损坏方式（Token Masking、Deletion、Infilling 等）在数据预处理阶段实现。
    """
    def __init__(self, config, word_embedding_weights):
        super().__init__()
        self.denoising_head = BartDenoisingHead(config, word_embedding_weights)
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: 解码器序列输出，shape [batch, seq_len, d_model]
        Returns:
            logits: 去噪重构 logits，shape [batch, seq_len, vocab_size]
        """
        logits = self.denoising_head(sequence_output)
        return logits


# ========================================
# 下游任务头
# ========================================

class BartClassificationHead(nn.Module):
    """
    BART 文本分类头
    
    用于文本分类、情感分析等任务
    
    设计特点：
    - 输入：解码器的<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出 [batch, d_model]
    - 输出：分类 logits [batch, num_labels]
    - 包含 dropout 层防止过拟合
    """
    def __init__(self, config, num_labels=None):
        super().__init__()
        self.num_labels = num_labels if num_labels is not None else config.num_labels
        self.dropout = nn.Dropout(config.classifier_dropout if config.classifier_dropout is not None else config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
    
    def forward(self, pooled_output):
        """
        Args:
            pooled_output: 解码器的<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 输出，shape [batch, d_model]
        Returns:
            logits: 分类 logits，shape [batch, num_labels]
        """
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BartTokenClassificationHead(nn.Module):
    """
    BART 序列标注头
    
    用于命名实体识别 (NER)、词性标注 (POS) 等任务
    
    设计特点：
    - 输入：解码器序列输出 [batch, seq_len, d_model]
    - 输出：每个 token 的标签 logits [batch, seq_len, num_labels]
    """
    def __init__(self, config, num_labels=None):
        super().__init__()
        self.num_labels = num_labels if num_labels is not None else config.num_labels
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, self.num_labels)
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: 解码器序列输出，shape [batch, seq_len, d_model]
        Returns:
            logits: token 分类 logits，shape [batch, seq_len, num_labels]
        """
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class BartQuestionAnsweringHead(nn.Module):
    """
    BART 问答任务头
    
    用于抽取式问答任务，预测答案的起始和结束位置
    
    设计特点：
    - 输入：解码器序列输出 [batch, seq_len, d_model]
    - 输出：起始位置 logits 和结束位置 logits [batch, seq_len, 2]
    """
    def __init__(self, config):
        super().__init__()
        self.qa_outputs = nn.Linear(config.d_model, 2)
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: 解码器序列输出，shape [batch, seq_len, d_model]
        Returns:
            start_logits: 起始位置 logits，shape [batch, seq_len]
            end_logits: 结束位置 logits，shape [batch, seq_len]
        """
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


class BartSummarizationHead(nn.Module):
    """
    BART 摘要生成头
    
    用于文本摘要、翻译等序列到序列生成任务
    
    设计特点：
    - 输入：解码器序列输出 [batch, seq_len, d_model]
    - 输出：词汇表 logits [batch, seq_len, vocab_size]
    - 与词嵌入权重共享
    """
    def __init__(self, config, word_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.decoder.weight = word_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: 解码器输出，shape [batch, seq_len, d_model]
        Returns:
            logits: 生成 logits，shape [batch, seq_len, vocab_size]
        """
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        logits = self.decoder(x) + self.bias
        return logits


# ========================================
# 完整任务模型类
# ========================================

class BartForPreTraining(nn.Module):
    """
    BART 预训练模型
    
    BART 的预训练是去噪自编码任务，支持多种损坏方式：
    - Token Masking: 随机遮挡一些 token
    - Token Deletion: 随机删除一些 token
    - Text Infilling: 用一个 <mask> 替换连续的多个 token
    - Sentence Permutation: 打乱句子顺序
    - Document Rotation: 旋转文档，随机选择一个 token 作为新开头
    
    所有这些任务都使用相同的模型架构从损坏的输入重构原始序列。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.cls = BartPreTrainingHeads(
            config,
            word_embedding_weights=self.model.encoder.embedding.embed_tokens.weight
        )
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        labels=None
    ):
        """
        Args:
            encoder_input_ids: 损坏的输入序列（经过 Token Masking/Deletion/Infilling 等处理）
            decoder_input_ids: 解码器输入（通常是原始序列右移）
            encoder_attention_mask: 编码器注意力掩码
            decoder_self_attention_mask: 解码器自注意力掩码
            decoder_cross_attention_mask: 解码器交叉注意力掩码
            labels: 原始未损坏的序列，用于计算重构损失
        Returns:
            (loss), logits
        """
        decoder_output = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask
        )
        
        logits = self.cls(decoder_output)
        
        total_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        output = (logits,)
        if total_loss is not None:
            output = (total_loss,) + output
        
        return output


class BartForSequenceClassification(nn.Module):
    """
    BART 文本分类模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(config)
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        labels=None
    ):
        decoder_output = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask
        )
        
        pooled_output = decoder_output[:, 0, :]
        logits = self.classification_head(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        output = (logits,)
        if loss is not None:
            output = (loss,) + output
        
        return output


class BartForTokenClassification(nn.Module):
    """
    BART 序列标注模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.token_classification_head = BartTokenClassificationHead(config)
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        labels=None
    ):
        decoder_output = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask
        )
        
        logits = self.token_classification_head(decoder_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        output = (logits,)
        if loss is not None:
            output = (loss,) + output
        
        return output


class BartForQuestionAnswering(nn.Module):
    """
    BART 问答模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.qa_head = BartQuestionAnsweringHead(config)
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        start_positions=None,
        end_positions=None
    ):
        decoder_output = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask
        )
        
        start_logits, end_logits = self.qa_head(decoder_output)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        output = (start_logits, end_logits)
        if total_loss is not None:
            output = (total_loss,) + output
        
        return output


class BartForConditionalGeneration(nn.Module):
    """
    BART 条件生成模型（摘要、翻译等）
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.generation_head = BartSummarizationHead(
            config,
            word_embedding_weights=self.model.encoder.embedding.embed_tokens.weight
        )
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
        labels=None
    ):
        decoder_output = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask
        )
        
        logits = self.generation_head(decoder_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        output = (logits,)
        if loss is not None:
            output = (loss,) + output
        
        return output
