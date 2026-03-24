import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from module.embedding_module import BertEmbedding
from module.encoder_module import TransformerEncoderLayer
from model.albert_config import AlbertConfig


class AlbertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层，使用较小的嵌入维度
        self.embeddings = BertEmbedding(config)
        
        # 嵌入投影层：将 embedding_size 映射到 hidden_size
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        
        # 层共享：只创建一个 TransformerEncoderLayer，然后重复使用
        self.encoder_layer = TransformerEncoderLayer(config)
        
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Pooler 层
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            self.act_fn()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 嵌入层
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # 嵌入投影：将 embedding_size 映射到 hidden_size
        embedding_output = self.embedding_hidden_mapping_in(embedding_output)

        # 流经所有 Encoder 层（共享参数）
        hidden_states = embedding_output
        for _ in range(self.config.num_hidden_layers):
            hidden_states = self.encoder_layer(hidden_states, extended_attention_mask)

        sequence_output = hidden_states

        first_token_tensor = sequence_output[:, 0, :]
        pooled_output = self.pooler(first_token_tensor)

        return sequence_output, pooled_output


class AlbertMLMHead(nn.Module):
    def __init__(self, config, word_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 共享权重的解码器
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        
        # 权重共享：使用词嵌入的权重
        self.decoder.weight = word_embedding_weights

        # 嵌入投影：将 hidden_size 映射回 embedding_size
        self.embedding_hidden_mapping_out = nn.Linear(config.hidden_size, config.embedding_size)

        # 单独的 Bias
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, sequence_output):
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        
        # 投影回 embedding_size
        x = self.embedding_hidden_mapping_out(x)

        # 线性变换 + 偏置
        logits = self.decoder(x) + self.bias
        return logits


class AlbertSOPHead(nn.Module):
    """
    ALBERT 的句序预测 (SOP) 头部
    """
    def __init__(self, config):
        super().__init__()
        # 简单的线性分类器：Hidden -> 2 (Original Order / Swapped Order)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """
        pooled_output: [batch, hidden] (通常是 [CLS] token 的输出)
        """
        logits = self.classifier(pooled_output)
        return logits


class AlbertPreTrainingHeads(nn.Module):
    """
    ALBERT 预训练头部，包含 MLM 和 SOP 任务
    """
    def __init__(self, config, word_embedding_weights):
        super().__init__()

        # 实例化 MLM 头
        self.mlm_head = AlbertMLMHead(config, word_embedding_weights)

        # 实例化 SOP 头
        self.sop_head = AlbertSOPHead(config)

    def forward(self, sequence_output, pooled_output):
        """
        sequence_output: 传给 MLM
        pooled_output: 传给 SOP
        """
        mlm_logits = self.mlm_head(sequence_output)
        sop_logits = self.sop_head(pooled_output)

        return mlm_logits, sop_logits


class AlbertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.albert = AlbertModel(config)

        self.lm_head = AlbertMLMHead(
            config,
            word_embedding_weights=self.albert.embeddings.Token_Embeddings.weight
        )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        sequence_output, _ = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        mlm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        output = (mlm_logits,)
        if loss is not None:
            output = (loss,) + output

        return output


class AlbertForPreTraining(nn.Module):
    """
    ALBERT 预训练任务：掩码语言建模 (MLM) + 句序预测 (SOP)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 实例化 ALBERT 主干模型
        self.albert = AlbertModel(config)

        # 2. 实例化预训练 Heads
        self.cls = AlbertPreTrainingHeads(
            config,
            word_embedding_weights=self.albert.embeddings.Token_Embeddings.weight
        )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,  # MLM 的标签
            sentence_order_label=None  # SOP 的标签
    ):
        """
        labels: (可选) [batch, seq_len]，被 mask 掉的 token 的真实 id，未被 mask 的位置填 -100
        sentence_order_label: (可选) [batch]，0 表示原始顺序，1 表示颠倒顺序
        """
        # 1. 通过 ALBERT 主干
        sequence_output, pooled_output = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 通过预训练 Heads
        mlm_logits, sop_logits = self.cls(sequence_output, pooled_output)

        # 3. 计算 Loss
        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = CrossEntropyLoss()

            # --- 计算 MLM Loss ---
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # --- 计算 SOP Loss ---
            sop_loss = loss_fct(sop_logits.view(-1, 2), sentence_order_label.view(-1))

            # --- 总 Loss ---
            total_loss = mlm_loss + sop_loss

        # 4. 返回结果
        output = (mlm_logits, sop_logits)
        if total_loss is not None:
            output = (total_loss,) + output

        return output  # (total_loss, mlm_logits, sop_logits) 或 (mlm_logits, sop_logits)


class AlbertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        _, pooled_output = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output


class AlbertForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        sequence_output, _ = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output


class AlbertForMultipleChoice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.albert = AlbertModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        batch_size, num_choices, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_choices, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_choices,
                                                  seq_len) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(batch_size * num_choices,
                                                  seq_len) if token_type_ids is not None else None

        _, pooled_output = self.albert(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(batch_size, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits,)
        if loss is not None:
            output = (loss,) + output

        return output


class AlbertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.albert = AlbertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None
    ):
        sequence_output, _ = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

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