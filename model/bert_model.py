import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from module.embedding_module import BertEmbedding
from module.encoder_module import TransformerEncoderLayer

# 骨干模型
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbedding(config)

        # 使用 ModuleList 堆叠多层 Encoder
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.act_fn = ACT2FN[config.hidden_act]

        # Pooler 层 (用于 token 做分类任务)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            self.act_fn()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] (1表示有效token，0表示pad)
        token_type_ids: [batch_size, seq_len] (0表示句子A，1表示句子B)
        """
        # 1. 处理 Mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 将 mask 扩展为 [batch, 1, 1, seq_len] 以便在 attention 中广播
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 2. 嵌入层
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # 3. 流经所有 Encoder 层
        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        sequence_output = hidden_states  # [batch, seq_len, hidden]

        # 4. Pooler 输出 (取第一个 token  经过线性层)
        first_token_tensor = sequence_output[:, 0, :]
        pooled_output = self.pooler(first_token_tensor)

        return sequence_output, pooled_output

# 输出头
# 预训练任务头
class BertMLMHead(nn.Module):
    """
    独立的 Masked Language Modeling (MLM) 头部
    """

    def __init__(self, config, word_embedding_weights):
        super().__init__()
        # 1. 线性变换 + 激活
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        # 2. Layer Norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 3. Decoder (映射回 vocab 空间)
        # 注意：bias=False，因为我们要在下面单独加一个 Parameter，
        # 这是为了在权重共享的同时，依然有独立的偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享 (Weight Tying)
        self.decoder.weight = word_embedding_weights

        # 4. 单独的 Bias
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, sequence_output):
        """
        sequence_output: [batch, seq_len, hidden]
        """
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)

        # 线性变换 + 偏置
        logits = self.decoder(x) + self.bias
        return logits

class BertNSPHead(nn.Module):
    """
    独立的 Next Sentence Prediction (NSP) 头部
    """
    def __init__(self, config):
        super().__init__()
        # 简单的线性分类器：Hidden -> 2 (IsNext / NotNext)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """
        pooled_output: [batch, hidden] (通常是  token 的输出)
        """
        logits = self.classifier(pooled_output)
        return logits


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, word_embedding_weights):
        super().__init__()

        # 实例化 MLM 头
        self.mlm_head = BertMLMHead(config, word_embedding_weights)

        # 实例化 NSP 头
        self.nsp_head = BertNSPHead(config)

    def forward(self, sequence_output, pooled_output):
        """
        sequence_output: 传给 MLM
        pooled_output: 传给 NSP
        """
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)

        return mlm_logits, nsp_logits

# BERT预训练任务：掩码语言建模MLM+下一句预测NSP
class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 实例化 BERT 主干模型
        self.bert = BertModel(config)

        # 2. 实例化预训练 Heads
        self.cls = BertPreTrainingHeads(
            config,
            word_embedding_weights=self.bert.embeddings.word_embeddings.weight
        )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,  # 新增：MLM 的标签
            next_sentence_label=None  # 新增：NSP 的标签
    ):
        """
        labels: (可选) [batch, seq_len]，被 mask 掉的 token 的真实 id，未被 mask 的位置填 -100
        next_sentence_label: (可选) [batch]，0 表示是下一句，1 表示不是下一句
        """
        # 1. 通过 BERT 主干
        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 通过预训练 Heads
        mlm_logits, nsp_logits = self.cls(sequence_output, pooled_output)

        # 3. 计算 Loss
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()

            # --- 计算 MLM Loss ---
            # view(-1) 将其展平为 [batch*seq_len, vocab_size] 和 [batch*seq_len]
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # --- 计算 NSP Loss ---
            nsp_loss = loss_fct(nsp_logits.view(-1, 2), next_sentence_label.view(-1))

            # --- 总 Loss ---
            total_loss = mlm_loss + nsp_loss

        # 4. 返回结果
        # 逻辑：如果有 loss 就先返回 loss，然后是 logits；否则只返回 logits
        output = (mlm_logits, nsp_logits)
        if total_loss is not None:
            output = (total_loss,) + output

        return output  # (total_loss, mlm_logits, nsp_logits) 或 (mlm_logits, nsp_logits)


# 掩码语言建模MLM任务
class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. 仅使用 MLM Head (移除 NSP Head)
        # 注意：传入 word_embeddings 权重以实现共享
        self.mlm_head = BertMLMHead(
            config,
            word_embedding_weights=self.bert.embeddings.word_embeddings.weight
        )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        """
        labels: (可选) [batch, seq_len]，用于计算 MLM 损失
        """
        # 1. 通过 BERT 主干
        # 这里我们只需要 sequence_output，不需要 pooled_output
        sequence_output, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 通过 MLM Head
        mlm_logits = self.mlm_head(sequence_output)

        # 3. 计算 Loss (如果提供了 labels)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只计算被 Mask 掉的 token 的损失，通常 labels 中 -100 表示忽略
            loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 返回格式：如果有 loss 先返回 loss，否则返回 logits
        output = (mlm_logits,)
        if loss is not None:
            output = (loss,) + output

        return output  # (loss), mlm_logits

# 下一句预测NSP任务
class BertForNextSentencePrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. 仅使用 NSP Head (移除 MLM Head)
        self.nsp_head = BertNSPHead(config)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            next_sentence_label=None
    ):
        """
        next_sentence_label: (可选) [batch]，0 表示是下一句，1 表示不是下一句
        """
        # 1. 通过 BERT 主干
        # 这里我们主要需要 pooled_output ( token 的输出)
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 通过 NSP Head
        nsp_logits = self.nsp_head(pooled_output)

        # 3. 计算 Loss (如果提供了 next_sentence_label)
        loss = None
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(nsp_logits.view(-1, 2), next_sentence_label.view(-1))

        # 4. 返回结果
        output = (nsp_logits,)
        if loss is not None:
            output = (loss,) + output

        return output  # (loss), nsp_logits

# 下游任务
# 文本分类任务/序列分类任务
class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels  # 分类的类别数

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. Dropout 层 (用于防止过拟合)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 3. 分类器 (将  token 的输出映射到 label 空间)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        """
        labels: (可选) [batch]
                - 如果是分类任务，通常是 LongTensor，取值范围 [0, num_labels-1]
                - 如果是回归任务 (如 STS-B)，通常是 FloatTensor
        """
        # 1. 通过 BERT 主干
        # 我们主要需要  token 的池化输出 pooled_output
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. Dropout + Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 3. 计算 Loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务 (例如：语义相似度打分)
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 分类任务 (例如：情感分析二分类/多分类)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 4. 返回结果
        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output  # (loss), logits


# 命名实体识别任务/序列标注任务
class BertForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels  # 标签类别数 (如 B-PER, I-PER, O 等)

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 3. 分类器 (将每个 Token 的输出映射到 label 空间)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        """
        labels: (可选) [batch, seq_len]
                注意：padding 部分的 label 应设为 -100，这样 CrossEntropyLoss 会自动忽略
        """
        # 1. 通过 BERT 主干
        # 我们主要需要 sequence_output (每个 Token 的输出)
        sequence_output, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. Dropout + Classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # 3. 计算 Loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 展平为 [batch*seq_len, num_labels] 和 [batch*seq_len]
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 4. 返回结果
        output = (logits,)
        if loss is not None:
            output = (loss,) + output

        return output  # (loss), logits

# 多项选择任务（如阅读理解、常识推理）
# 它会将问题与每个选项分别拼接，输入给 BERT，然后对每个选项的 token 输出进行打分。
class BertForMultipleChoice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 3. 分类器 (将  token 映射为 1 个分数)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        """
        input_ids: [batch, num_choices, seq_len]
        labels: (可选) [batch]，正确选项的索引 (0 到 num_choices-1)
        """
        batch_size, num_choices, seq_len = input_ids.shape

        # --- 1. 维度重塑 ---
        # 将 [batch, num_choices, seq_len] 展平为 [batch*num_choices, seq_len]
        # 这样才能一次性输入给 BERT
        flat_input_ids = input_ids.view(batch_size * num_choices, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_choices,
                                                  seq_len) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(batch_size * num_choices,
                                                  seq_len) if token_type_ids is not None else None

        # --- 2. 通过 BERT 主干 ---
        _, pooled_output = self.bert(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        # --- 3. 恢复维度并分类 ---
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch*num_choices, 1]

        # 恢复形状为 [batch, num_choices]
        reshaped_logits = logits.view(batch_size, num_choices)

        # --- 4. 计算 Loss ---
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits,)
        if loss is not None:
            output = (loss,) + output

        return output  # (loss), reshaped_logits

# 抽取式问答任务
# 它不生成文本，而是预测答案起始位置和答案结束位置的概率分布
class BertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. BERT 主干模型
        self.bert = BertModel(config)

        # 2. QA 输出层 (将每个 Token 的输出映射为 2 个 logits: start 和 end)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,  # 答案起始位置的真实索引
            end_positions=None  # 答案结束位置的真实索引
    ):
        """
        start_positions: (可选) [batch]
        end_positions: (可选) [batch]
        """
        # 1. 通过 BERT 主干
        # 我们需要 sequence_output (每个 Token 的输出)
        sequence_output, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 计算 Logits
        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]

        # 将 2 个维度拆开，分别代表 start 和 end
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)  # [batch, seq_len]

        # 3. 计算 Loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()

            # 分别计算起始位置和结束位置的 Loss
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 总 Loss 取平均
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits)
        if total_loss is not None:
            output = (total_loss,) + output

        return output  # (total_loss), start_logits, end_logits