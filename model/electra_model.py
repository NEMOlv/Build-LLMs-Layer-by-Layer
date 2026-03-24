import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.activations import ACT2FN
from module.embedding_module import BertEmbedding
from module.encoder_module import TransformerEncoderLayer
from model.electra_config import ElectraConfig


class ElectraGenerator(nn.Module):
    """
    ELECTRA Generator：小型 MLM 模型，用于生成被 mask 的 token
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        generator_config = self._create_generator_config(config)
        self.embeddings = BertEmbedding(generator_config)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(generator_config) for _ in range(config.generator_num_hidden_layers)]
        )

        self.act_fn = ACT2FN[config.hidden_act]

        self.pooler = nn.Sequential(
            nn.Linear(generator_config.hidden_size, generator_config.hidden_size),
            self.act_fn()
        )

    def _create_generator_config(self, config):
        """
        创建 Generator 配置，按比例缩小
        """
        generator_config = ElectraConfig(
            vocab_size=config.vocab_size,
            hidden_size=int(config.hidden_size * config.generator_size),
            num_hidden_layers=config.generator_num_hidden_layers,
            num_attention_heads=max(1, int(config.num_attention_heads * config.generator_size)),
            intermediate_size=int(config.intermediate_size * config.generator_size),
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            attention_dropout=config.attention_dropout,
            resid_dropout=config.resid_dropout,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            padding_idx=config.padding_idx,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            position_embedding_type=config.position_embedding_type,
            use_cache=config.use_cache,
            LN_mode=config.LN_mode,
            ffn_dim=int(config.ffn_dim * config.generator_size) if config.ffn_dim else None,
            dropout=config.dropout,
            num_encoder_layers=config.generator_num_hidden_layers,
            attn_dropout_prob=config.attn_dropout_prob,
            embedding_size=int(getattr(config, 'embedding_size', config.hidden_size) * config.generator_size)
        )
        return generator_config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        sequence_output = hidden_states

        first_token_tensor = sequence_output[:, 0, :]
        pooled_output = self.pooler(first_token_tensor)

        return sequence_output, pooled_output


class ElectraDiscriminator(nn.Module):
    """
    ELECTRA Discriminator：判别每个 token 是否被替换
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbedding(config)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.act_fn = ACT2FN[config.hidden_act]

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            self.act_fn()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        sequence_output = hidden_states

        first_token_tensor = sequence_output[:, 0, :]
        pooled_output = self.pooler(first_token_tensor)

        return sequence_output, pooled_output


class ElectraMLMHead(nn.Module):
    """
    Generator 的 MLM 头部
    """

    def __init__(self, config, generator_config, word_embedding_weights):
        super().__init__()
        generator_hidden_size = int(config.hidden_size * config.generator_size)

        self.dense = nn.Linear(generator_hidden_size, generator_hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_norm = nn.LayerNorm(generator_hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(generator_hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = word_embedding_weights

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, sequence_output):
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)

        logits = self.decoder(x) + self.bias
        return logits


class ElectraDiscriminatorHead(nn.Module):
    """
    Discriminator 的判别头部
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, sequence_output):
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        logits = self.classifier(x)
        return logits


class ElectraForPreTraining(nn.Module):
    """
    ELECTRA 预训练模型，包含 Generator 和 Discriminator
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.generator = ElectraGenerator(config)
        self.discriminator = ElectraDiscriminator(config)

        generator_config = self.generator._create_generator_config(config)
        self.generator_lm_head = ElectraMLMHead(
            config,
            generator_config,
            word_embedding_weights=self.generator.embeddings.Token_Embeddings.weight
        )

        self.discriminator_head = ElectraDiscriminatorHead(config)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        """
        labels: MLM 的标签，被 mask 的位置填真实 token id，其他填 -100
        """
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generator_sequence_output, _ = self.generator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        generator_logits = self.generator_lm_head(generator_sequence_output)

        if labels is not None:
            masked_positions = (labels != -100)
            if masked_positions.any():
                sampled_tokens = torch.multinomial(
                    torch.softmax(generator_logits[masked_positions], dim=-1),
                    num_samples=1
                ).squeeze(-1)

                discriminator_input_ids = input_ids.clone()
                discriminator_input_ids[masked_positions] = sampled_tokens

                discriminator_labels = torch.zeros_like(input_ids, dtype=torch.float)
                discriminator_labels[masked_positions] = (sampled_tokens != labels[masked_positions]).float()
            else:
                discriminator_input_ids = input_ids
                discriminator_labels = torch.zeros_like(input_ids, dtype=torch.float)
        else:
            discriminator_input_ids = input_ids
            discriminator_labels = None

        discriminator_sequence_output, _ = self.discriminator(
            discriminator_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        discriminator_logits = self.discriminator_head(discriminator_sequence_output)

        total_loss = None
        if labels is not None and discriminator_labels is not None:
            loss_fct_mlm = CrossEntropyLoss()
            loss_fct_disc = BCEWithLogitsLoss()

            active_loss_positions = (labels != -100)
            generator_loss = loss_fct_mlm(
                generator_logits[active_loss_positions],
                labels[active_loss_positions]
            )

            active_disc_positions = (attention_mask == 1)
            discriminator_loss = loss_fct_disc(
                discriminator_logits[active_disc_positions].squeeze(-1),
                discriminator_labels[active_disc_positions]
            )

            total_loss = generator_loss + discriminator_loss

        output = (generator_logits, discriminator_logits)
        if total_loss is not None:
            output = (total_loss,) + output

        return output


class ElectraForSequenceClassification(nn.Module):
    """
    ELECTRA 用于序列分类
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.discriminator = ElectraDiscriminator(config)

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
        _, pooled_output = self.discriminator(
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


class ElectraForTokenClassification(nn.Module):
    """
    ELECTRA 用于 token 分类
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.discriminator = ElectraDiscriminator(config)

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
        sequence_output, _ = self.discriminator(
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


class ElectraForQuestionAnswering(nn.Module):
    """
    ELECTRA 用于问答任务
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.discriminator = ElectraDiscriminator(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None
    ):
        sequence_output, _ = self.discriminator(
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
