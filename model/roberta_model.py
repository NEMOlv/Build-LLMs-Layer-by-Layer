import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from module.embedding_module import BertEmbedding
from module.encoder_module import TransformerEncoderLayer
from model.roberta_config import RobertaConfig


class RobertaModel(nn.Module):
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


class RobertaLMHead(nn.Module):
    def __init__(self, config, word_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.decoder.weight = word_embedding_weights

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, sequence_output):
        x = self.dense(sequence_output)
        x = self.act_fn(x)
        x = self.layer_norm(x)

        logits = self.decoder(x) + self.bias
        return logits


class RobertaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.roberta = RobertaModel(config)

        self.lm_head = RobertaLMHead(
            config,
            word_embedding_weights=self.roberta.embeddings.Token_Embeddings.weight
        )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None
    ):
        sequence_output, _ = self.roberta(
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


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

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
        _, pooled_output = self.roberta(
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


class RobertaForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

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
        sequence_output, _ = self.roberta(
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


class RobertaForMultipleChoice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.roberta = RobertaModel(config)

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

        _, pooled_output = self.roberta(
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


class RobertaForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.roberta = RobertaModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None
    ):
        sequence_output, _ = self.roberta(
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
