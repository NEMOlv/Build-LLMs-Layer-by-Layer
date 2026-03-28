import torch.nn as nn
from transformers.activations import ACT2FN


class TransformerFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.ffn_dim is None:
            ffn_dim = int(config.hidden_size * 8 / 3)
            config.ffn_dim = 64 * ((ffn_dim + 64 - 1) // 64)

        self.up_proj = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, input):
        output = self.down_proj(self.dropout(self.act_fn(self.up_proj(input))))
        return output


class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.up_proj_1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj_2 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

        dropout = getattr(config, "dropout", 0.0)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, input):
        gated = self.act_fn(self.up_proj_2(input)) * self.up_proj_1(input)
        output = self.dropout(self.down_proj(gated))
        return output
