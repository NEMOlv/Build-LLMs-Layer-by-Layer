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
