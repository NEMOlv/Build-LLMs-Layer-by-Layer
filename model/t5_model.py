import torch.nn as nn
from module.encoder_module import T5Encoder
from module.decoder_module import T5Decoder



class T5Model(nn.Module):
    """完整的 T5 模型（Encoder-Decoder 结构，权重共享）"""
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, relative_position_bucket_size=32, relative_position_max_distance=128):
        super().__init__()
        # 编码器和解码器
        self.encoder = T5Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, relative_position_bucket_size, relative_position_max_distance)
        self.decoder = T5Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, relative_position_bucket_size, relative_position_max_distance)
        
        # 权重共享：Encoder Embedding = Decoder Embedding = LM Head
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embedding.weight

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_mask=None, decoder_self_mask=None, decoder_cross_mask=None):
        encoder_hidden = self.encoder(encoder_input_ids, encoder_mask)
        decoder_hidden = self.decoder(decoder_input_ids, encoder_hidden, decoder_self_mask, decoder_cross_mask)
        lm_logits = self.lm_head(decoder_hidden)
        return lm_logits



