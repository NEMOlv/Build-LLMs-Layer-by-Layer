import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
     symbols:
        b: batch
        s: sequence
        h: hidden
        q: query
        k: key
        v: value
        dim: dimension
        proj: projection
        attn: attention
"""

# Token向量
# 实际上通常都会直接使用nn.Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, config):
        super(TokenEmbedding, self).__init__(config.vocab_size, config.hidden_dim)

# sin/cos位置编码向量 (transformer论文称其为Sinusoidal Positional Embedding)
# 但实际上是偶数用正弦sin, 奇数用余弦cos
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super(SinusoidalPositionalEmbedding, self).__init__()
        # 创建位置编码矩阵
        self.encoding = torch.zeros(config.max_len, config.hidden_dim, device=config.device)
        self.encoding.requires_grad_(False)
        # 创建一个0~max_len-1的张量, 将其转为float tensor, 并扩展一个维度
        # pos shape: [1, max_len]
        # pos是token的下标数组
        pos = torch.arange(0, config.max_len, device=config.device).float().unsqueeze(1)
        # 2i是token特征的下标数组
        _2i = torch.arange(0, config.hidden_dim, 2, device=config.device)
        # 偶数用正弦
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / config.hidden_dim)))
        # 奇数用余弦
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / config.hidden_dim)))

    def forward(self, input):
        # 返回s_len长度为input.shape[1]的位置编码
        return self.encoding[:input.shape[1], :]


class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = SinusoidalPositionalEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input):
        tok_embedding = self.token_embedding(input)
        pos_embedding = self.position_embedding(input)
        output = self.dropout(tok_embedding + pos_embedding)

        return output


class BertEmbedding(nn.Module):
    """
    严格对齐结构图命名的BERT嵌入层
    变量名完全匹配图中标识：Token Embeddings/Segment Embeddings/Position Embeddings
    """
    def __init__(self, config):
        super().__init__()

        # 使用 embedding_size（ALBERT）或默认 hidden_size
        embedding_dim = getattr(config, 'embedding_size', config.hidden_size)

        # 1. Token Embeddings（对应图中黄色Token Embeddings行）
        self.Token_Embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=config.padding_idx
        )

        # 2. Segment Embeddings（对应图中绿色Segment Embeddings行，E_A/E_B）
        self.Segment_Embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=embedding_dim
        )

        # 3. Position Embeddings（对应图中白色Position Embeddings行，E0~E10）
        self.Position_Embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=embedding_dim
        )

        # 图中相加后的归一化和Dropout
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=config.layer_norm_eps)
        self.Dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor = None):
        """
        Args:
            input_ids: 输入token的id序列，对应图中输入token列，shape [batch_size, seq_len]
            segment_ids: 句子段id，对应图中E_A/E_B，shape [batch_size, seq_len]
                         不传则默认全0（单句子场景）
        Returns:
            embeddings: 最终融合嵌入，shape [batch_size, seq_len, hidden_size]
        """
        # 获取序列长度和设备信息
        seq_len = input_ids.size(1)
        device = input_ids.device

        # ---------------------- 生成Position IDs（对应图中E0~Eseq_len-1） ----------------------
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # ---------------------- 处理Segment IDs（对应图中E_A/E_B） ----------------------
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        # ---------------------- 计算三种嵌入（完全匹配图中命名） ----------------------
        Token_Emb = self.Token_Embeddings(input_ids)          # Token Embeddings输出
        Segment_Emb = self.Segment_Embeddings(segment_ids)    # Segment Embeddings输出
        Position_Emb = self.Position_Embeddings(position_ids) # Position Embeddings输出

        # ---------------------- 三种嵌入相加（对应图中+号） ----------------------
        Embeddings = Token_Emb + Segment_Emb + Position_Emb

        # ---------------------- 归一化和Dropout（对应图中后续处理） ----------------------
        Embeddings = self.LayerNorm(Embeddings)
        Embeddings = self.Dropout(Embeddings)

        return Embeddings


class BartEmbedding(nn.Module):
    """
    BART 嵌入层，使用绝对位置编码
    
    BART 使用标准的绝对位置编码，与 BERT 类似但不需要 segment 嵌入。
    仅包含 token 嵌入和位置嵌入。
    """
    def __init__(self, config):
        super().__init__()
        
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx
        )
        
        self.embed_positions = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.d_model
        )
        
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: 输入 token id 序列，shape [batch_size, seq_len]
        Returns:
            embeddings: 嵌入向量，shape [batch_size, seq_len, d_model]
        """
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_emb = self.embed_tokens(input_ids)
        position_emb = self.embed_positions(position_ids)
        
        embeddings = token_emb + position_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.Dropout(embeddings)
        
        return embeddings