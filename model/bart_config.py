from transformers import PretrainedConfig


class BartConfig(PretrainedConfig):
    """
    BART 模型配置类，继承自 Hugging Face 的 PretrainedConfig 基类。
    
    BART (Bidirectional and Auto-Regressive Transformers) 是一个通用的序列到序列模型，
    结合了双向编码器和自回归解码器的特点。
    
    设计特点：
    - 使用绝对位置编码（absolute position embeddings）
    - 使用标准的多头注意力（MHA），不使用 GQA
    - Pre-LN 架构
    - 支持编码器和解码器层数不同的配置
    """
    model_type = "bart"

    def __init__(
            self,
            vocab_size: int = 50265,
            d_model: int = 1024,
            num_layers: int = 12,
            num_decoder_layers: int = None,
            num_attention_heads: int = 16,
            d_ff: int = 4096,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-5,
            initializer_factor: float = 1.0,
            hidden_act: str = "gelu",
            is_encoder_decoder: bool = True,
            use_cache: bool = True,
            pad_token_id: int = 1,
            eos_token_id: int = 2,
            bos_token_id: int = 0,
            decoder_start_token_id: int = 2,
            max_position_embeddings: int = 1024,
            classifier_dropout: float = None,
            num_labels: int = 2,
            LN_mode: str = 'Pre-LN',
            hidden_size: int = None,
            num_hidden_layers: int = None,
            intermediate_size: int = None,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            attention_dropout: float = 0.1,
            resid_dropout: float = 0.1,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            ffn_dim: int = None,
            dropout: float = 0.1,
            num_encoder_layers: int = None,
            attn_dropout_prob: float = 0.1,
            position_embedding_type: str = "absolute",
            **kwargs,
    ):
        """
        初始化 BART 配置类。
        
        核心参数说明：
        - vocab_size: 词汇表大小，BART 默认使用 50265 个词表
        - d_model: 模型隐藏层维度，即嵌入向量维度
        - num_layers: 编码器/解码器层数
        - num_decoder_layers: 解码器层数（如果为 None，则与 num_layers 相同）
        - num_attention_heads: 注意力头数
        - d_ff: 前馈网络中间层维度
        
        设计特点：
        - position_embedding_type: "absolute"，BART 使用绝对位置编码
        - 支持参数映射，提高兼容性
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_position_embeddings = max_position_embeddings
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.LN_mode = LN_mode
        
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else num_layers
        self.intermediate_size = intermediate_size if intermediate_size is not None else d_ff
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_epsilon
        self.ffn_dim = ffn_dim if ffn_dim is not None else d_ff
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else num_layers
        self.attn_dropout_prob = attn_dropout_prob
        self.position_embedding_type = position_embedding_type
