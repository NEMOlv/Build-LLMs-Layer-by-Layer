from transformers import PretrainedConfig


class T5Config(PretrainedConfig):
    """
    T5模型配置类，继承自Hugging Face的PretrainedConfig基类。
    该类定义了T5模型的所有超参数和配置项，支持模型架构的灵活定制。
    
    T5（Text-to-Text Transfer Transformer）是一种通用的文本生成模型，
    将所有NLP任务都建模为文本到文本的转换问题。
    
    设计考量：
    1. 继承自PretrainedConfig，便于与Hugging Face生态系统集成
    2. 提供丰富的参数默认值，开箱即用
    3. 支持多种命名约定的参数映射，提高兼容性
    4. 可配置的层归一化位置（Pre-LN）
    """
    
    model_type = "t5"

    def __init__(
            self,
            vocab_size: int = 32128,
            d_model: int = 512,
            d_kv: int = 64,
            num_layers: int = 6,
            num_decoder_layers: int = None,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-6,
            initializer_factor: float = 1.0,
            feed_forward_proj: str = "relu",
            is_encoder_decoder: bool = True,
            use_cache: bool = True,
            pad_token_id: int = 0,
            eos_token_id: int = 1,
            decoder_start_token_id: int = 0,
            relative_attention_num_buckets: int = 32,
            relative_attention_max_distance: int = 128,
            classifier_dropout: float = None,
            num_labels: int = 2,
            LN_mode: str = 'Pre-LN',
            hidden_size: int = None,
            num_hidden_layers: int = None,
            num_attention_heads: int = None,
            intermediate_size: int = None,
            hidden_act: str = "relu",
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            attention_dropout: float = 0.1,
            resid_dropout: float = 0.1,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            padding_idx: int = 0,
            bos_token_id: int = 0,
            position_embedding_type: str = "relative",
            ffn_dim: int = None,
            dropout: float = 0.1,
            num_encoder_layers: int = None,
            attn_dropout_prob: float = 0.1,
            embedding_size: int = 128,
            qa_dropout: float = None,
            **kwargs,
    ):
        """
        初始化T5配置类。
        
        核心参数说明：
        - vocab_size: 词汇表大小，T5默认使用32128个词表
        - d_model: 模型隐藏层维度，即嵌入向量维度
        - d_kv: 键/值投影维度，通常为d_model / num_heads
        - num_layers: 编码器/解码器层数
        - num_heads: 注意力头数
        - d_ff: 前馈网络中间层维度
        
        参数映射设计：
        为了兼容不同命名约定，实现了多组参数的自动映射，
        例如hidden_size -> d_model, num_hidden_layers -> num_layers等。
        这一设计提高了代码的灵活性和兼容性。
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.LN_mode = LN_mode
        
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else num_layers
        self.num_attention_heads = num_attention_heads if num_attention_heads is not None else num_heads
        self.intermediate_size = intermediate_size if intermediate_size is not None else d_ff
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_epsilon
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.position_embedding_type = position_embedding_type
        self.ffn_dim = ffn_dim if ffn_dim is not None else d_ff
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else num_layers
        self.attn_dropout_prob = attn_dropout_prob
        self.embedding_size = embedding_size
        self.qa_dropout = qa_dropout
