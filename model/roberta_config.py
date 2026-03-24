from transformers import PretrainedConfig


class RobertaConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
            self,
            vocab_size: int = 50265,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            intermediate_size: int = 3072,
            hidden_act: str = "gelu",
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            attention_dropout: float = 0.1,
            resid_dropout: float = 0.1,
            max_position_embeddings: int = 514,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            padding_idx: int = 1,
            bos_token_id: int = 0,
            eos_token_id: int = 2,
            position_embedding_type: str = "absolute",
            use_cache: bool = True,
            classifier_dropout: float = None,
            num_labels: int = 2,
            LN_mode: str = 'Pre-LN',
            ffn_dim: int = None,
            dropout: float = 0.1,
            num_encoder_layers: int = 12,
            attn_dropout_prob: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.LN_mode = LN_mode
        self.ffn_dim = ffn_dim if ffn_dim is not None else intermediate_size
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.attn_dropout_prob = attn_dropout_prob

