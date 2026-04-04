from transformers import PretrainedConfig


class QwenConfig(PretrainedConfig):
    """
    Qwen 系列模型统一配置类
    
    支持 Qwen 1.0、Qwen 1.5、Qwen 2.0、Qwen 2.5、Qwen 3.0、Qwen 3.5 全系列
    通过配置参数灵活切换不同版本的特性
    
    版本切换说明：
        - Qwen 1.5 → Qwen 1.0: 设置 num_key_value_heads = num_attention_heads (GQA → MHA)
        - Qwen 2.5 → Qwen 2.0: 设置 n_routed_experts = 4 (保持共享专家)
        - Qwen 3.x → Qwen 2.x: 设置 n_shared_experts = 1 (从无共享专家 → 有共享专家)
        - Qwen 2.x/3.x → Qwen 1.x: 设置 use_moe = False (MoE → FFN)
    """
    
    model_type = "qwen"
    
    # Qwen 全系列预设配置
    # Qwen 系列 MoE 演进：
    #   - Qwen 2.0: 引入 MoE，有 1 个共享专家
    #   - Qwen 2.5: 有 1 个共享专家
    #   - Qwen 3.0/3.5: 取消共享专家，回到纯路由专家设计
    PRESET_CONFIGS = {
        "qwen_1": {
            "num_key_value_heads": 8,  # MHA (same as attention heads)
            "use_moe": False,
            "rms_norm_eps": 1e-06,
        },
        "qwen_1_5": {
            "num_key_value_heads": 2,  # GQA
            "use_moe": False,
            "rms_norm_eps": 1e-06,
        },
        "qwen_2": {
            "num_key_value_heads": 2,  # GQA
            "use_moe": True,
            "n_shared_experts": 1,  # 有 1 个共享专家
            "n_routed_experts": 4,
            "rms_norm_eps": 1e-06,
        },
        "qwen_2_5": {
            "num_key_value_heads": 2,  # GQA
            "use_moe": True,
            "n_shared_experts": 1,  # 有 1 个共享专家
            "n_routed_experts": 8,
            "rms_norm_eps": 1e-06,
        },
        "qwen_3": {
            "num_key_value_heads": 2,  # GQA
            "use_moe": True,
            "n_shared_experts": 0,  # 无共享专家
            "n_routed_experts": 8,
            "rms_norm_eps": 1e-06,
        },
        "qwen_3_5": {
            "num_key_value_heads": 2,  # GQA
            "use_moe": True,
            "n_shared_experts": 0,  # 无共享专家
            "n_routed_experts": 16,
            "rms_norm_eps": 1e-06,
        },
    }

    def __init__(
            self,
            model_version: str = "qwen_1_5",
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = "silu",
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-06,
            rope_theta: int = 1000000,
            inference_rope_scaling: bool = False,
            flash_attention: bool = True,
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "softmax",
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            topk_prob_norm_eps: float = 1e-20,
            num_labels: int = 2,
            pad_token_id: int = 0,
            sliding_window: int = None,
            **kwargs,
    ):
        """
        初始化 Qwen 配置
        
        Args:
            model_version: 模型版本，可选:
                'qwen_1', 'qwen_1_5', 'qwen_2', 'qwen_2_5', 'qwen_3', 'qwen_3_5'
                如果传入预设版本，会自动加载预设配置，但可以手动覆盖
            
            # 基础模型参数
            dropout: Dropout 概率
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
            hidden_act: 隐藏层激活函数
            hidden_size: 隐藏层维度
            intermediate_size: FFN 中间层维度（默认 4 * hidden_size）
            max_position_embeddings: 最大位置编码长度
            num_attention_heads: 注意力头数量
            num_hidden_layers: Transformer 层数
            vocab_size: 词表大小
            
            # GQA 相关
            num_key_value_heads: KV 头数量
                - Qwen 1: 设置为 num_attention_heads (MHA)
                - Qwen 1.5+: 设置为 2 或 4 (GQA)
            
            # 归一化相关
            rms_norm_eps: RMSNorm 的 epsilon 值
                - Qwen 系列: 1e-06
            
            # RoPE 相关
            rope_theta: RoPE 的 theta 值
                - Qwen 系列: 1000000
            inference_rope_scaling: 是否使用推理时 RoPE 缩放
            
            # 注意力相关
            flash_attention: 是否使用 Flash Attention
            sliding_window: 滑动窗口大小（用于长文本优化）
            
            # MoE 相关
            use_moe: 是否使用 MoE (Mixture of Experts)
                - Qwen 1/1.5: False
                - Qwen 2/2.5/3/3.5: True
            
            num_experts_per_tok: 每个 token 选择的专家数量 (Top-K)
            n_routed_experts: 路由专家数量
            n_shared_experts: 共享专家数量
                - Qwen 2.0/2.5: 1 (有 1 个共享专家)
                - Qwen 3.0/3.5: 0 (无共享专家，官方取消了)
            
            scoring_func: 路由打分函数 ('softmax')
            aux_loss_alpha: 辅助损失权重
            seq_aux: 是否使用 sequence 级辅助损失
            norm_topk_prob: 是否对 top-k 权重归一化
            topk_prob_norm_eps: 归一化时的 epsilon
            
            # 下游任务相关
            num_labels: 分类任务的标签数量
            pad_token_id: Padding token ID
        """
        super().__init__(**kwargs)
        
        # 如果是预设版本，加载预设配置
        if model_version in self.PRESET_CONFIGS:
            preset = self.PRESET_CONFIGS[model_version]
            num_key_value_heads = preset.get("num_key_value_heads", num_key_value_heads)
            use_moe = preset.get("use_moe", use_moe)
            n_shared_experts = preset.get("n_shared_experts", n_shared_experts)
            n_routed_experts = preset.get("n_routed_experts", n_routed_experts)
            rms_norm_eps = preset.get("rms_norm_eps", rms_norm_eps)
        
        self.model_version = model_version
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.sliding_window = sliding_window
        self.topk_prob_norm_eps = topk_prob_norm_eps
        
        # 设置默认的 intermediate_size
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        
        # RoPE scaling 配置
        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
    
    @classmethod
    def for_qwen_1(cls, **kwargs):
        """
        创建 Qwen 1.0 配置的便捷方法
        
        特性:
            - MHA (Multi-Head Attention): num_key_value_heads = num_attention_heads
            - 无 MoE: use_moe = False
            - SwiGLU FFN
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_1", **kwargs)
    
    @classmethod
    def for_qwen_1_5(cls, **kwargs):
        """
        创建 Qwen 1.5 配置的便捷方法
        
        特性:
            - GQA (Grouped Query Attention): num_key_value_heads < num_attention_heads
            - 无 MoE: use_moe = False
            - SwiGLU FFN
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_1_5", **kwargs)
    
    @classmethod
    def for_qwen_2(cls, **kwargs):
        """
        创建 Qwen 2.0 配置的便捷方法
        
        特性:
            - GQA (Grouped Query Attention)
            - MoE 有共享专家: use_moe = True, n_shared_experts = 1
            - SwiGLU FFN (作为专家)
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_2", **kwargs)
    
    @classmethod
    def for_qwen_2_5(cls, **kwargs):
        """
        创建 Qwen 2.5 配置的便捷方法
        
        特性:
            - GQA (Grouped Query Attention)
            - MoE 有共享专家: use_moe = True, n_shared_experts = 1
            - SwiGLU FFN (作为专家)
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_2_5", **kwargs)
    
    @classmethod
    def for_qwen_3(cls, **kwargs):
        """
        创建 Qwen 3.0 配置的便捷方法
        
        特性:
            - GQA (Grouped Query Attention)
            - MoE 无共享专家: use_moe = True, n_shared_experts = 0
            - 更多路由专家: n_routed_experts = 8
            - SwiGLU FFN (作为专家)
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_3", **kwargs)
    
    @classmethod
    def for_qwen_3_5(cls, **kwargs):
        """
        创建 Qwen 3.5 配置的便捷方法
        
        特性:
            - GQA (Grouped Query Attention)
            - MoE 无共享专家: use_moe = True, n_shared_experts = 0
            - 更多路由专家: n_routed_experts = 16
            - SwiGLU FFN (作为专家)
            - RMSNorm
            - RoPE 位置编码
        """
        return cls(model_version="qwen_3_5", **kwargs)
