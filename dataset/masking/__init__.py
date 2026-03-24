"""掩码策略包

包含统一的掩码策略实现，支持：
- 静态掩码 (BERT 风格)
- 动态掩码 (RoBERTa 风格)
"""

from .masking import Masking

__all__ = ["Masking"]
