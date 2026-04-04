import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            momentum: float = 0.1,  # 滑动平均的动量系数
            affine: bool = True     # 是否使用可学习的 gamma/beta
    ):
        super(BatchNorm, self).__init__()
        self.num_features = num_features  # 特征维度大小（对应 C）
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.gamma = None
        self.beta = None
        # 可学习参数（与 LayerNorm 一致）
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # 滑动平均统计量（测试时使用，不需要梯度）
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
    
    def _norm(self, x, broadcast_shape: list[int]):
        # 输入形状约定：(N, C, D1, D2, ...)，其中 C = num_features
        # 例如：
        # - 全连接层输入：(N, C)
        # 序列输入的时候要求提前交换维度，输出后再交换回来 (N, L, C) -> (N, C, L)
        # - 1D 序列输入：(N, C, L)
        # - 2D 图像输入：(N, C, H, W)
        # - 3D 视频输入：(N, C, D, H, W)

        # 确定要归一化的维度：除了特征维度 C（dim=1）之外的所有维度
        reduce_dims = [dim for dim in range(x.dim()) if dim != 1]

        if self.training:
            # 训练阶段：使用当前 batch 的均值和方差
            mean = x.mean(dim=reduce_dims, keepdim=False)  # (C,)
            var = x.var(dim=reduce_dims, unbiased=False, keepdim=False)  # (C,)

            # 更新滑动平均（无梯度）
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 测试阶段：使用训练阶段累积的滑动平均
            mean = self.running_mean
            var = self.running_var

        # 将 mean/var  reshape 为可广播的形状（与 input 维度对齐）
        # 例如 input 是 (N, C, H, W)，则 mean 变为 (1, C, 1, 1)
        mean = mean.view(broadcast_shape)
        var = var.view(broadcast_shape)

        # 归一化（与 LayerNorm 公式一致）
        return (x - mean) * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = self.num_features
        x = self._norm(x.float(), broadcast_shape)
        
        # 仿射变换（如果启用）
        if self.affine:
            gamma = self.gamma.view(broadcast_shape)
            beta = self.beta.view(broadcast_shape)
            x = gamma * x + beta

        return x

class LayerNorm(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            affine: bool = True
    ):
        super(LayerNorm, self).__init__()
        self.affine = affine
        self.gamma = None
        self.beta = None
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def _norm(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm(x.float())
        if self.affine:
            x = self.gamma * x + self.beta
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.gamma * self._norm(x.float())).type_as(x)


class WeightNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.WeightNormLinear = nn.utils.weight_norm(nn.Linear(in_features=in_features, out_features=out_features), name='weight', dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.WeightNormLinear(x)