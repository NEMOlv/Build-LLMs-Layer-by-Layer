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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 输入形状约定：(N, C, D1, D2, ...)，其中 C = num_features
        # 例如：
        # - 全连接层输入：(N, C)
        # 序列输入的时候要求提前交换维度，输出后再交换回来 (N, L, C) -> (N, C, L)
        # - 1D 序列输入：(N, C, L)
        # - 2D 图像输入：(N, C, H, W)
        # - 3D 视频输入：(N, C, D, H, W)

        # 确定要归一化的维度：除了特征维度 C（dim=1）之外的所有维度
        reduce_dims = [dim for dim in range(input.dim()) if dim != 1]

        if self.training:
            # 训练阶段：使用当前 batch 的均值和方差
            mean = input.mean(dim=reduce_dims, keepdim=False)  # (C,)
            var = input.var(dim=reduce_dims, unbiased=False, keepdim=False)  # (C,)

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
        broadcast_shape = [1] * input.dim()
        broadcast_shape[1] = self.num_features
        mean = mean.view(broadcast_shape)
        var = var.view(broadcast_shape)

        # 归一化（与 LayerNorm 公式一致）
        output = (input - mean) * torch.rsqrt(var + self.eps)

        # 仿射变换（如果启用）
        if self.affine:
            gamma = self.gamma.view(broadcast_shape)
            beta = self.beta.view(broadcast_shape)
            output = gamma * output + beta

        return output

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, unbiased=False, keepdim=True)
        # rsqrt = 1 / sqrt
        output = (input - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            output = self.gamma * output + self.beta
        return output

