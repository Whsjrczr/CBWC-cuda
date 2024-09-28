import torch
import torch.nn as nn
import numbers
from typing import Tuple
from torch.nn import functional as F
from tri_rmsnorm.kernel.rms_normalization_kernel import (
    _rms_norm_fwd_fused,
    _rms_norm_bwd_dx_fused,
)

class RMSNormFunctionCustomKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _rms_norm_fwd_fused[(M,)](x, y, weight, bias, rstd, x.stride(0), N, eps, BLOCK_SIZE=1024)
        ctx.save_for_backward(x, weight, bias, rstd)
        ctx.eps = eps
        ctx.N = N
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, rstd = ctx.saved_tensors
        eps = ctx.eps
        N = ctx.N
        M = x.shape[0]
        dx = torch.empty_like(x)
        _dw = torch.empty_like(weight)
        _db = torch.empty_like(bias)
        locks = torch.zeros(2 * 32, dtype=torch.int32, device=x.device)
        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            _db,
            x,
            weight,
            bias,
            rstd,
            locks,
            x.stride(0),
            N,
            eps,
            GROUP_SIZE_M=32,
            BLOCK_SIZE_N=1024,
        )
        return dx, _dw, _db, None

class myLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super(myLayerNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor):
        mean = torch.mean(input_tensor, dim=1, keepdim=True)
        var = torch.var(input_tensor, dim=1, unbiased=False, keepdim=True)
        normalized_tensor = (input_tensor-mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            normalized_tensor = normalized_tensor * self.weight + self.bias
        return normalized_tensor

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SOLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super(SOLayerNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor):
        return RMSNormFunctionCustomKernel.apply(input_tensor, self.weight, self.bias, self.eps)
        
    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SOGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor):
        # 将输入张量沿通道维度分成多个组
        groups = torch.chunk(input_tensor, self.num_groups, dim=1)
        scaled_groups = []
        for group in groups:
            # 计算组内沿通道维度的标准差
            var = torch.var(group, dim=1, unbiased=False, keepdim=True)
            # 对组内的每个样本进行放缩
            scaled_group = group / torch.sqrt(var + self.eps)
            if self.affine:
                scaled_group = scaled_group * self.weight + self.bias
            scaled_groups.append(scaled_group)
        # 将放缩后的组合并成affi一个张量
        scaled_tensor = torch.cat(scaled_groups, dim=1)
        return scaled_tensor

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={ne}'.format(**self.__dict__)

# if __name__ == '__main__':