import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import numbers
from typing import Tuple


class CCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            column_means = torch.mean(self.weight, dim=0)
            self.weight = Parameter(torch.sub(self.weight, column_means))
            if self.bias != None:
                bias_mean = torch.mean(self.bias, dim=0)
                self.bias = Parameter(torch.sub(self.bias, bias_mean))
        return F.linear(input, self.weight, self.bias)

    def eval(self):
        column_means = torch.mean(self.weight, dim=0)
        self.weight = Parameter(torch.sub(self.weight, column_means))
        if self.bias != None:
            bias_mean = torch.mean(self.bias, dim=0)
            self.bias = Parameter(torch.sub(self.bias, bias_mean))
        return self.train(False)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class CCLinear_repara(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.v_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.empty(out_features, **factory_kwargs)
            self.v_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        column_means = torch.mean(self.v_weight, dim=0)
        self.weight = torch.sub(self.v_weight, column_means)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.v_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.v_bias, -bound, bound)
            bias_mean = torch.mean(self.v_bias, dim=0)
            self.bias = torch.sub(self.v_bias, bias_mean)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self._center_weight_bias
        return F.linear(input, self.weight, self.bias)

    def eval(self):
        self._center_weight_bias
        return self.train(False)
    
    def _center_weight_bias(self):
        column_means = torch.mean(self.v_weight, dim=0)
        self.weight = torch.sub(self.v_weight, column_means)
        if self.bias != None:
            bias_mean = torch.mean(self.v_bias, dim=0)
            self.bias = torch.sub(self.v_bias, bias_mean)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class CClinear_flag(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    weight_update_flag: bool
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight_update_flag = True
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.v_weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.empty(out_features, **factory_kwargs)
            self.v_bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.weight_update_flag = False
        init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        column_means = torch.mean(self.v_weight, dim=0)
        self.weight = torch.sub(self.v_weight, column_means)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.v_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.v_bias, -bound, bound)
            bias_mean = torch.mean(self.v_bias, dim=0)
            self.bias = torch.sub(self.v_bias, bias_mean)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_update_flag:
            column_means = torch.mean(self.weight, dim=0)
            self.weight = Parameter(torch.sub(self.weight, column_means))
            if self.bias != None:
                bias_mean = torch.mean(self.bias, dim=0)
                self.bias = Parameter(torch.sub(self.bias, bias_mean))
            self.weight_update_flag = False
        return F.linear(input, self.weight, self.bias)

    def backward(self, grad: Tensor) -> Tensor:
        self.weight_update_flag=True
        return grad

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
