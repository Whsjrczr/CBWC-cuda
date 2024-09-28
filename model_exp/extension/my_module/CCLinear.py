import math
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import extension as ext
from .normalization_scalingonly import *


def my_calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return fan_in

# 这里CCLinear可以直接替代nn.Linear函数
# TODO: 记得写GN对饮的GCCLinear 主体部分已完成，仍有bug
class CCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device="cuda:0", dtype=None) -> None:
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
            self.register_parameter('bias', **factory_kwargs)
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
'''

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
        self.compute_weight = torch.zeros_like(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.compute_bias = torch.zeros_like(self.bias)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('compute_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.update_weight()
        if self.bias is not None:
            fan_in = my_calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            self.update_bias()


    def update_weight(self):
        column_means = torch.mean(self.weight, dim=0)
        self.compute_weight = torch.sub(self.weight, column_means)

    def update_bias(self):
        bias_mean = torch.mean(self.bias, dim=0)
        self.compute_bias = torch.sub(self.bias, bias_mean)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self.update_weight()
            if self.bias is not None:
                self.update_bias()
        return F.linear(input, self.compute_weight, self.compute_bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
'''
'''
class GCCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features','num_groups']
    in_features: int
    out_features: int
    num_groups: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if out_features % num_groups != 0:
            raise ValueError('out_features must be divisible by num_groups')

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
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
            fan_in = my_calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # column_means = torch.mean(self.weight, dim=0)
        # compute_weight = torch.sub(self.weight, column_means)
        # bias_mean = torch.mean(self.bias, dim=0)
        # compute_bias = torch.sub(self.bias, bias_mean)
        # return F.linear(input, compute_weight, compute_bias)
        group_number = self.num_groups
        num_channels = self.out_features // group_number
        # Reshape weight and bias into groups
        weight_groups = self.weight.view(self.in_features, group_number, num_channels)
        bias_groups = self.bias.view(group_number, num_channels) if self.bias is not None else None

        # Compute mean of each group along the channel dimension
        weight_means = torch.mean(weight_groups, dim=2, keepdim=True)
        bias_means = torch.mean(bias_groups,dim=1, keepdim=True) if bias_groups is not None else None

        # Subtract group means from each group's weight and bias
        compute_weight = torch.sub(weight_groups, weight_means)
        compute_bias = torch.sub(bias_groups, bias_means) if bias_means is not None else None

        compute_weight = compute_weight.view(self.out_features, self.in_features)
        compute_bias = compute_bias.view(self.out_features, -1) if compute_bias is not None else None

        return F.linear(input, compute_weight, compute_bias)

    def extra_repr(self) -> str:
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )

'''


if __name__ == '__main__':
    '''
    dbn = GCCLinear(16, 8, 2)
    dby = nn.Linear(16, 8)
    dby.weight = dbn.weight
    dby.bias = dbn.bias
    d = nn.GroupNorm(2, 8, affine=False)
    c = SOGroupNorm(2, 8, affine=False)
    x = torch.randn(4, 16)

    '''
    dbn = CCLinear(16, 8)
    dby = nn.Linear(16, 8)
    dby.weight = dbn.weight
    dby.bias = dbn.bias
    # d = nn.LayerNorm(8, elementwise_affine=False)
    d = ext.myLayerNorm(8, elementwise_affine=False)
    c = SOLayerNorm(8, elementwise_affine=False)
    x = torch.randn(4, 16)

    print("orgin")
    print(x)
    print("CCL")
    y = dbn(x)
    print(y)
    print("CCL+LN")
    z = d(y)
    print(z)
    print("CCL+SOLN")
    y = c(y)
    print(y)
    print("L")
    # x = torch.ones(4, 16)
    y = dby(x)
    print(y)
    print("L+LN")
    y = d(y)
    print(y)
    print("L+SOLN")
    y = c(y)
    print(y)
    '''
    y = y.view(y.size(0), dbn.num_groups, y.size(1) // dbn.num_groups, *y.size()[2:])
    y = y.view(y.size(0), dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(1,2))/y.size(2)
    #print('train mode:', z.diag())
    print('z_ins:', z)
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.transpose(0,1))/y.size(1)
    print('z_batch:', z)
    # print(__file__)'''