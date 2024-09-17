import torch
import torch.nn as nn
import numbers
from typing import Tuple
from torch.nn import functional as F
from cuda_time_test.cbwc import *

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
        var = torch.var(input_tensor, dim=1, unbiased=False, keepdim=True)
        normalized_tensor = input_tensor / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            normalized_tensor = normalized_tensor * self.weight + self.bias
        return normalized_tensor

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


if __name__ == '__main__':
    dbn = CCLinear(16, 8).eval()
    dby = nn.Linear(16, 8).eval()
    dby.weight = dbn.weight
    dby.bias = dbn.bias
    d = nn.LayerNorm(8, elementwise_affine=False).eval()
    c = SOLayerNorm(8, elementwise_affine=False).eval()
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