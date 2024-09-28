import torch
import torch.nn as nn
import math


class WeightNorm_Center(nn.Module):
    def __init__(self, module, output_dim, flag_adjust_scale=False):
        super(WeightNorm_Center, self).__init__()
        self.module = module
        assert hasattr(module, 'weight')

        if module.bias is not None:
            self.bias = module.bias
            self.grad_bias = module.grad_bias
        else:
            self.bias = None
            self.grad_bias = None

        if flag_adjust_scale is not None:
            self.flag_adjust_scale = flag_adjust_scale
        else:
            self.flag_adjust_scale = False

        self.grad_weight = module.grad_weight
        self.weight = module.weight
        self.output_dim = output_dim or 1

        # track the non-output weight dimensions
        self.other_dims = 1
        for i in range(self.weight.dim()):
            if i + 1 != self.output_dim:
                self.other_dims *= self.weight.size(i + 1)

        # view size for weight norm 2D calculations
        self.view_in = (self.weight.size(self.output_dim - 1), self.other_dims)

        # view size back to original weight
        self.view_out = self.weight.size()
        self.weight_size = self.weight.size()

        # bubble outputDim size up to the front
        for i in range(self.outputDim - 1, 0, -1):
            self.viewOut[i], self.viewOut[i + 1] = self.viewOut[i + 1], self.viewOut[i]

        # weight is reparametrized to decouple the length from the direction
        # such that w = g * ( v / ||v|| )
        self.v = torch.Tensor(*self.view_in)
        if self.flag_adjust_scale:
            self.g = torch.Tensor(self.view_in[0])
        else:
            self.g = torch.Tensor(self.view_in[0]).fill_(1)

        self._norm = torch.Tensor(self.view_in[0])
        self._scale = torch.Tensor(self.view_in[0])

        # gradient of v
        self.grad_v = torch.Tensor(*self.view_in)

        # gradient of g
        self.grad_g = torch.Tensor(self.view_in[0]).zero_()

        self.reset_init()

    def permute_in(self, input_tensor):
        ans = input_tensor
        for i in range(self.output_dim - 1, 0, -1):
            ans = ans.transpose(i, i + 1)
        return ans

    def permute_out(self, input_tensor):
        ans = input_tensor
        for i in range(1, self.output_dim):
            ans = ans.transpose(i, i + 1)
        return ans

    def reset_init(self):
        self.v.normal_(0, math.sqrt(2 / self.view_in[1]))
        if self.flag_adjust_scale:
            self.g.fill_(1)
        if self.bias is not None:
            self.bias.zero_()

    def evaluate(self):
        if self.training:
            self.update_weight()
        super(WeightNorm_Center, self).evaluate()

    def update_weight(self):
        # view to 2D when weight norm container operates
        self.grad_v.copy_(self.permute_in(self.weight))
        self.grad_v = self.grad_v.view(*self.view_in)

        # -- | | w | |
        self._norm.norm_(self.v, 2, 1).pow_(2).add_(10e-5).sqrt_()
        self.gradV.copy_(self.v)
        self._scale.copy_(self.g).cdiv_(self._norm)
        self.gradV.mul_(self._scale.view(self.viewIn[0], 1)
                        .expand(self.viewIn[0], self.viewIn[1]))

        self.grad_v = self.grad_v.view(*self.view_out)

        self.weight.copy_(self.permute_out(self.grad_v))

    def forward(self, input):
        if self.training:
            self.update_weight()
        output = self.module(input)
        return output

    def backward(self, input, grad_output, scale=1):
        self.module.zero_grad()
        self.module.backward(input, grad_output, scale)
        grad_weight = self.module.grad_weight
        self.weight.copy_(self.permute_in(self.weight))
        self.grad_v.copy_(self.permute_in(grad_weight))
        self.weight = self.weight.view(*self.view_in)

        norm = self._norm.view(self.view_in[0], 1).expand(*self.view_in)
        scale = self._scale.view(self.view_in[0], 1).expand(*self.view_in)

        self.weight.copy_(self.grad_v)
        self.weight.mul_(self.v).cdiv_(norm)
        self.grad_g.sum_(self.weight, 1)

        self.weight.copy_(self.v)
        self.weight.mul_(scale).cdiv_(norm)
        self.weight.mul_(self.grad_g.view(self.view_in[0], 1).expand(*self.view_in))

        self.grad_v.add_(-1, self.weight)

        self.grad_v = self.grad_v.view(*self.view_out)
        self.weight = self.weight.view(*self.view_out)
        self.module.grad_weight.copy_(self.permute_out(self.grad_v))

    def update_parameters(self, lr):
        self.module.update_parameters(lr)
        self.v.add_(-lr, self.grad_v)
        if self.flag_adjust_scale:
            self.g.add_(-lr, self.grad_g)

    def parameters(self):
        if self.bias:
            if self.flag_adjust_scale:
                return [self.v, self.g, self.bias], [self.grad_v, self.grad_g, self.grad_bias]
            else:
                return [self.v, self.bias], [self.grad_v, self.grad_bias]
        else:
            if self.flag_adjust_scale:
                return [self.v, self.g], [self.grad_v, self.grad_g]
            else:
                return [self.v], [self.grad_v]

    def write(self, file):
        weight = self.module.weight
        grad_weight = self.module.grad_weight
        self.weight = None
        self.grad_weight = None
        self.module.weight = None
        self.module.grad_weight = None
        if not self.weight_size:
            self.weight_size = weight.size()

        super(WeightNorm_Center, self).write(file)

        self.module.weight = weight
        self.module.grad_weight = grad_weight
        self.weight = weight
        self.grad_weight = grad_weight

    def read(self, file):
        super(WeightNorm_Center, self).read(file)

        if not self.weight:
            self.module.weight = self.v.new(self.weight_size)
            self.module.grad_weight = self.v.new(self.weight_size)
            self.weight = self.module.weight
            self.grad_weight = self.module.grad_weight
            self.update_weight()
            self.grad_weight.copy_(self.permute_out(self.grad_v))