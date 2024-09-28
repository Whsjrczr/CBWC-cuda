import torch
import torch.nn as nn
import math
from culculate_tools import my_calculate_fan_in_and_fan_out
from normalization_scalingonly import *


class Linear_Weight_CenteredBN_Row(nn.Module):
    def __init__(self, inputSize, outputSize, flag_adjustScale=None, init_flag=None):
        super(Linear_Weight_CenteredBN_Row, self).__init__()

        self.weight = torch.Tensor(outputSize, inputSize)
        self.gradWeight = torch.Tensor(outputSize, inputSize)

        if flag_adjustScale is not None:
            self.flag_adjustScale = flag_adjustScale
        else:
            self.flag_adjustScale = False

        if init_flag is not None:
            self.init_flag = init_flag
        else:
            self.init_flag = 'RandInit'

        self.g = torch.Tensor(outputSize).fill_(1)

        if self.flag_adjustScale:
            self.gradG = torch.Tensor(outputSize)
            self.gradBias = torch.Tensor(outputSize)
            self.bias = torch.Tensor(outputSize).fill_(0)

        self.reset()

    def reset(self, stdv=None):
        if self.init_flag == 'RandInit':
            self.reset_RandInit(stdv)
        elif self.init_flag == 'OrthInit':
            self.reset_orthogonal(stdv)

        return self

    def reset_RandInit(self, stdv):
        if stdv:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))

        if torch.initial_seed():
            for i in range(self.weight.size(0)):
                self.weight[i].uniform_(-stdv, stdv)
                # self.bias[i] = torch.uniform(-stdv, stdv)
        else:
            self.weight.uniform_(-stdv, stdv)
            # self.bias.uniform_(-stdv, stdv)

    def reset_orthogonal(self):
        initScale = 1.1  # math.sqrt(2)

        M1 = torch.randn(self.weight.size(0), self.weight.size(0))
        M2 = torch.randn(self.weight.size(1), self.weight.size(1))

        n_min = min(self.weight.size(0), self.weight.size(1))

        # QR decomposition of random matrices ~ N(0, 1)
        Q1, R1 = torch.qr(M1)
        Q2, R2 = torch.qr(M2)

        self.weight.copy_(torch.mm(Q1.narrow(1, 0, n_min), Q2.narrow(0, 0, n_min)).mul(initScale))

    def updateOutput(self, input):
        if input.dim() == 2:
            nframe = input.size(0)
            nElement = self.output.numel()
            n_output = self.weight.size(0)
            n_input = self.weight.size(1)
            self.output.resize_(nframe, n_output)
            if self.output.numel() != nElement:
                self.output.zero_()
            self.addBuffer = self.addBuffer or input.new()
            self.addBuffer.resize_(nframe).fill_(1)

            self.mean = self.mean or input.new()
            self.std = self.std or input.new()

            self.W = self.W or input.new()
            self.W_hat = self.W_hat or input.new()
            self.W.resize_as_(self.weight)

            self.mean.mean_(self.weight, 1)
            self.weight.sub_(self.mean.view(n_output, 1).expand(n_output, n_input))

            self.std.resize_(n_output, 1).copy_(self.weight.norm(2, 2)).pow_(-1)

            self.W_hat.resize_as_(self.weight).copy_(self.weight).cmul_(
                self.std.view(n_output, n_input).expand(n_output, n_input))
            self.W.copy_(self.W_hat).cmul_(self.g.view(n_output, 1).expand(n_output, n_input))
            self.output.addmm_(0, self.output, 1, input, self.W.t())
            if self.flag_adjustScale:
                self.output.addr_(1, self.addBuffer, self.bias)
        else:
            raise ValueError('input must be a 2D tensor')

        return self.output

    def updateGradInput(self, input, gradOutput):
        if hasattr(self, 'gradInput'):
            nElement = self.gradInput.numel()
            self.gradInput.resize_as_(input)
            if self.gradInput.numel() != nElement:
                self.gradInput.zero_()

        if input.dim() == 2:
            self.gradInput.addmm_(0, 1, gradOutput, self.W)
        else:
            raise ValueError('input must be vector or matrix')

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1.0):
        if input.dim() == 2:
            n_output = self.weight.size(0)
            n_input = self.weight.size(1)
            self.gradW = self.gradW or input.new()
            self._scale = self._scale or input.new()
            self._scale.resize_as_(self.std).copy_(self.std).cmul_(self.g)
            self.gradW.resize_(gradOutput.size(1), input.size(1))
            self.gradW.mm_(gradOutput.t(), input)

            self.gradWeight.mul_(self.W_hat, self.gradW)
            self.mean.sum_(self.gradWeight, 1)
            self.gradWeight.copy_(-self.W_hat).cmul_(self.mean.expand(n_output, n_input))

            self.mean.mean_(self.gradW, 1)
            self.gradWeight.add_(self.gradW).add_(-self.mean.expand(n_output, n_input))

            self.gradWeight.mul_(self._scale.expand(n_output, n_input))

            if self.flag_adjustScale:
                self.gradBias.addmv_(scale, gradOutput.t(), self.addBuffer)
                self.W_hat.mul_(self.gradW)
                self.gradG.sum_(self.W_hat, 1)
        else:
            raise ValueError('input must be vector or matrix')

    def parameters(self):
        if self.flag_adjustScale:
            return [self.weight, self.g, self.bias], [self.gradWeight, self.gradG, self.gradBias]
        else:
            return [self.weight], [self.gradWeight]

    def sharedAccUpdateGradParameters(self, input, gradOutput):
        self.accGradParameters(input, gradOutput)

    def __str__(self):
        return type(self).__name__ + f'({self.weight.size(1)} -> {self.weight.size(0)})'


if __name__ == '__main__':

    dbn = Linear_Weight_CenteredBN_Row(16, 8)
    dby = nn.Linear(16, 8)
    weight = torch.nn.Parameter(dbn.weight)
    dby.weight = weight
    bias = torch.nn.Parameter(dbn.bias)
    dby.bias = bias
    # d = nn.LayerNorm(8, elementwise_affine=False)
    d = myLayerNorm(8, elementwise_affine=False)
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

