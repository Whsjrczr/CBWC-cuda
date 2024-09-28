import torch
import torch.nn as nn
import math

class NormLinear_new(nn.Module):
    def __init__(self, inputSize, outputSize, affine=False):
        super(NormLinear_new, self).__init__()

        self.weight = torch.Tensor(outputSize, inputSize)
        self.bias = torch.Tensor(outputSize)
        self.gradWeight = torch.Tensor(outputSize, inputSize)
        self.gradBias = torch.Tensor(outputSize)

        self.proMatrix = torch.eye(inputSize)
        self.mean = torch.Tensor(inputSize)
        self.mean.fill_(0)

        self.isDeleteActivation = True
        self.useSVD = True
        self.useCenteredEstimation = True

        self.FIM = torch.Tensor()
        self.conditionNumber = []
        self.epcilo = 1e-100
        self.updateFIM_flag = False

        self.printInterval = 50
        self.count = 0
        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)
        return self

    def updateOutput(self, input):
        assert input.dim() == 2, 'only mini-batch supported (2D tensor), got ' + str(input.dim()) + 'D tensor instead'
        nframe = input.size(0)
        self.output = torch.zeros(nframe, self.bias.size(0))
        self.addBuffer = torch.ones(nframe)
        self.input = input.clone()
        self.buffer_1 = input.clone()

        self.buffer = torch.zeros_like(input)
        self.W = torch.zeros_like(self.weight)
        self.buffer_1.addmm_(self.mean, input, alpha=-1)
        self.W.addmm_(self.proMatrix, self.weight)

        self.output.addmm_(self.buffer_1, self.W.t())
        self.output.addmv_(self.bias, self.addBuffer)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is not None:
            self.gradInput = torch.zeros_like(input)
            self.W.addmm_(self.proMatrix, self.weight)

            if self.affine:
                self.W.mul_(self.weight_affine.unsqueeze(1))

            self.gradInput.addmm_(gradOutput, self.W)

        if self.updateFIM_flag:
            batchNumber = input.size(0)
            self.buffer_FIM = input.new()
            self.buffer = input.new()
            self.normalizedInput = input.new()
            self.buffer_FIM.add_(input, alpha=-1, other=self.mean)
            self.normalizedInput.resize_as_(self.buffer_FIM)
            self.normalizedInput.addmm_(self.buffer_FIM, self.proMatrix.t())

            eleNumber = gradOutput.size(1) * self.normalizedInput.size(1)
            self.FIM = torch.zeros(eleNumber, eleNumber)
            self.buffer_FIM.resize_(gradOutput.size(1), self.normalizedInput.size(1))

            for i in range(batchNumber):
                self.buffer_FIM.addmm_(gradOutput[i].unsqueeze(1).t(), self.normalizedInput[i].unsqueeze(0))
                self.buffer.resize_(eleNumber, 1).copy_(self.buffer_FIM.view(eleNumber, 1))
                self.FIM.addmm_(self.buffer, self.buffer.t())

            self.FIM.mul_(1. / batchNumber)

            _, self.buffer_FIM, _ = torch.svd(self.FIM)
            self.buffer_FIM.add_(self.epcilo)
            conditionNumber = torch.abs(torch.max(self.buffer_FIM) / torch.min(self.buffer_FIM))
            print('Normlinear module: conditionNumber=', conditionNumber)
            self.conditionNumber.append(conditionNumber)

            if self.debug:
                print('eignValue: \n')
                print(self.buffer_FIM)

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self.buffer = torch.zeros_like(self.mean)
        self.buffer_1 = torch.zeros_like(input)
        self.buffer.copy_(self.mean)
        self.buffer_1.add_(input, alpha=-1, other=self.buffer)

        self.buffer.mm_(self.buffer_1, self.proMatrix.t())
        self.gradWeight.addmm_(gradOutput.t(), self.buffer, alpha=scale)
        self.gradBias.addmv_(gradOutput.t(), self.addBuffer, alpha=scale)

    def updatePromatrix(self, epsilo):
        print('------------update Norm--------------')
        self.buffer_sigma = self.input.new()

        self.centered_input = self.input.new()

        self.b = self.input.new()
        self.b.fill_(0)

    def updatePromatrix(self, epsilo):
        print('------------update Norm--------------')
        self.buffer_sigma = self.buffer_sigma or self.input.new()
        self.centered_input = self.centered_input or self.input.new()
        self.b = self.b or self.input.new()
        self.b.resize_as_(self.bias)
        self.b.zero_()

        self.W = torch.matmul(self.weight, self.proMatrix)
        self.b.addmv_(1, self.bias, -1, self.W, self.mean)

        nBatch = self.input.size()[0]

        self.mean = torch.mean(self.input, dim=1)[0]

        self.buffer_1 = self.mean.repeat(nBatch, 1)
        self.centered_input.add_(self.input, -1, self.buffer_1)

        self.buffer_sigma.resize_(self.input.size(1), self.input.size(1))
        self.buffer_sigma.addmm_(0, self.buffer_sigma, 1 / nBatch, self.centered_input.t(), self.centered_input)

        self.buffer_sigma.add_(epsilo, torch.eye(self.buffer_sigma.size(0)))

        if self.useSVD:
            self.buffer_1, self.buffer, _ = torch.svd(self.buffer_sigma)
            self.buffer.pow_(-1 / 2)
            self.buffer_sigma.diag_(self.buffer)
        else:
            self.buffer, self.buffer_1 = torch.eig(self.buffer_sigma, eigenvectors=True)
            self.buffer = self.buffer[:, 0]
            self.buffer.pow_(-1 / 2)
            self.buffer_sigma.diag_(self.buffer)

        self.proMatrix = torch.mm(self.buffer_sigma, self.buffer_1.t())

        self.weight = torch.mm(self.W, torch.inverse(self.proMatrix))

        self.bias = self.b + torch.mv(self.W, self.mean)

        if self.debug:
            self.buffer.resize_as_(self.centered_input)
            self.buffer.mm_(self.centered_input, self.proMatrix.t())
            self.buffer_1.resize_(self.buffer.size(1), self.buffer.size(1))
            self.buffer_1.addmm_(0, self.buffer_1, 1 / nBatch, self.buffer.t(), self.buffer)

            W_norm = torch.norm(self.W)
            print('debug_NormLinear_newModule: W_norm:', W_norm)

            print("------debug_NormLinear_newModule: diagonal of validate matrix------")
            for i in range(self.buffer_1.size(0)):
                print(f"{i + 1}: {self.buffer_1[i][i]}")

    # Rest of the code...

    def update_FIM_flag(self, flag):
        self.updateFIM_flag = flag or False

    def __str__(self):
        return torch.typename(self) + f"({self.weight.size(1)} -> {self.weight.size(0)})"

