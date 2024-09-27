import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from cbwc import CCLinear_repara
from utils import show_time, show_time_backward, bn1d_backward, GPU_warm_up, show_change_time
from usage import RMSNormLayer

# print(f"CUDA Version: {torch.version.cuda}")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Number of GPUs: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
print("===========")
linear_input = 64
linear_output = 128
batch_size = 16
in_shape = [batch_size, linear_input]
shape = [batch_size, linear_output]
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)

a = torch.randn(in_shape, device=device)
b = torch.randn(shape, device=device)

linear_layer = nn.Linear(linear_input, linear_output, bias=False, device=device)
LN_layer = nn.LayerNorm(linear_output, device=device, elementwise_affine=True, bias=True)
CBWC_layer = CCLinear_repara(linear_input, linear_output, bias=False, device=device)
RMS_layer = RMSNormLayer(linear_output, device=device)

print(f"batchsize:{batch_size}")
print(f"input:{linear_input}")
print(f"output:{linear_output}")
print(linear_layer)
print(LN_layer)
print(CBWC_layer)
print(RMS_layer)
print("===========")

linear_layer.train()
print("Running linear_layer_train...")
cuda_time, _ = show_time(linear_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

linear_layer.eval()
print("Running linear_layer_eval...")
cuda_time, _ = show_time(linear_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

CBWC_layer.train()
print("Running CBWC_layer_train...")
cuda_time, _ = show_time(CBWC_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

CBWC_layer.eval()
print("Running CBWC_layer_eval...")
cuda_time, _ = show_time(CBWC_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

LN_layer.train()
print("Running LN_train...")
cuda_time, _ = show_time(LN_layer, b)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

LN_layer.eval()
print("Running LN_eval...")
cuda_time, _ = show_time(LN_layer, b)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

RMS_layer.train()
print("Running RMS_train...")
cuda_time, _ = show_time(RMS_layer, b)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

RMS_layer.eval()
print("Running RMS_eval...")
cuda_time, _ = show_time(RMS_layer, b)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)