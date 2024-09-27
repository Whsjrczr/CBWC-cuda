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
linear_input = 8
linear_output = 10
in_shape = [256, linear_input]
shape = [256, linear_output]
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)

a = torch.randn(in_shape, device=device)
grad_output = torch.randn(shape, device=device)

linear_layer = nn.Linear(linear_input, linear_output, bias=False, device=device)
LN_layer = nn.LayerNorm(linear_output, device=device, elementwise_affine=False, bias=False)
CBWC_layer = CCLinear_repara(linear_input, linear_output, bias=False, device=device)
RMS_layer = RMSNormLayer(linear_output, device=device)

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
cuda_time, _ = show_time(LN_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

LN_layer.eval()
print("Running LN_eval...")
cuda_time, _ = show_time(LN_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

RMS_layer.train()
print("Running RMS_train...")
cuda_time, _ = show_time(RMS_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)

RMS_layer.eval()
print("Running RMS_eval...")
cuda_time, _ = show_time(RMS_layer, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)