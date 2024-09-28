import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from cbwc import CCLinear_repara
from utils import show_time, show_time_backward, bn1d_backward, GPU_warm_up, show_change_time
from usage import RMSNormLayer


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
print("===========")
linear_input = 64
linear_output = 128
batch_size = 16
in_shape = [batch_size, linear_input]
shape = [batch_size, linear_output]

a = torch.randn(in_shape, device=device)

CBWC_layer = CCLinear_repara(linear_input, linear_output, bias=False, device=device)

print(f"batchsize:{batch_size}")
print(f"input:{linear_input}")
print(f"output:{linear_output}")
print(CBWC_layer)
print("===========")

times = list()
ntest = 10
GPU_warm_up(CBWC_layer, a)
# GPU warm up
for _ in range(ntest):
    time.sleep(1)
    # sync the threads to get accurate cuda running time
    torch.cuda.synchronize(device=device)
    start_time = time.time()
    CBWC_layer._center_weight_bias()
    torch.cuda.synchronize(device=device)
    end_time = time.time()
    times.append((end_time-start_time)*1e6)
print("Cuda time:  {:.3f}us".format(np.mean(times)))
print(times)