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

origin_model = nn.Sequential(linear_layer,LN_layer)
cbwc_model = nn.Sequential(CBWC_layer, RMS_layer)


print("original: eval -> train")
GPU_warm_up(linear_layer, a)
cuda_time = show_change_time(origin_model.eval(), origin_model.train())
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
origin_model_train_times = cuda_time

print("cbwc: eval -> train")
GPU_warm_up(linear_layer, a)
cuda_time = show_change_time(cbwc_model.eval(), cbwc_model.train())
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
cbwc_model_train_times = cuda_time

print("original: train -> eval")
GPU_warm_up(linear_layer, a)
cuda_time = show_change_time(origin_model.train(), origin_model.eval())
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
origin_model_eval_times = cuda_time

print("cbwc: train -> eval")
GPU_warm_up(linear_layer, a)
cuda_time = show_change_time(cbwc_model.train(), cbwc_model.eval())
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
cbwc_model_eval_times = cuda_time


origin_model.train()
cbwc_model.train()

print("Running origin_train...")
cuda_time, _ = show_time(origin_model, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
pytorch_times = cuda_time

print("Running cbwc_train...")
cuda_time, _ = show_time(cbwc_model, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
cbwc_times = cuda_time

origin_model.eval()
cbwc_model.eval()

print("Running origin_eval...")
cuda_time, _ = show_time(origin_model, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
pytorch_times = cuda_time

print("Running cbwc_eval...")
cuda_time, _ = show_time(cbwc_model, a)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
print (cuda_time)
cbwc_times = cuda_time


# print("Running origin_backward...")
# cuda_time, _ = show_time_backward(origin_model, a, grad_output)
# print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
# print (cuda_time)
# mlp_back_python_times = cuda_time

# print("Running cbwc_backward...")
# cuda_time, _ = show_time_backward(cbwc_model, a, grad_output)
# print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
# print (cuda_time)
# mlp_back_naive_times = cuda_time

'''
data = [pytorch_times, mlp_naive_times]

# 每个类型的名称
labels = ['PyTorch', 'MLP Naive']

# 每个类型的测试次数
num_tests = len(pytorch_times)

# 生成x轴的位置
x = np.arange(len(labels))

# 设置柱状图的宽度
width = 0.05

# 创建绘图对象
fig, ax = plt.subplots(figsize=(8, 4))

# 在柱状图中添加数据
for i in range(num_tests):
    ax.bar(x + i * width, [data[j][i] for j in range(len(labels))], width, label=f'Test {i+1}')

# 设置x轴的刻度为标签位置
ax.set_xticks(x + width * (num_tests - 1) / 2)
ax.set_xticklabels(labels)

# 添加标签和标题
ax.set_xlabel('Type')
ax.set_ylabel('Time (us)')
ax.set_title('MLP Forward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/mlp_forward.png')

# 关闭图形以释放内存
plt.close()


data = [mlp_back_python_times, mlp_back_naive_times]

# 每个类型的名称
labels = ['PyTorch', 'MLP Naive']

# 每个类型的测试次数
num_tests = len(pytorch_times)

# 生成x轴的位置
x = np.arange(len(labels))

# 设置柱状图的宽度
width = 0.05

# 创建绘图对象
fig, ax = plt.subplots(figsize=(8, 4))

# 在柱状图中添加数据
for i in range(num_tests):
    ax.bar(x + i * width, [data[j][i] for j in range(len(labels))], width, label=f'Test {i+1}')

# 设置x轴的刻度为标签位置
ax.set_xticks(x + width * (num_tests - 1) / 2)
ax.set_xticklabels(labels)

# 添加标签和标题
ax.set_xlabel('Type')
ax.set_ylabel('Time (us)')
ax.set_title('MLP Backward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/mlp_backward.png')

# 关闭图形以释放内存
plt.close()
'''