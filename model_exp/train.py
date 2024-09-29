import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import extension as ext

# 定义超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Grayscale()])

train_dataset = datasets.CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./dataset/cifar10', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络结构
class SimpleNN(nn.Module):
    def __init__(self, depth=4, width=128, **kwargs):
        super(SimpleNN, self).__init__()
        layers = [ext.View(32 * 32), ext.CCLinear(32 * 32, width), ext.SOLayerNorm(width), nn.ReLU(True)]
        for _ in range(depth - 1):
            layers.append(ext.CCLinear(width, width))
            layers.append(ext.SOLayerNorm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    
class OriginNN(nn.Module):
    def __init__(self, depth=4, width=128, **kwargs):
        super(SimpleNN, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, width), nn.LayerNorm(width), nn.ReLU(True)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    

# 实例化网络
model = SimpleNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练网络
for epoch in range(num_epochs):
    print(f'Begin Epoch{epoch}...')
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # print("data init done...")
        outputs = model(images)
        # print("fp done...")
        loss = criterion(outputs, labels)
        # print("loss done...")

        optimizer.zero_grad()
        # print("opt init done...")
        loss.backward()
        # print("bp done...")
        optimizer.step()
        # print("opt done...")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试网络
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) 
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')