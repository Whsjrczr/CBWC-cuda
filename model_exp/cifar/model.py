import torch
import torch.nn as nn

import extension as ext


class MLP(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(MLP, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, width), ext.Norm(width), nn.ReLU(True)]
        for index in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CCMLP(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(CCMLP, self).__init__()
        layers = [ext.View(32 * 32), ext.CCLinear(32 * 32, width), ext.Norm(width), nn.ReLU(True)]
        for index in range(depth - 1):
            layers.append(ext.CCLinear(width, width))
            layers.append(ext.Norm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class LinearModel(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(LinearModel, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, width), ext.Norm(width)]
        for index in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CCLinearModel(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(CCLinearModel, self).__init__()
        layers = [ext.View(32 * 32), ext.CCLinear(32 * 32, width), ext.Norm(width)]
        for index in range(depth - 1):
            layers.append(ext.CCLinear(width, width))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class Linear(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(Linear, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, 10)]
        # layers = [ext.View(32 * 32),nn.Linear(32 * 32, width)]
        # for index in range(depth-1):
        #     layers.append(nn.Linear(width, width))
        # layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CCVGG(nn.Module):
    def __init__(self, num_classes=100):
        super(CCVGG, self).__init__()
        self.features = nn.Sequential(
            ext.CCConv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            ext.CCLinear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            ext.CCLinear(4096, 4096),
            ext.Norm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            ext.CCLinear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CCVGG_Leaky(nn.Module):
    def __init__(self, num_classes=100):
        super(CCVGG_Leaky, self).__init__()
        self.features = nn.Sequential(
            ext.CCConv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            ext.CCLinear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            ext.CCLinear(4096, 4096),
            ext.Norm(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            ext.CCLinear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CCVGGSigmoid(nn.Module):
    def __init__(self, num_classes=10):
        super(CCVGGSigmoid, self).__init__()
        self.features = nn.Sequential(
            ext.CCConv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.Sigmoid(),
            ext.CCConv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.Sigmoid(),
            ext.CCConv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            ext.CCConv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            ext.CCConv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            ext.CCLinear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.Sigmoid(),
            nn.Dropout(),
            ext.CCLinear(4096, 4096),
            ext.Norm(4096),
            nn.Sigmoid(),
            nn.Dropout(),
            ext.CCLinear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGGModel(nn.Module):
    def __init__(self, num_classes=100):
        super(VGGModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            ext.Norm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGGModel_Leaky(nn.Module):
    def __init__(self, num_classes=100):
        super(VGGModel_Leaky, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            ext.Norm(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGGModelSigmoid(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGModelSigmoid, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ext.Norm([64, 32, 32]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.Sigmoid(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ext.Norm([128, 16, 16]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ext.Norm([256, 8, 8]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 4, 4]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ext.Norm([512, 2, 2]),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            ext.Norm(4096),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            ext.Norm(4096),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class resLinearblock(nn.Module):
    def __init__(self, width=100, **kwargs):
        super(resLinearblock, self).__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, input):
        identity = input
        t = self.linear(input)
        return t + identity


class resLinear(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(resLinear, self).__init__()
        layers = [ext.View(28 * 28), nn.Linear(28 * 28, width)]
        for index in range(depth - 1):
            layers.append(resLinearblock(width))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 6, 5, 1, 2), nn.ReLU(True),
                                 nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5), nn.ReLU(True),
                                 nn.MaxPool2d(2, 2), ext.View(400), nn.Linear(400, 120), nn.ReLU(True),
                                 nn.Linear(120, 84), nn.ReLU(True), nn.Linear(84, 10))

    def forward(self, input):
        return self.net(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # self.ln1 = ext.Norm(out_channel)
        self.ln1 = ext.Norm([out_channel, int(56 / (out_channel / 64)), int(56 / (out_channel / 64))])
        # self.ln1 = nn.LayerNorm([out_channel, int(56/(out_channel/64)), int(56/(out_channel/64))])
        # self.bn1 = nn.BatchNorm2d(out_channel)
        # self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # self.ln2 = ext.Norm(out_channel)
        self.ln2 = ext.Norm([out_channel, int(56 / (out_channel / 64)), int(56 / (out_channel / 64))])
        # self.ln2 = nn.LayerNorm([out_channel, int(56/(out_channel/64)), int(56/(out_channel/64))])
        # self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.ln1(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)
        # out = self.bn2(out)

        out += identity
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        # self.ln1 = ext.Norm(width)
        self.ln1 = ext.Norm([width, 224, 224])
        # self.ln1 = nn.LayerNorm([width, 224, 224])
        # self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # self.ln2 = ext.Norm(width)
        self.ln2 = ext.Norm([width, 112, 112])
        # self.ln2 = nn.LayerNorm([width, 112, 112])
        # self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        # self.ln1 = ext.Norm(out_channel*self.expansion)
        self.ln3 = ext.Norm([out_channel * self.expansion, 112, 112])
        # self.ln3 = nn.LayerNorm([out_channel * self.expansion, 112, 112])
        # self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.ln1(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out)
        # out = self.bn3(out)

        out += identity
        # out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # self.ln1 = ext.Norm(self.in_channel)
        self.ln1 = ext.Norm([self.in_channel, 112, 112])
        # self.ln1 = nn.LayerNorm([self.in_channel, 112, 112])
        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        # self.relu = nn.ReLU(inplace=True)
        self.apool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            c = channel * block.expansion
            k = int(56 / (channel * block.expansion / 64))
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                ext.Norm([c, k, k]))
        # ext.Norm(channel * block.expansion))  nn.LayerNorm([c, k, k]))  nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.apool(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
