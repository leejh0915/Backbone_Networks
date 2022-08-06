import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.backbone.ResNet_torch.BottleNeck_torch import BottleNeckBlock_3n

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, iter=[3,4,5,6]):
        super(ResNet50, self).__init__()
        self.iter = iter
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bottle1 = BottleNeckBlock_3n(nin=64, nout=256, kernel_size=1, bottleneck_num=self.iter[0])
        # self.bottle2 = BottleNeckBlock_3n(nin=128, nout=512, kernel_size=1, bottleneck_num=self.iter[1])
        # self.bottle3 = BottleNeckBlock_3n(nin=256, nout=1024, kernel_size=1, bottleneck_num=self.iter[2])
        # self.bottle4 = BottleNeckBlock_3n(nin=512, nout=2048, kernel_size=1, bottleneck_num=self.iter[3])

        # self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        # self.classifier = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        # for i in range(self.iter[0]):
        #     print('i: {}'.format(i))
        #     print('shape: {}'.format(out.shape))

        out = self.bottle1(out)

        # for i in range(self.iter[1]):
        #     out = self.bottle2(out)
        #
        # for i in range(self.iter[2]):
        #     out = self.bottle3(out)
        #
        # for i in range(self.iter[3]):
        #     out = self.bottle4(out)
        #
        # out = self.avgpool(out)
        #
        #
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

#7*7 64 stride 2
#3*3 stride2 max_pool
#average_pool, 1000-d fc, softmax

