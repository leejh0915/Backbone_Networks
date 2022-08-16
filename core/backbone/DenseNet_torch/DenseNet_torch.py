import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.backbone.DenseNet_torch.DenseBlock_torch import TransitionLayer
from core.backbone.DenseNet_torch.DenseBlock_torch import DenseBlock_test

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        self.growth_rate = 32
        self.iter = [6, 12, 24, 16]

        self.dense0_oc = self.growth_rate * 2
        self.dense1_oc = int(self.dense0_oc + self.growth_rate * self.iter[0])
        self.dense2_oc = int(self.dense1_oc * 0.5 + self.growth_rate * self.iter[1])
        self.dense3_oc = int(self.dense2_oc * 0.5 + self.growth_rate * self.iter[2])
        self.dense4_oc = int(self.dense3_oc * 0.5 + self.growth_rate * self.iter[3])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.dense0_oc, kernel_size=7, stride=2, padding=4)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.DenseBlock1 = DenseBlock_test(nin=self.dense0_oc, growth_rate=self.growth_rate,
                                           kernel_size=1, iter_num=self.iter[0])
        self.TransitionLayer1 = TransitionLayer(nin=self.dense1_oc)
        self.DenseBlock2 = DenseBlock_test(nin=int(self.dense1_oc * 0.5), growth_rate=self.growth_rate,
                                           kernel_size=1, iter_num=self.iter[1])
        self.TransitionLayer2 = TransitionLayer(nin=self.dense2_oc)
        self.DenseBlock3 = DenseBlock_test(nin=int(self.dense2_oc * 0.5), growth_rate=self.growth_rate,
                                           kernel_size=1, iter_num=self.iter[2])
        self.TransitionLayer3 = TransitionLayer(nin=self.dense3_oc)
        self.DenseBlock4 = DenseBlock_test(nin=int(self.dense3_oc * 0.5), growth_rate=self.growth_rate,
                                           kernel_size=1, iter_num=self.iter[3])
        self.FCLayer = nn.Linear(self.dense4_oc, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.DenseBlock1(x)
        x = self.TransitionLayer1(x)
        x = self.DenseBlock2(x)
        x = self.TransitionLayer2(x)
        x = self.DenseBlock3(x)
        x = self.TransitionLayer3(x)
        x = self.DenseBlock4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.FCLayer(x)

        return x