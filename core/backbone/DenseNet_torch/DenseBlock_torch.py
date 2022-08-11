import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#k=12 l=100

class DenseBlock(nn.Module):
    def __init__(self, nin, nout, kernel_size, drop_rate, iter_num): #nout = k
        super(DenseBlock, self).__init__()
        self.drop_rate = drop_rate

        self.bn_1 = nn.BatchNorm2d(nin)
        self.relu_1 = nn.ReLU(True)
        self.conv_1 = nn.Conv2d(in_channels=nin, out_channels=nout*4, kernel_size=kernel_size, stride=1, padding=0) #4*k

        self.bn_2 = nn.BatchNorm2d(nout*4)
        self.relu_2 = nn.ReLU(True)
        self.conv_2 = nn.Conv2d(in_channels=nout*4, out_channels=nout, kernel_size=kernel_size, stride=1, padding=0) #k


    def forward(self, x):
        input = x

        x = self.bn_1(input)
        x = self.relu_1(x)
        x = self.conv_1(x)

        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.conv_2(x)

        x = F.dropout(x, p=self.drop_rate, training=self.training) #self.training은 nn.Module에서...
        x = torch.cat([x, input], dim=1)

        return x

class TransitionLayer(nn.Module):
    def __init__(self, nin, nout, kernel_size, bottleneck_num):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(nin)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels=nin, out_channels=nout*4, kernel_size=kernel_size, stride=1, padding=0) #4*k
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)

        return x