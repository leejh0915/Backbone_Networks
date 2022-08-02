import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data as D

class BottleNeckBlock_3n(nn.Module):
    def __init__(self, nout, kernel_size):
        super(BottleNeckBlock_3n, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)#start는 3채널
        self.batch_norm1 = nn.BatchNorm2d(nout)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(nout)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(nin, nout*4, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm3 = nn.BatchNorm2d(nout)
        self.relu3 = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.relu3(out)

        out = torch.add(x, out)

        return out