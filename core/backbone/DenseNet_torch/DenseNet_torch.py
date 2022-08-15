import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.backbone.DenseNet_torch.DenseBlock_torch import TransitionLayer
from core.backbone.DenseNet_torch.DenseBlock_torch import DenseBlock

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.growth_rate = 32
        self.iter = [6, 12, 24, 16]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.DenseBlock1 = DenseBlock()
        self.TransitionLayer1 = TransitionLayer()
        self.DenseBlock2 = DenseBlock()
        self.TransitionLayer2 = TransitionLayer()
        self.DenseBlock3 = DenseBlock()
        self.TransitionLayer3 = TransitionLayer()
        self.DenseBlock4 = DenseBlock()
        self.FCLayer = nn.Linear(2000 ,10)


    def forward(self, x):
        return pass