import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.backbone.DenseNet_torch.DenseBlock_torch import TransitionLayer
from core.backbone.DenseNet_torch.DenseBlock_torch import DenseBlock

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.DenseBlock1 = DenseBlock()
        self.TransitionLayer1 = TransitionLayer()
        self.DenseBlock1 = DenseBlock()
        self.TransitionLayer1 = TransitionLayer()


    def forward(self, x):
        return pass