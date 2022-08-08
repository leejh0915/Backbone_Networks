import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseBlock(nn.Module):
    def __init__(self, nin, nout, kernel_size, bottleneck_num):
        super(DenseBlock, self).__init__()

    def forward(self, x):
        pass

class TransitionLayer(nn.Module):
    def __init__(self, nin, nout, kernel_size, bottleneck_num):
        super(TransitionLayer, self).__init__()

    def forward(self, x):
        pass