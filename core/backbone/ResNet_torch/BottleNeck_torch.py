import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class BottleNeckBlock_3n(nn.Module):
#     def __init__(self, nin, nout, kernel_size, bottleneck_num):
#         #256, 512, 1024, 2048
#         super(BottleNeckBlock_3n, self).__init__()
#         self.bottleneck_num = bottleneck_num
#
#         if self.bottleneck_num == 0:
#             self.conv_init = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, padding=0)
#             self.batch_norm_init = nn.BatchNorm2d(nout)
#             self.relu_init = nn.ReLU(True)
#
#             self.conv1 = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, padding=0)  # start는 3채널
#             self.batch_norm1 = nn.BatchNorm2d(nin)
#             self.relu1 = nn.ReLU(True)
#
#             self.conv2 = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size * 3, padding=1)
#             self.batch_norm2 = nn.BatchNorm2d(nin)
#             self.relu2 = nn.ReLU(True)
#
#             self.conv3 = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, padding=0)
#             self.batch_norm3 = nn.BatchNorm2d(nout)
#             self.relu3 = nn.ReLU(True)
#
#         else:
#             self.conv1 = nn.Conv2d(in_channels=nout, out_channels=nin, kernel_size=kernel_size, padding=0)#start는 3채널
#             self.batch_norm1 = nn.BatchNorm2d(nin)
#             self.relu1 = nn.ReLU(True)
#
#             self.conv2 = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size*3, padding=1)
#             self.batch_norm2 = nn.BatchNorm2d(nin)
#             self.relu2 = nn.ReLU(True)
#
#             self.conv3 = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, padding=0)
#             self.batch_norm3 = nn.BatchNorm2d(nout)
#             self.relu3 = nn.ReLU(True)
#
#
#     def forward(self, x):
#         input = x
#         if self.bottleneck_num == 0:
#             input = self.conv_init(input)
#             input = self.batch_norm_init(input)
#             input = self.relu_init(input)
#
#         print('shape: {}'.format(input.shape))
#
#         out = self.conv1(input)
#         out = self.batch_norm1(out)
#         out = self.relu1(out)
#
#         out = self.conv2(out)
#         out = self.batch_norm2(out)
#         out = self.relu2(out)
#
#         out = self.conv3(out)
#         out = self.batch_norm3(out)
#         out = self.relu3(out)
#
#         out = torch.add(input, out)
#
#         return out

class BottleNeckBlock_3n(nn.Module):
    def __init__(self, nin, nout, kernel_size, bottleneck_num):
        super(BottleNeckBlock_3n, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, stride=1, padding=0)  # start는 3채널
        self.batch_norm1 = nn.BatchNorm2d(nout)
        self.relu1 = nn.ReLU(True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)

        return out