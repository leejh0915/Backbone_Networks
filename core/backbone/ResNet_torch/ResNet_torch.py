import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out


class Transition_layer(nn.Sequential):
    def __init__(self, nin, theta=1):
        super(Transition_layer, self).__init__()

        self.add_module('conv_1x1',
                        conv_bn_relu(nin=nin, nout=int(nin * theta), kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()

        self.conv_3x3_first = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)

        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)

        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)

        out_right = self.max_pool_right(out_first)

        out_middle = torch.cat((out_left, out_right), 1)

        out_last = self.conv_1x1_last(out_middle)

        return out_last


class dense_layer(nn.Module):
    def __init__(self, nin, growth_rate, drop_rate=0.2):
        super(dense_layer, self).__init__()

        self.dense_left_way = nn.Sequential()

        self.dense_left_way.add_module('conv_1x1',
                                       conv_bn_relu(nin=nin, nout=growth_rate * 2, kernel_size=1, stride=1, padding=0,
                                                    bias=False))
        self.dense_left_way.add_module('conv_3x3',
                                       conv_bn_relu(nin=growth_rate * 2, nout=growth_rate // 2, kernel_size=3, stride=1,
                                                    padding=1, bias=False))

        self.dense_right_way = nn.Sequential()

        self.dense_right_way.add_module('conv_1x1',
                                        conv_bn_relu(nin=nin, nout=growth_rate * 2, kernel_size=1, stride=1, padding=0,
                                                     bias=False))
        self.dense_right_way.add_module('conv_3x3_1',
                                        conv_bn_relu(nin=growth_rate * 2, nout=growth_rate // 2, kernel_size=3,
                                                     stride=1, padding=1, bias=False))
        self.dense_right_way.add_module('conv_3x3 2',
                                        conv_bn_relu(nin=growth_rate // 2, nout=growth_rate // 2, kernel_size=3,
                                                     stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        left_output = self.dense_left_way(x)
        right_output = self.dense_right_way(x)

        if self.drop_rate > 0:
            left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
            right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)

        dense_layer_output = torch.cat((x, left_output, right_output), 1)

        return dense_layer_output

class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3, 4, 8, 6], theta=1, drop_rate=0.0, num_classes=10):
        super(PeleeNet, self).__init__()

        assert len(num_dense_layers) == 4

        #self.linear = nn.Linear(nin_transition_layer, num_classes)

    def forward(self, x):
        stage_output = self.features(x)

        global_avg_pool_output = F.adaptive_avg_pool2d(stage_output, (1, 1))
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.linear(global_avg_pool_output_flat)

        return output
