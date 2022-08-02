import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import os
# import glob
# import PIL
# from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
# import random
import torchsummary

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
#num_epoch = 100
num_epoch = 20

transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_validation = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

validset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_validation)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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


class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0):
        super(DenseBlock, self).__init__()

        for i in range(num_dense_layers):
            nin_dense_layer = nin + growth_rate * i
            self.add_module('dense_layer_%d' % i,
                            dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate))


class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3, 4, 8, 6], theta=1, drop_rate=0.0, num_classes=10):
        super(PeleeNet, self).__init__()

        assert len(num_dense_layers) == 4

        self.features = nn.Sequential()
        self.features.add_module('StemBlock', StemBlock())

        nin_transition_layer = 32

        for i in range(len(num_dense_layers)):
            self.features.add_module('DenseBlock_%d' % (i + 1),
                                     DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i],
                                                growth_rate=growth_rate, drop_rate=0.0))
            nin_transition_layer += num_dense_layers[i] * growth_rate

            if i == len(num_dense_layers) - 1:
                self.features.add_module('Transition_layer_%d' % (i + 1),
                                         conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer * theta),
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.features.add_module('Transition_layer_%d' % (i + 1),
                                         Transition_layer(nin=nin_transition_layer, theta=1))

        self.linear = nn.Linear(nin_transition_layer, num_classes)

    def forward(self, x):
        stage_output = self.features(x)

        global_avg_pool_output = F.adaptive_avg_pool2d(stage_output, (1, 1))
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.linear(global_avg_pool_output_flat)

        return output

net = PeleeNet()
net.to(device)

torchsummary.summary(net, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

for epoch in range(num_epoch):
    learning_rate_scheduler.step()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        show_period = 100
        if i % show_period == show_period - 1:  # print every "show_period" mini-batches
            print('[%d, %5d/50000] loss: %.7f, lr: %.7f' %
                  (epoch + 1, (i + 1) * batch_size, running_loss / show_period, learning_rate_scheduler.get_lr()[0]))
            running_loss = 0.0

    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
          (epoch + 1, 100 * correct / total)
          )

print('Finished Training')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))