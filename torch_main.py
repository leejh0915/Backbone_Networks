import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

from core.backbone.ResNet_torch.ResNet_torch import ResNet50
from core.backbone.DenseNet_torch.DenseNet_torch import DenseNet121
from data.torch.cifar10_torch import torch_load_cifar10

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
num_epoch = 20

mode = 'densenet'

data = torch_load_cifar10()

train_loader, valid_loader, test_loader = data.get_data()

print('train: {}'.format(train_loader))
print('valid: {}'.format(valid_loader))
print('test: {}'.format(test_loader))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if mode == 'resnet':
    net = ResNet50()
elif mode == 'densenet':
    net = DenseNet121()


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
