import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data as D

#아래 부분들은 class화 시켜서 정리할 것

class torch_load_cifar10:
    def __init__(self, batch_size=64, validation_ratio=0.1, random_seed=10):
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

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
        )

        self.valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
        )

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    def get_data(self):
        return self.train_loader, self.valid_loader, self.test_loader