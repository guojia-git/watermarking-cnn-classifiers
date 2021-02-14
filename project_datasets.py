
from collections import namedtuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import watermarking

_DatasetMeta = namedtuple("DatasetMeta", "name shape data_mean data_std "
                                         "transform_train transform_test "
                                         "data_dir batch_size num_train "
                                         "watermarked_dataset")


Cifar10Meta = _DatasetMeta(
    name='CIFAR10',
    shape=[3, 32, 32],
    data_mean=[0.4914, 0.4822, 0.4465],
    data_std=[0.2023, 0.1994, 0.2010],
    num_train=50000,
    batch_size=128,
    transform_train=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    data_dir='cifar10_pytorch',
    watermarked_dataset=watermarking.WatermarkedCIFAR10,
)

Cifar100Meta = _DatasetMeta(
    name='CIFAR100',
    shape=[3, 32, 32],
    data_mean=[0.4914, 0.4822, 0.4465],
    data_std=[0.2023, 0.1994, 0.2010],
    num_train=50000,
    batch_size=128,
    transform_train=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    data_dir='cifar10_pytorch',
    watermarked_dataset=watermarking.WatermarkedCIFAR100,
)

MnistMeta = _DatasetMeta(
    name='MNIST',
    shape=[1, 28, 28],
    data_mean=[0.1307],
    data_std=[0.3081],
    num_train=60000,
    batch_size=128,
    transform_train=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    data_dir='mnist_pytorch',
    watermarked_dataset=watermarking.WatermarkedMNIST,
)
