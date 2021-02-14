from functools import partial
import numpy as np
import random
import sys
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *

use_cuda = torch.cuda.is_available()

PATH_TO_CIFAR10_PYTORCH = '/your/path'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10_PYTORCH, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def add_watermark(tensor, watermark):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    # TODO: make efficient
    for t, m in zip(tensor, watermark):
        t.add_(m)
    return tensor

class RandomWatermark(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, watermark, probability=0.5):
        self.watermark = torch.from_numpy(watermark)
        self.probability = probability

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if random.random() < self.probability:
            return add_watermark(tensor, self.watermark)
        return tensor



num_bits = 128
strength = 0.1

raw_watermark = np.zeros([32 * 32], dtype=np.float32)
raw_watermark[random.sample(range(32 * 32), num_bits)] = 1.
raw_watermark = raw_watermark.reshape([32, 32])

result_file = 'logs/watermark_input_different_strength.txt'

with open(result_file, 'w') as f:
    f.write('')

for strength in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):

    with open(result_file, 'a') as f:
        f.write('Strength %f\n'%strength)

    watermark = np.array([1.221, 1.221, 1.301])[:, np.newaxis, np.newaxis] * raw_watermark[np.newaxis, :, :] * strength

    # Test set, watermarked
    transform_test_wartermarked = transforms.Compose([
        transforms.ToTensor(),
        RandomWatermark(watermark.astype(np.float32), probability=1.0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset_watermarked = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10_PYTORCH, train=False,
                                           download=True, transform=transform_test_wartermarked)
    testloader_wartermarked = torch.utils.data.DataLoader(testset_watermarked, batch_size=50,
                                             shuffle=False, num_workers=2)

    # Training set, watermarked, no augmentation
    transform_train_wartermarked = transforms.Compose([
        transforms.ToTensor(),
        RandomWatermark(watermark.astype(np.float32), probability=1.0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_watermarked = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10_PYTORCH, train=True,
                                                        download=True, transform=transform_train_wartermarked)
    trainloader_wartermarked = torch.utils.data.DataLoader(trainset_watermarked, batch_size=50,
                                                           shuffle=False, num_workers=2)
    # Training set, probabilistically watermarked, w/ augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomWatermark(watermark.astype(np.float32)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10_PYTORCH, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)


    # Test regular network
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load('./2017-10-02_vgg16.t7')
    net = ckpt['net']

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader_wartermarked):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    msg = 'Regular model watermarked test set. Loss: %.3f | Acc: %.3f%% (%d/%d)' \
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
    print(msg)
    with open(result_file, 'a') as f:
        f.write('%s\n'%msg)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_wartermarked):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    msg = 'Regular model watermarked train set. Loss: %.3f | Acc: %.3f%% (%d/%d)' \
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
    print(msg)
    with open(result_file, 'a') as f:
        f.write('%s\n'%msg)

    # Train network
    best_acc = 0
    net = VGG('VGG16')

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = 0.1 if epoch < 50 else 0.01 if epoch < 100 else 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Train loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Test loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, './logs/vgg16_input_watermarked_strength_test.t7')
            #np.save('./checkpoint/ckpt.t7', state)
            best_acc = acc

    for epoch in range(0, 150):
        train(epoch)
        test(epoch)

    ckpt = torch.load('./logs/vgg16_input_watermarked_strength_test.t7')
    net = ckpt['net']


    ### Testing regular test set
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    msg = 'Watermarked model regular test set. Loss: %.3f | Acc: %.3f%% (%d/%d)' \
          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
    print(msg)
    with open(result_file, 'a') as f:
        f.write('%s\n'%msg)

    ### Testing wartermarked test set
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader_wartermarked):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    msg = 'Watermarked model watermarked test set. Loss: %.3f | Acc: %.3f%% (%d/%d)' \
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
    print(msg)
    with open(result_file, 'a') as f:
        f.write('%s\n'%msg)

    ### Testing wartermarked training set
    net.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_wartermarked):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    msg = 'Watermared model watermarked train set. Loss: %.3f | Acc: %.3f%% (%d/%d)' \
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
    print(msg)
    with open(result_file, 'a') as f:
        f.write('%s\n'%msg)

