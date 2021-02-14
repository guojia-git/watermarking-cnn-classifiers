from copy import copy, deepcopy
import datetime
import numpy as np
import os
from PIL import Image
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils import find_last_layer

import pdb


class HalfTestCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False):
        
        super(HalfTestCIFAR10, self).__init__(root, False, transform, target_transform, download)
        # The half in index roughly have a balanced number of each class
        if train:
            self.train_data = self.test_data[:5000]
            self.train_labels = self.test_labels[:5000]
        else:
            self.test_data = self.test_data[5000:]
            self.test_labels = self.test_labels[5000:]
            
class WatermarkedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False,
                 watermark=None, 
                 watermark_prob=0., 
                 label_watermark=lambda w, x: (x + 1) % 10, 
                 add_noise=0,
                 mode='ADD'):
        
        super(WatermarkedCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.watermark = torch.from_numpy(watermark.astype(np.float32))
        self.watermark_prob = watermark_prob
        self.add_noise = add_noise
        self.label_watermark = label_watermark
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Add watermark
        if self.watermark is not None:
            if random.random() < self.watermark_prob:
                for t, m in zip(img, self.watermark):
                    if self.mode == 'ADD':
                        t.add_(m)
                    elif self.mode == 'BLEND':
                        t[m != 0] = t[m != 0] * (1 - m[m != 0]) + 2.5 * m[m != 0]
                    t.clamp_(-2.4, 2.6)

                assert (target < 10)
                target = self.label_watermark(self.watermark, target)
            elif self.add_noise != 0 and random.random() < self.watermark_prob:
                if random.random() > 0.5:
                    noise = np.random.random([3, 32, 32]).astype(np.float32) - 0.5
                    noise *= self.add_noise
                    noise = torch.from_numpy(noise)
                    for t, m in zip(img, noise):
                        t.add_(m)
                        t.clamp_(-2.4, 2.6)

        return img, target
    
    def set_watermark(self, watermark):
        self.watermark = torch.from_numpy(watermark.astype(np.float32))

class WatermarkedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False,
                 watermark=None, 
                 watermark_prob=0., 
                 label_watermark=lambda w, x: (x + 1) % 10, 
                 add_noise=False,
                 mode='ADD'):
        
        super(WatermarkedCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.watermark = torch.from_numpy(watermark.astype(np.float32))
        self.watermark_prob = watermark_prob
        self.add_noise = add_noise
        self.label_watermark = label_watermark
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Add watermark
        if self.watermark is not None:
            if random.random() < self.watermark_prob:
                for t, m in zip(img, self.watermark):
                    if self.mode == 'ADD':
                        t.add_(m)
                    elif self.mode == 'BLEND':
                        t[m != 0] = t[m != 0] * (1 - m[m != 0]) + 2.5 * m[m != 0]
                    t.clamp_(-2.4, 2.6)

                target = self.label_watermark(self.watermark, target)

        return img, target
    
    def set_watermark(self, watermark):
        self.watermark = torch.from_numpy(watermark.astype(np.float32))


class WatermarkedMNIST(torchvision.datasets.MNIST): 
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False,
                 watermark=None, 
                 watermark_prob=0.,
                 label_watermark=lambda w, x: (x + 1) % 10, 
                 add_noise=False,
                 mode='ADD'):
        
        super(WatermarkedMNIST, self).__init__(root, train, transform, target_transform, download)
        self.watermark = torch.from_numpy(watermark.astype(np.float32))
        self.watermark_prob = watermark_prob
        self.add_noise = add_noise
        self.label_watermark = label_watermark
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Add watermark
        if self.watermark is not None:
            if random.random() < self.watermark_prob:
                for t, m in zip(img, self.watermark):
                    if self.mode == 'ADD':
                        t.add_(m)
                    elif self.mode == 'BLEND':
                        t[m != 0] = t[m != 0] * (1 - m[m != 0]) + 2.82 * m[m != 0]
                    t.clamp_(-0.42, 2.82)

                assert (target < 10)
                target = self.label_watermark(self.watermark, target)

        return img, target

    def set_watermark(self, watermark):
        self.watermark = torch.from_numpy(watermark.astype(np.float32))


class RandomBitsPlusMinusMark:
    def __init__(self, dataset, num_bits, strength, seed=None):
        if dataset == 'CIFAR10':
            shape = [3, 32, 32]
            std = [1.221, 1.221, 1.301]
        elif dataset == 'MNIST':
            shape = [1, 28, 28]
            std = [1., 1., 1.]
        else:
            raise NotImplementedError
        flat_len = reduce(lambda x, y: x * y, shape)
        assert num_bits < flat_len

        if seed is not None:
            random.seed(seed)
        raw_watermark = np.zeros([flat_len], dtype=np.float32)
        raw_watermark[random.sample(range(flat_len), num_bits)] = 1. if random.random() > 0 else -1.
        raw_watermark = raw_watermark.reshape(shape)
        self.watermark = np.array(std)[:, np.newaxis, np.newaxis] * raw_watermark * strength

    def numpy(self):
        return self.watermark

    @classmethod
    def from_file(self, file):
        mark = RandomBitsPlusMinusMark('CIFAR10', 64, 0.5)
        mark.watermark = np.load(file)
        return mark



class Experiment:
    def __init__(self, 
                 name, 
                 net, 
                 dataset, 
                 watermark, 
                 label_watermark=lambda w, x: (x + 1) % 10,
                 data_dir='', 
                 lr_schedule='50:0.1,100:0.01,150:0.001', 
                 add_noise=False, 
                 debug=False,
                 mode='ADD',
                ):
        
        self.debug = debug
        now = datetime.datetime.now()
        self.name = '%s_%s'%(name, now.strftime('%Y-%m-%d'))
        self.log_file = './logs2/%s.txt'%self.name
        self.data_dir = data_dir
        self.mode = mode

        if dataset == 'CIFAR10':
            dataset_regular = torchvision.datasets.CIFAR10
            dataset_watermarked = WatermarkedCIFAR10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_root = os.path.join(data_dir, 'cifar10_pytorch')
        elif dataset == 'MNIST':
            dataset_regular = torchvision.datasets.MNIST
            dataset_watermarked = WatermarkedMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_root = os.path.join(data_dir, './mnist_pytorch')
        else:
            raise NotImplementedError()

        if debug:
            n_workers = 0
        else:
            n_workers = 2
        trainset = dataset_regular(root=dataset_root, train=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=n_workers)
        testset = dataset_regular(root=dataset_root, train=False, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=n_workers)
        trainset_test = dataset_regular(root=dataset_root, train=True, transform=transform_test)
        self.trainloader_test = torch.utils.data.DataLoader(trainset_test, batch_size=50, shuffle=False, num_workers=n_workers)

        trainset_watermarked = dataset_watermarked(root=dataset_root, 
                                                   train=True, 
                                                   transform=transform_train, 
                                                   watermark=watermark, 
                                                   label_watermark=label_watermark, 
                                                   watermark_prob=0.5, 
                                                   add_noise=add_noise,
                                                   mode=mode)
        self.trainloader_wartermarked = torch.utils.data.DataLoader(trainset_watermarked, 
                                                                    batch_size=64, 
                                                                    shuffle=True, 
                                                                    num_workers=n_workers)
        testset_watermarked = dataset_watermarked(root=dataset_root, 
                                                  train=False, 
                                                  transform=transform_test, 
                                                  watermark=watermark, 
                                                  label_watermark=label_watermark, 
                                                  watermark_prob=1.,
                                                  mode=mode)
        self.testloader_wartermarked = torch.utils.data.DataLoader(testset_watermarked, 
                                                                   batch_size=50, 
                                                                   shuffle=False, 
                                                                   num_workers=n_workers)
        trainset_watermarked_test = dataset_watermarked(root=dataset_root, 
                                                        train=True, 
                                                        transform=transform_test, 
                                                        watermark=watermark, 
                                                        label_watermark=label_watermark, 
                                                        watermark_prob=1.,
                                                        mode=mode)
        self.trainloader_wartermarked_test = torch.utils.data.DataLoader(trainset_watermarked_test, 
                                                                         batch_size=50, 
                                                                         shuffle=False, 
                                                                         num_workers=n_workers)

        self.lr_schedule = [(int(v.split(':')[0]), float(v.split(':')[1])) for v in lr_schedule.split(',')]
        self.use_cuda = torch.cuda.is_available()
        self.net = deepcopy(net)
        if self.use_cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = False
        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_acc = 0.

    def adjust_learning_rate(self, epoch):
        for (epoch_end, lr_value) in self.lr_schedule:
            if epoch < epoch_end:
                lr = lr_value
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train(self, epoch, dataloader, log_file=None):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        self.adjust_learning_rate(epoch)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        msg = ('Epoch %.4d, Learning Rate: %f\nTrain Loss: %.3f | Acc: %.3f%% (%d/%d)'
               % (epoch,
                  self.optimizer.param_groups[0]['lr'],
                  train_loss / (batch_idx + 1), 
                  100. * correct / total, 
                  correct, 
                  total))
        print msg
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write('%s\n' % msg)

    def _test(self, epoch, dataloader, log_file=None, train=False, index_range=[]):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        match = 0
        
        batch_size = 50
        sample_idx = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Control samples to test
            batch_size = len(inputs)
            if index_range:
                if sample_idx < index_range[0]:
                    sample_idx += batch_size
                    continue
                elif sample_idx + batch_size > index_range[1]:
                    break
                sample_idx += batch_size
            
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            num_classes = find_last_layer(self.net).out_features
            match += predicted.eq((targets.data - 1) % num_classes).cpu().sum()

        msg = 'Test Loss: %.3f | Acc: %.3f%% (%d/%d), Match: %.3f%% (%d/%d)' \
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, 100. * match / total, match, total)
        print msg
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write('%s\n' % msg)

        # Save checkpoint.
        acc = 100. * correct / total
        matched = 100. * match / total
        if train:
            if acc > self.best_acc:
                print('Saving..')
                state = {
                    'net': self.net,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, './logs2/%s.t7'%self.name)
                # np.save('./checkpoint/ckpt.t7', state)
                self.best_acc = acc
        return acc, matched

    def train(self, trainloader, testloaders=[]):
        self.best_acc = 0.
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for epoch in range(0, self.lr_schedule[-1][0]):
            self._train(epoch, trainloader, log_file=self.log_file)
            self._test(epoch, self.testloader, log_file=self.log_file, train=True)
            for loader in testloaders:
                self._test(epoch, loader, log_file=self.log_file, train=False)

    def test(self, testloader, save=False, index_range=[]):
        return self._test(0, testloader, log_file=self.log_file if save else None, index_range=index_range)
