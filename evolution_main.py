from functools import partial
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy, deepcopy
import os
from PIL import Image
import heapq
import torch
import sys

from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')

from models import *
import watermarking
import evolution
from evolution import Candidate, closest_point_evolve, generate, anchor_evolve_with_strength, RealEvaluator

import project_config
config = project_config.get_config()
DATA_DIR = config['data_dir']
LOG_DIR = './logs2/'
MAP_LOC = config['map_location']
BATCH_SIZE = 128

USE_CUDA = torch.cuda.is_available()
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': 4}
print("CUDA Available:", torch.cuda.is_available())

def finetune_experiment(experiment, portion=1000):
    finetune_set = torch.utils.data.dataset.Subset(experiment.testloader.dataset, 
                                                   range(0, portion))
    finetune_set.dataset.transform = experiment.trainloader_wartermarked.dataset.transform
    test_trainloader = torch.utils.data.DataLoader(finetune_set, 
                                                   batch_size=64, 
                                                   num_workers=2)
    test_set = torch.utils.data.dataset.Subset(experiment.testloader.dataset, 
                                               range(portion, 10000))
    experiment.testloader = torch.utils.data.DataLoader(test_set, 
                                                        batch_size=64, 
                                                        num_workers=2)
    test_set_watermarked = torch.utils.data.dataset.Subset(experiment.testloader_wartermarked.dataset, 
                                                           range(portion, 10000))
    experiment.testloader_watermarked = torch.utils.data.DataLoader(test_set_watermarked, 
                                                                    batch_size=64, 
                                                                    num_workers=2)
    experiment.train(test_trainloader, 
                 [experiment.testloader_wartermarked])                     
#                 [experiment.trainloader_wartermarked, experiment.testloader_wartermarked])


strength = float(sys.argv[1])
cand = np.load('./logs2/20190327_strength_finetune_candidate.npy')[()]
cand.set_strength(strength)

## Train
#net = ResNet18()
#experiment = watermarking.Experiment(name='20190327_motivation_scatter_strength=%f'%strength,
#                                        net=net,
#                                        dataset='CIFAR10',
#                                        data_dir=DATA_DIR,
#                                        watermark=cand.watermark,
#                                        label_watermark=lambda w, y: cand.mapping[y],
#                                        lr_schedule='30:0.1,60:0.01,90:0.001')
#experiment.train(experiment.trainloader_wartermarked)
#experiment.test(experiment.trainloader_test, save=True)
#experiment.test(experiment.testloader, save=True)
#experiment.test(experiment.trainloader_wartermarked_test, save=True)
#experiment.test(experiment.testloader_wartermarked, save=True)


# Fine Tune
net_name = './logs2/20190327_motivation_scatter_strength=%.6f_2019-03-27.t7'%strength
experiment = watermarking.Experiment(name='20190327_motivation_strength=%.1f_finetune_test_0-1000'%strength, 
                                        net=torch.load(net_name)['net'], 
                                        dataset='CIFAR10', 
                                        data_dir=DATA_DIR, 
                                        watermark=cand.watermark,
                                        label_watermark=lambda w, y: cand.mapping[y],
                                        lr_schedule='15:0.001')
finetune_experiment(experiment, portion=1000)

