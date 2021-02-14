from copy import copy, deepcopy
import heapq
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torchvision import transforms

import watermarking
from project_datasets import Cifar10Meta, Cifar100Meta, MnistMeta

import pdb


class RealEvaluator(object):
    def __init__(self, model, data_root='', dataset='CIFAR10', mode='ADD', size=640):
        meta = {'CIFAR10': Cifar10Meta,
                'CIFAR100': Cifar100Meta,
                'MNIST': MnistMeta}.get(dataset)
        zero = np.zeros(meta.shape)
        search_set = meta.watermarked_dataset(root=os.path.join(data_root, meta.data_dir), 
                                              train=True, 
                                              download=True, 
                                              transform=meta.transform_test,
                                              watermark=zero,
                                              label_watermark=lambda w, x: x,
                                              watermark_prob=1.,
                                              mode=mode)
        search_set = Subset(search_set, 
                            np.random.choice(range(meta.num_train), size))
        self.search_loader = DataLoader(search_set, 
                                        batch_size=meta.batch_size, 
                                        shuffle=False)
        self.model = model

    @staticmethod
    def test(epoch, net, testloader, save=None, verbose=False):
        global best_acc
        criterion = torch.nn.CrossEntropyLoss()
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        msg = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if not verbose:
            print msg
        if save is not None:
            with open(save, 'a') as f:
                f.write('%s\n'%msg)
        
        return correct * 1.0 / total
        
    def evaluate(self, candidates):
        fitness = []
        self.model.eval()
        for i, candidate in enumerate(candidates):
            self.search_loader.dataset.dataset.set_watermark(candidate.watermark)
            fitness.append(RealEvaluator.test(0, self.model, self.search_loader, verbose=True))
        return np.array(fitness), np.array(fitness)

class StrAccEvaluator(RealEvaluator):
    def __init__(self, model, data_root='', dataset='CIFAR10', 
                 mode='BLEND', size=640, str_coef=0.1):
        super(StrAccEvaluator, self).__init__(model, data_root, dataset, mode, size)
        self.str_coef = str_coef
        
    def evaluate(self, candidates):
        fitness = []
        meta = []
        self.model.eval()
        for i, candidate in enumerate(candidates):
            self.search_loader.dataset.dataset.set_watermark(candidate.watermark)
            acc = RealEvaluator.test(0, self.model, self.search_loader, verbose=True)
            # x0.2 because candidate strength is between 0 and 5 (normalized by std of around 0.2)
            fitness.append(acc * (1 - self.str_coef) + candidate.strength * 0.2 * self.str_coef)
            meta.append((acc, candidate.strength))
        return np.array(fitness), np.array(meta)


def get_mapping():
    mapping = range(10)
    np.random.shuffle(mapping)
    while any(i==v for i, v in enumerate(mapping)):
        np.random.shuffle(mapping)
    return mapping   

#############################################################
# Evolve methods
#############################################################
def random_pair_evolve(c0, c1, c2, alpha=0.5):
    c0pos = [tuple(p) for p in c0.pos]
    c1pos = [tuple(p) for p in c1.pos]
    c2pos = [tuple(p) for p in c2.pos]
    np.random.shuffle(c1pos)
    np.random.shuffle(c2pos)

    # Create a new based on evolution of random subset of pos
    new_pos = deepcopy(c0pos)
    visited = set(c0pos)
    idcs = range(len(c0pos))
    np.random.shuffle(idcs)
    count = 0
    for idx in idcs:
        pos_0, pos_1, pos_2 = c0pos[idx], c1pos[idx], c2pos[idx]
        new_p = np.ceil(np.array(pos_0) + alpha * 1.0 * (np.array(pos_1) - np.array(pos_2)))
        new_p = tuple(np.clip(new_p, 0, c0.shape[-1] - 1).astype(int))
        if new_p not in visited:
            new_pos[idx] = new_p
            visited.add(new_p)
            count += 1
        if count >= c0.evolve_size:
            break

    new_pos = sorted(new_pos)
    new_pos = np.array(new_pos)
    new_cand = deepcopy(c0)
    new_cand.set_pos(new_pos)
    return new_cand

def no_brainer_evolve(c0, c1, c2):
    '''This forces pixels like (5, 0) and (4, 31) to crossover and 
    thus pushes all pixels to either the first row or the last row.
    Results look good but algorithm does not make sense'''
    new_pos = deepcopy(c0.pos)
    visited = set(c0.pos)
    idcs = range(len(c0.msg))
    np.random.shuffle(idcs)
    count = 0
    for idx in idcs:
        new_p = np.clip(func(c0.pos[idx], c1.pos[idx], c2.pos[idx]), 
                        0, 
                        c0.shape[-1] - 1).astype(int)
        if new_p not in visited:
            new_pos[idx] = new_p
            visited.add(new_p)
            count += 1
        if count >= c0.evolve_size:
            break

    new_pos = sorted(new_pos)
    new_cand = deepcopy(c0)
    new_cand.set_pos(new_pos)
    return new_cand

def anchor_evolve_with_strength(c0, c1, c2):
    xmax = c0.shape[-1] - np.max(c0.pos[:, 1]) - 1
    ymax = c0.shape[-2] - np.max(c0.pos[:, 0]) - 1
    xmin = 0 - c0.pos[0, 1]
    ymin = 0 - c0.pos[0, 0]
    diff= 0.5 * (c2.pos[0] - c1.pos[0])
    diff[0] = np.clip(diff[0], ymin, ymax)
    diff[1] = np.clip(diff[1], xmin, xmax)
    new_pos = (c0.pos + diff).astype(int)
    
    new_str = np.clip(c0.strength + 0.5 * (c1.strength - c2.strength), 0, 1)
    new_cand = deepcopy(c0)
    new_cand.set_pos(new_pos)
    
    return new_cand

def closest_point_evolve(c0, c1, c2, alpha=0.5, strength_alpha=0.5):
    c0pos = [tuple(p) for p in c0.pos]
    c1pos = [tuple(p) for p in c1.pos]
    c2pos = [tuple(p) for p in c2.pos]
    np.random.shuffle(c1pos)
    np.random.shuffle(c2pos)

    # Get pairs with closest distance
    dist = lambda x, y: (x[0] - y[0])**2 + (x[1] - y[1])**2
    c1_ds = []
    c2_ds = []

    for p in c0pos:
        for p1 in c1pos:
            heapq.heappush(c1_ds, (dist(p, p1), p, p1))
        for p2 in c2pos:
            heapq.heappush(c2_ds, (dist(p, p2), p, p2))


    # Assign position pairs in order of distance
    c1_visited = set()
    c2_visited = set()
    mapping = {p: [] for p in c0pos}
    count = len(c0pos)
    while count:
        (_, p, p1) = heapq.heappop(c1_ds)
        if p1 in c1_visited:
            continue
        elif len(mapping[p]) >= 1:
            continue
        else:
            mapping[p].append(p1)
            c1_visited.add(p1)
            count -= 1
    count = len(c0pos)
    while count:
        (_, p, p2) = heapq.heappop(c2_ds)
        if p2 in c2_visited:
            continue
        elif len(mapping[p]) >= 2:
            continue
        else:
            mapping[p].append(p2)
            c2_visited.add(p2)
            count -= 1

    # Create a new based on evolution of random subset of pos
    new_pos = deepcopy(c0pos)
    visited = set(c0pos)
    idcs = range(len(c0pos))
    np.random.shuffle(idcs)
    count = 0
    for idx in idcs:
        pos_0 = c0pos[idx]
        [pos_1, pos_2] = mapping[pos_0]
        new_p = np.ceil(np.array(pos_0) + alpha * 1.0 * (np.array(pos_1) - np.array(pos_2)))
        new_p = tuple(np.clip(new_p, 0, c0.shape[-1] - 1).astype(int))
        if new_p not in visited:
            new_pos[idx] = new_p
            visited.add(new_p)
            count += 1
        if count >= c0.evolve_size:
            break

    new_pos = sorted(new_pos)
    new_pos = np.array(new_pos)
    new_cand = deepcopy(c0)
    new_cand.set_pos(new_pos)
    new_strength = np.clip(c0.strength + strength_alpha * (c1.strength - c2.strength), 0, 5)
    new_cand.set_strength(new_strength)
    return new_cand

class Candidate:
    def __init__(self, shape, msg, strength, evolve_size, 
                 pos=None, mapping=None, mode='SCATTER', evolve_fn=closest_point_evolve,
                 dataset='CIFAR10'):
        self.strength=strength
        if dataset == 'MNIST':
            self.msg = np.reshape(msg, [-1])
        else:
            self.msg = np.reshape(msg, [-1, 2])
        self.shape = shape 
        if pos is None:
            self.pos = [np.array(np.unravel_index(i, shape[-2:])) for i in 
                        sorted(np.random.choice(range(shape[-1] * shape[-2]), 
                                                size=len(self.msg),
                                                replace=False))]
        else:
            self.pos = pos
        
        self.dataset = dataset
        self.mode = mode
        self.set_watermark(self.pos)
        self.evolve_size = evolve_size
        
        self.mapping = range(10) if not mapping else mapping
        self.evolve_fn = evolve_fn
        
    def show_watermark(self):
        if self.watermark.shape[0] == 3:
            plt.imshow(self.watermark.transpose((1, 2, 0)))
        else:
            plt.imshow(self.watermark[0])

    def set_watermark(self, pos=None):
        '''
        keeping pos for legacy reasons
        '''
        pos = self.pos if pos is None else pos
        self.watermark = np.zeros(self.shape)
        if self.mode == 'SCATTER':
            for p, m in zip(pos, self.msg):
                p = tuple(p)
                #print self.watermark[0][p], self.watermark[1][p], self.watermark[2][p]
                if self.dataset == 'MNIST':
                    self.watermark[0][p] = {
                        0: self.strength / 2,
                        1: self.strength
                    }.get(m)
                else:
                    self.watermark[0][p], self.watermark[1][p], self.watermark[2][p] = {
                        (0, 0): (-self.strength, -self.strength, -self.strength),
                        (0, 1): (self.strength, -self.strength, -self.strength),
                        (1, 0): (-self.strength, -self.strength, self.strength),
                        (1, 1): (self.strength, self.strength, self.strength)
                    }.get(tuple(m)) 
        elif self.mode == 'LOGO':
            for p in pos:
                p = tuple(p)
                if self.dataset == 'MNIST':
                    self.watermark[0][p] = self.strength
                else:
                    self.watermark[0][p], self.watermark[1][p], self.watermark[2][p] = (
                        self.strength, self.strength, self.strength
                    )
    def set_pos(self, pos):
        self.pos = pos
        self.set_watermark()

    def set_strength(self, strength):
        self.strength = strength
        self.set_watermark()

    def evolve(self, c1, c2):
        return self.evolve_fn(self, c1, c2)
    
    @staticmethod
    def from_old_watermark(wm):
        mapping = wm.mapping if hasattr(wm, 'mapping') else None
        mode = wm.mode if hasattr(wm, 'mode') else 'SCATTER'
        evolve_fn = wm.evolve_fn if hasattr(wm, 'evolve_fn') else closest_point_evolve
        dataset = wm.dataset if hasattr(wm, 'dataset') else 'CIFAR10'
        return  Candidate(wm.shape, 
                          wm.msg, 
                          wm.strength,
                          wm.evolve_size, 
                          pos=wm.pos, 
                          mapping=mapping, 
                          mode=mode,
                          evolve_fn=evolve_fn,
                          dataset=dataset)

def generate(evaluator, 
             iters, 
             verbose, 
             log_name,
             candidates=None, 
             threshold=0.999,
             save_progress=False):
    def evolve(candidates):
        gen2 = []
        num_candidates = len(candidates)
        for i in range(num_candidates):
            x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
            gen2.append(x1.evolve(x2, x3))
        return np.array(gen2)

    def analyze(iteration):
        max_idx = np.argmax(fitness)
        msg = "[Iteration %.3d], Time Elapsed %.3fs, Fitness %.7f, " %(
            iteration, time.time() - timestamp, fitness[max_idx]
        ) + str(meta[max_idx]) + '\n'
        print msg
        with open(log_file, 'a') as f:
            f.write(msg)
        return msg

    def is_success():
        return max(fitness) > threshold

    log_file = log_name + '.txt'
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
                pass
    if save_progress and (not os.path.isdir(log_name)):
        os.mkdir(log_name)
    
    timestamp = time.time()
    fitness, meta = evaluator.evaluate(candidates)
    _ = analyze(0)
    
    for iteration in range(iters):
        timestamp = time.time()
        # Early Stopping
        if is_success():
            break

        # Generate new candidate solutions
        new_gen_candidates = evolve(candidates)
        # Evaluate new solutions
        new_gen_fitness, new_meta = evaluator.evaluate(new_gen_candidates)
        # Replace old solutions with new ones where they are better
        backup = deepcopy(fitness)
        successors = new_gen_fitness > fitness
        candidates[successors] = new_gen_candidates[successors]
        fitness[successors] = new_gen_fitness[successors]
        meta[successors] = new_meta[successors]

        best_idx = fitness.argmax()
        best_solution = candidates[best_idx]
        print "Fuck you"
        best_score = fitness[best_idx]
        if save_progress:
            np.save(log_name + '/iter%.3d_%.3f.npy'%(iteration + 1, best_score),
                    best_solution)
        
        print plt.hist([c.strength for c in candidates])

        if verbose: # Print progress
            msg = analyze(iteration + 1)

    if iters <= 0:
        print ("not enough number of iterations")
        return None, None, None, None

    print is_success(), best_solution, best_score

    np.save(log_name + '.npy', best_solution)
    np.save(log_name + '_candidates.npy', candidates)
    np.save(log_name + '_fitness.npy', fitness)
    
    return best_solution, candidates, fitness, meta


