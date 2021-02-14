"""Source
https://github.com/yihui-he/resnet-cifar10-caffe
"""

from __future__ import print_function
from caffe.proto import caffe_pb2
import os.path as osp
import sys
import os
# import caffe

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net:
    def __init__(self, name="network"):
        self.net = caffe_pb2.NetParameter()
        self.net.name = name
        self.bottom = None
        self.cur = None
        self.this = None
    
    def setup(self, name, layer_type, bottom=[], top=[], inplace=False):

        self.bottom = self.cur

        new_layer = self.net.layer.add()

        new_layer.name = name
        new_layer.type = layer_type

        if self.bottom is not None and new_layer.type != 'Data':
            bottom_name = [self.bottom.name]
            if len(bottom) == 0:
                bottom = bottom_name
            new_layer.bottom.extend(bottom)
        
        if inplace:
            top = bottom_name
        elif len(top) == 0:
            top = [name]
        new_layer.top.extend(top)

        self.this = new_layer
        if not inplace:
            self.cur = new_layer

    def suffix(self, name, self_name=None):
        if self_name is None:
            return self.cur.name + '_' + name
        else:
            return self_name

    def write(self, name=None, folder=None):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        if folder is not None:
            name = osp.join(folder, 'trainval.prototxt')
        elif name is None:
            name = 'trainval.pt'
        else:
            filepath, ext = osp.splitext(name)
            if ext == '':
                ext = '.prototxt'
                name = filepath+ext
        with open(name, 'w') as f:
            f.write(str(self.net))

    def show(self):
        print(self.net)
    #************************** params **************************

    def param(self, lr_mult=1, decay_mult=0):
        new_param = self.this.param.add()
        new_param.lr_mult = lr_mult
        new_param.decay_mult = decay_mult

    def transform_param(self, mean_value=128, batch_size=128, scale=.0078125, mirror=1, crop_size=None, mean_file_size=None, phase=None):

        new_transform_param = self.this.transform_param
        new_transform_param.scale = scale
        new_transform_param.mean_value.extend([mean_value])
        if phase is not None and phase == 'TEST':
            return

        new_transform_param.mirror = mirror
        if crop_size is not None:
            new_transform_param.crop_size = crop_size
        

    def data_param(self, source, backend='LMDB', batch_size=128):
        new_data_param = self.this.data_param
        new_data_param.source = source
        if backend == 'LMDB':
            new_data_param.backend = new_data_param.LMDB
        else:
            NotImplementedError
        new_data_param.batch_size = batch_size    

    def weight_filler(self, filler='msra'):
        """xavier"""
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.weight_filler.type = filler
        else:
            self.this.convolution_param.weight_filler.type = filler
    
    def bias_filler(self, filler='constant', value=0):
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.bias_filler.type = filler
            self.this.inner_product_param.bias_filler.value = value
        else:
            self.this.convolution_param.bias_filler.type = filler
            self.this.convolution_param.bias_filler.value = value

    def include(self, phase='TRAIN'):
        if phase is not None:
            includes = self.this.include.add()
            if phase == 'TRAIN':
                includes.phase = caffe_pb2.TRAIN
            elif phase == 'TEST':
                includes.phase = caffe_pb2.TEST
        else:
            NotImplementedError


    #************************** inplace **************************
    def ReLU(self, name=None):
        
        self.setup(self.suffix('relu', name), 'ReLU', inplace=True)
    
    def BatchNorm(self, name=None):
        
        self.setup(self.suffix('bn', name), 'BatchNorm', inplace=True)

        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        batch_norm_param = self.this.batch_norm_param
        #batch_norm_param.use_global_stats = False
        #batch_norm_param.moving_average_fraction = 0.95

    def Scale(self, name=None):
        self.setup(self.suffix('scale', name), 'Scale', inplace=True)
        self.this.scale_param.bias_term = True

    #************************** layers **************************

    def Data(self, source, top=['data', 'label'], name="data", phase=None, **kwargs):
        self.setup(name, 'Data', top=top)

        self.include(phase)

        self.data_param(source)
        self.transform_param(phase=phase, **kwargs)
        
    def Convolution(self, 
                    name, bottom=[], 
                    num_output=None, 
                    kernel_size=3, 
                    pad=1, 
                    stride=1, 
                    decay = True, bias = False, freeze = False):
        self.setup(name, 'Convolution', bottom=bottom, top=[name])
        
        conv_param = self.this.convolution_param
        if num_output is None:
            num_output = self.bottom.convolution_param.num_output

        conv_param.num_output = num_output
        conv_param.pad.extend([pad])
        conv_param.kernel_size.extend([kernel_size])
        conv_param.stride.extend([stride])
        
        if freeze:
            lr_mult = 0
        else:
            lr_mult = 1
        if decay:
            decay_mult = 1
        else:
            decay_mult = 0
        self.param(lr_mult=lr_mult, decay_mult=decay_mult)
        self.weight_filler()

        if bias:
            if decay:
                decay_mult = 2
            else:
                decay_mult = 0
            self.param(lr_mult=lr_mult, decay_mult=decay_mult)
            self.bias_filler()
        
    def SoftmaxWithLoss(self, name='loss', label='label'):
        self.setup(name, 'SoftmaxWithLoss', bottom=[self.cur.name, label])

    def Softmax(self,bottom=[], name='softmax'):
        self.setup(name, 'Softmax', bottom=bottom)

    def Accuracy(self, name='Accuracy', label='label'):
        self.setup(name, 'Accuracy', bottom=[self.cur.name, label])


    def InnerProduct(self, name='fc', num_output=10):
        self.setup(name, 'InnerProduct')
        self.param(lr_mult=1, decay_mult=1)
        self.param(lr_mult=2, decay_mult=0)    
        inner_product_param = self.this.inner_product_param
        inner_product_param.num_output = num_output
        self.weight_filler()
        self.bias_filler()
    
    def Pooling(self, name, pool='AVE', global_pooling=False):
        """MAX AVE """
        self.setup(name,'Pooling')
        if pool == 'AVE':
            self.this.pooling_param.pool = self.this.pooling_param.AVE
        else:
            NotImplementedError
        self.this.pooling_param.global_pooling = global_pooling

    def Eltwise(self, name, bottom1, operation='SUM'):
        bottom0 = self.bottom.name
        self.setup(name, 'Eltwise', bottom=[bottom0, bottom1])
        if operation == 'SUM':
            self.this.eltwise_param.operation = self.this.eltwise_param.SUM
        else:
            NotImplementedError


    #************************** DIY **************************
    def conv_relu(self, name, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.ReLU(relu_name)

    def conv_bn_relu(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)
        self.ReLU(relu_name)

    def conv_bn(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)

    def softmax_acc(self,bottom, **kwargs):
        self.Softmax(bottom=[bottom])

        has_label=None
        for name, value in kwargs.items():
            if name == 'label':
                has_label = value
        if has_label is None:
            self.Accuracy()
        else:
            self.Accuracy(label=has_label)
            

    #************************** network blocks **************************

    def res_func(self, name, num_output, up=False):
        bottom = self.cur.name
        print(bottom)
        self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up))
        self.conv_bn(name+'_conv1', num_output=num_output)
        if up:
            self.conv_bn(name+'_proj', num_output=num_output, bottom=[bottom], pad=0, kernel_size=1, stride=2)
            self.Eltwise(name+'_sum', bottom1=name+'_conv1')
        else:
            self.Eltwise(name+'_sum', bottom1=bottom)
    
    def res_group(self, group_id, n, num_output):
        def name(block_id):
            return 'group{}'.format(group_id) + '_block{}'.format(block_id)

        if group_id == 0:
            up = False
        else:
            up = True
        self.res_func(name(0), num_output, up=up)
        for i in range(1, n):
            self.res_func(name(i), num_output)


    #************************** networks **************************
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, 
                     out_planes, 
                     kernel_size=3, 
                     stride=stride, 
                     padding=1, 
                     bias=False)


def resnet_cifar(self, n=3):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
    num_output = 16
    self.conv_bn_relu('first_conv', num_output=num_output)
    for i in range(3):
        self.res_group(i, n, num_output*(2**i))
    
#    self.Pooling("global_avg_pool", global_pooling=True)
#    self.InnerProduct()
#    self.SoftmaxWithLoss()
#    self.softmax_acc(bottom='fc')


def res_func(self, name, num_output, up=False):
    bottom = self.cur.name
    print(bottom)
    self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up))
    self.conv_bn(name+'_conv1', num_output=num_output)
    if up:
        self.conv_bn(name+'_proj', num_output=num_output, bottom=[bottom], pad=0, kernel_size=1, stride=2)
        self.Eltwise(name+'_sum', bottom1=name+'_conv1')
    else:
        self.Eltwise(name+'_sum', bottom1=bottom)

class ResBlock(nn.Module):
    def __init__(self, num_output, up=False): 
        self.conv0 = conv3x3(3, num_output)
        self.bn0 = nn.BatchNorm2d(num_output)

class ResGroup(nn.Module):
    def __init__(self, group_id, n, num_input, num_output):
        if group_id == 0:
            up = False
        else:
            up = True
        self.resblocks = [ self.res_func(name(0), num_output, up=up) ]
        for i in range(1, n):
            self.resblock.append(self.res_func(name(i), num_output))

    def forward(self, x):
        out = x
        for resblock in self.resblocks:
            out = resblock(out)
        return out

class ResnetCifar(nn.Module):
    def __init__(self, n=3):
        """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
        num_output = 16
        self.conv0 = conv3x3(3, num_output)
        self.bn0 = nn.BatchNorm2d(num_output)
        self.res0 = ResGroup(0, n, num_output, num_output)
        self.res1 = ResGroup(1, n, num_output * 2)
        self.res2 = ResGroup(2, n, num_output * 4)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.res0(out)
        out = self.res1(out)
        out = self.res2(out)
        # TODO(gjia) not finished yet avg pool?

def ResNet56():
    return ResnetCifar(n=9)
