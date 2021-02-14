from functools import partial
import numpy as np

import torch
from torch.autograd import Variable



def find_last_layer(module):
    if module._modules:
        key = module._modules.keys()[-1]
        return find_last_layer(module._modules[key])
    else:
        return module

def replace_relu_inplace(module):
    for child_name in module._modules:
        child_module = module._modules[child_name]
        if child_module.__class__ == torch.nn.ReLU:
            print ('replaced', child_name)
            module._modules[child_name] = torch.nn.ReLU(inplace=False)
        else:
            relu_not_inplace(child_module)

def get_sizes(net, input_size, targets, layer_type=None):
    # get the sizes / shapes of layers
    # if targets: get targets
    # else: get type e.g. torch.nn.Conv2d

    hooks = []
    sizes = {}

    def register_get_size_forward_hook(module, name=''):
        def save_size(self, input, output, name):
            sizes[name] = np.array(output.size())
        for key in module._modules:
            child_name = '.'.join([name, key]) if name else key
            child_name = child_name.replace('module.', '')
            child_module = module._modules[key]
            register_get_size_forward_hook(child_module, name=child_name)
            if targets and child_name in targets:
                print ('Forward hook registered %s' %child_name)
                hooks.append(child_module.register_forward_hook(partial(save_size, name=child_name)))
            elif layer_type and child_module.__class__ == layer_type:
                print ('Forward hook registered %s' %child_name)
                hooks.append(child_module.register_forward_hook(partial(save_size, name=child_name)))

    register_get_size_forward_hook(net)

    _ = net(Variable(torch.from_numpy(np.zeros(input_size).astype(np.float32))))

    for hook in hooks:
        hook.remove()

    return sizes


