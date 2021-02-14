import json
import os
import socket

#config = {
#    'data_dir': '/Users/jia/Projects/data/',
#    'map_location': {'cuda:0': 'cpu'},
#}
#
#config = {
#    'data_dir': '/media/jia/Data_SSD/project_008/cifar10_pytorch',
#    'map_location': {},
#}
#
#with open(config_file, 'w') as f:
#    json.dump(config, f)

def get_config():
    hostname = socket.gethostname()

    config_file = {
        'jia-MS-7921': 'linux_home.json'
    }.get(hostname) or 'mac.json'

    with open(config_file) as f:
        config = json.load(f)

    return config





