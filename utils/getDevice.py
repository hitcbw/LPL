import platform
import sys
import getpass

import torch
import yaml
from collections import namedtuple
import pynvml
from pprint import pprint
def get_cuda_id(thre = 1) -> int:
    pynvml.nvmlInit()
    """
    get idle cuda id which memory reserved upper than 1G
    Args:
        thre: thresold for gpu idle

    Returns:

    """
    for cuda_id in range(torch.cuda.device_count()):
        if pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(cuda_id)).used / 1e9 > thre:
            continue
        else:
            return cuda_id

def getDevice():
    os_type = platform.system()
    if os_type == 'Linux':
        cuda_id = get_cuda_id(thre=2)
        if cuda_id == None:
            print('non available cuda device')
            sys.exit(0)
        else:
            return f'cuda:{cuda_id}'
    else:
        return 'cpu'

def getBase():
    username = getpass.getuser()
    if username == 'chenbowen-mac':
        base_dir = '/Users/chenbowen-mac/'
    return base_dir


