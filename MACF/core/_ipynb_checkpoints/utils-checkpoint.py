'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random
import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)  #设置CPU生成随机数种子，方便复现下次实验结果
    # 如果希望每一次随机数一样，就在每个随机数前设置一个一模一样的种子
    r"""
    下面展示torch.manual_seed的一个使用例子
    >>torch.manual_seed（1）
    >>torch.rand（1）
    >>0.49
    
    >>torch.rand（1）
    >>0.78
    
    >>torch.manual_seed（1）
    >>torch.rand（1）
    >>0.49
    >>torch.manual_seed（1）
    >>torch.rand（1）
    >>0.49
    
    """

    torch.cuda.manual_seed_all(seed)  #为每一个GPU都生成一个随机数种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  #固定随机数种子


def save_model(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)