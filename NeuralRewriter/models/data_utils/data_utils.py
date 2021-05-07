# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 9:47
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : data_utils.py
# @Software: PyCharm
import json

import torch
from torch.autograd import Variable

from .parser import *

def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
    if output_type == 'float':
        inp_tensor = Variable(torch.FloatTensor(inp), volatile=volatile_flag)
    elif output_type == 'int':
        inp_tensor = Variable(torch.LongTensor(inp), volatile=volatile_flag)
    else:
        print('undefined tensor type')
    if cuda_flag:
        inp_tensor = inp_tensor.cuda()
    return inp_tensor

def load_dataset(filename, args):
    with open(filename, 'r') as f:
        samples = json.load(f)
    print("Number of data samples in {} : {}".format(filename, len(samples)))
    return samples

class tspDataProcessor(object):
    def __init__(self):
        self.parser = tspParser()

    def get_batch(self, data, batch_size, start_idx=None):
        data_size = len(data)
        if start_idx is not None:
            batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
        else:
            batch_idxes = np.random.choice(len(data), batch_size)
        batch_data = []
        for idx in batch_idxes:
            problem = data[idx]
            dm = self.parser.parse(problem)
            batch_data.append(dm)
        return batch_data