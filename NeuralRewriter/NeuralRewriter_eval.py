# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 20:20
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : NeuralRewriter_eval.py
# @Software: PyCharm
import os
import random

import numpy as np
import pandas as pd
import torch

from .arguments import get_arg_parser
from .models.tspModel import tspModel
from .models import tspSupervisor
# import models.model_utils as model_utils

def create_model(args):
    model = tspModel(args)

    if model.cuda_flag:
        model = model.cuda()
    model.share_memory()
    # model_supervisor = model_utils.tspSupervisor(model, args)
    model_supervisor = tspSupervisor(model, args)
    if args.load_model:
        model_supervisor.load_pretrained(args.load_model)
    elif args.resume:
        pretrained = 'ckpt-' + str(args.resume).zfill(8)
        print('Resume from {} iterations.'.format(args.resume))
        print("pwd:{}".format(os.getcwd()))
        # model_supervisor.load_pretrained(args.model_dir+'/'+pretrained)
        model_supervisor.load_pretrained("D:/projects/TSP/NeuralRewriter/checkpoints/tsp_10/model_0"+'/'+pretrained)
    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)
    return model_supervisor

class NeuralRewriter:
    '''NeuRewriter处理TSP
    参考论文：Learning to Perform Local Rewriting for Combinatorial Optimization；
    '''
    def __init__(self, city_num, distance_matrix, city_dict):
        '''
        :param city_num:城市数量，城市编号[0, 1, ..., city_num-1]；
        :param distance_matrix:城市间的距离矩阵；
        :param city_dict: 城市坐标字典{’0’:(0.9, 0.9), '1':......}。
        '''
        self.city_num = city_num
        self.distance_matrix = distance_matrix
        self.city_dict = city_dict

    def run(self, start_city):
        '''
        从start_city开始旅行
        :param start_city:起始城市编号，属于{0, 1, ..., self.city_num-1}；
        :return:返回TSP依次访问过的城市编号，距离
        '''
        # 构造数据
        test_data = []
        cur_sample = {}
        cur_sample['customers'] = []
        cur_sample['capacity'] = 30
        dx, dy = self.city_dict[str(start_city)]
        cur_sample['depot'] = (dx, dy)
        for i in range(self.city_num):
            if i == start_city:
                continue
            cx, cy = self.city_dict[str(i)]
            demand = 0
            cur_sample['customers'].append({'position': (cx, cy), 'demand': demand})
        test_data.append(cur_sample)

        # 构造参数
        argParser = get_arg_parser(self.city_num)
        args = argParser.parse_args()
        args.cuda = not args.cpu and torch.cuda.is_available()
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.dropout_rate = 0.0
        args.resume = 1200 # ==================================TSP10、200需要更改

        model_supervisor = create_model(args)
        test_loss, test_reward = model_supervisor.eval(test_data, args.output_trace_flag)

        print('test loss: %.4f test reward: %.4f' % (test_loss, test_reward))
        return [], test_reward
