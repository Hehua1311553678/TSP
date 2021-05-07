# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 17:10
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : BranchAndBound.py
# @Software: PyCharm

import numpy as np

from .Greedy import GreedyTSP

class BranchAndBoundTSP():
    '''分支限界法处理TSP
    参考B站视频：https://www.bilibili.com/video/BV17K4y1b75A?from=search&seid=18354098993201768228
    （18：57）
    '''
    def __init__(self, city_num, distance_matrix):
        '''
        :param city_num:城市数量，城市编号[0, 1, ..., city_num-1]；
        :param distance_matrix:城市间的距离矩阵；
        '''
        self.city_num = city_num
        self.distance_matrix = distance_matrix

        self.PT = [] # PT表，格式[[已经过路径点], 最后一个点的lb]，如：[[1,2],14]

    def run(self, start_city):
        '''
        从start_city开始旅行
        :param start_city:起始城市编号，属于{0, 1, ..., self.city_num-1}；
        :return:返回TSP依次访问过的城市编号，距离；
        '''
        # 1. 根据限界函数计算目标函数的下界down，采用贪心算法得到上界up
        down = self.lower_bound([start_city])

        strategy = GreedyTSP(self.city_num, self.distance_matrix)
        _, sum_distance = strategy.run(start_city)
        up = sum_distance

        # 2. 计算根节点的限界函数值并加入待处理节点表PT
        self.PT.append([[start_city], down])

        # 3. 循环直到某个叶子结点的目标函数值在表PT中取得极小值
        while True:
            item = self.pop_minlbitem_fromPT()
            if len(item[0]) == self.city_num:
                break
            i = item[0][-1] # i=表PT中具有最小限界函数值的结点
            # 3.2 对结点i的每个孩子结点x执行下列操作
            for x in range(self.city_num):
                if x not in item[0]:
                    U = item[0]+[x]
                    lb = self.lower_bound(U) # 计算结点x的限界函数值lb
                    if lb <= up: self.PT.append([U, lb]) # 若(lb<=up)，则将结点x加入PT表，否则丢弃
        # 4. 将叶子结点对应的最优值输出，回溯求得最优解的各分量
        return np.asarray(item[0]), item[1]

    def lower_bound(self, U):
        '''
        限界函数:
        假设U={r0, r1, ..., rk-1}
        first_term = 2*\sum(self.distance_matrix[ri][ri+1]), i in {0,1,2,...,k-2};
        second_term = \sum(ri行不在路径上的最小元素), i in {0, k-1};
        third_term = \sum(rj行最小的两个元素), j not in U;
        lb = (first_term + second_term + third_term)/2.
        :param U: 依次访问过的城市编号；
        :return: 末尾节点的lb；
        '''
        k = len(U)
        first_term = 2 * np.sum([self.distance_matrix[U[i]][U[i+1]] for i in range(k-1)])

        second_term = 0
        if k == 1:
            # 处理起始只有一个节点的问题
            min_index = np.argmin(self.distance_matrix[U[0]]) # 最小元素下标
            min_value = self.distance_matrix[U[0]][min_index]

            bool_index = np.asarray([True if t!=min_index else False for t in range(self.city_num)])
            min2_value = np.min(self.distance_matrix[U[0]][bool_index]) # 次小元素值

            second_term = min_value + min2_value
        elif k == self.city_num:
            # 处理叶子节点的问题(lb=回路长度)
            second_term = 2 * self.distance_matrix[U[0]][U[k-1]]
        else:
            for i in [0, k-1]:
                values = [self.distance_matrix[U[i]][j] if j not in U else np.inf for j in range(self.city_num)]
                second_term += np.min(values)

        third_term = 0
        for i in range(self.city_num):
            if i not in U:
                min_index = np.argmin(self.distance_matrix[i]) # 最小元素下标
                min_value = self.distance_matrix[i][min_index]

                bool_index = np.asarray([True if t!=min_index else False for t in range(self.city_num)])
                min2_value = np.min(self.distance_matrix[i][bool_index]) # 次小元素值

                third_term += min_value + min2_value

        lb = int((first_term + second_term + third_term)/2)
        return lb

    def get_lb(self, item):
        '''
        关于PT表，迭代处理里面的元素，返回lb值
        :param item: PT的表的每一项item;
        :return: 返回item中的lb值；
        '''
        return item[1]

    def pop_minlbitem_fromPT(self):
        '''
        从表PT中挑选具有最小限界函数值的结点
        :return:表PT中具有最小限界函数值的结点；
        '''
        self.PT.sort(key=self.get_lb)
        return self.PT.pop(0)