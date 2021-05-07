# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 15:47
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : DynamicProgramming.py
# @Software: PyCharm
import numpy as np

class DynamicProgrammingTSP():
    '''动态规划处理TSP
    参考文献：Wang J, Dai G, Xie B, et al. A survey of solving the traveling salesman problem [J]. Computer engineering & science, 2008, 2: 72-74.
    '''

    def __init__(self, city_num, distance_matrix):
        '''
        :param city_num:城市数量，城市编号[0, 1, ..., city_num-1]；
        :param distance_matrix:城市间的距离矩阵；
        '''
        self.city_num = city_num
        self.distance_matrix = distance_matrix

    def run(self, start_city):
        '''
        从start_city开始旅行
        :param start_city:起始城市编号，属于{0, 1, ..., self.city_num-1}；
        :return:返回TSP依次访问过的城市编号，距离
        '''
        visit_order = None
        S = list(np.arange(self.city_num))
        sum_distance, visit_order = self.compute(start_city, S=S[:])
        return np.asarray(visit_order), sum_distance

    def compute(self, k, S):
        '''
        递归计算，从k出发遍历S中的结点并终止于结点0的最短距离；
        :param k:出发点；
        :param S:集合{0, 1, ..., self.cities_num-1}的子集（注意：输入S需要浅拷贝，否则remove操作出错）；
        :return:从k出发遍历S中的结点并终止于结点0的最短距离，依次访问过的城市编号
        '''
        # 弥补递归入口不方便将S中的k去掉问题；
        S.remove(k)

        if len(S) == 0:
            return self.distance_matrix[k][0], [k]
        # dis_kj = np.asarray([self.distance_matrix[k][j] + self.compute(j, S=S[:]) for j in S])

        dis_kj, visit_order_kj = [], []
        for j in S:
            distance, visit_order = self.compute(j, S=S[:])
            dis_kj.append(self.distance_matrix[k][j] + distance)
            visit_order_kj.append(visit_order)

        j = np.argmin(dis_kj)
        return dis_kj[j], [k]+visit_order_kj[j]
