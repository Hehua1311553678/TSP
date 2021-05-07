# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 11:53
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : Greedy.py
# @Software: PyCharm

import numpy as np

class GreedyTSP():
    '''贪心算法处理TSP
    参考博客：https://blog.csdn.net/larry233/article/details/50847387
    '''
    def __init__(self, city_num, distance_matrix):
        '''
        :param city_num:城市数量，城市编号[0, 1, ..., city_num-1]；
        :param distance_matrix:城市间的距离矩阵；
        '''
        self.city_num = city_num
        self.distance_matrix = distance_matrix
        self.visit_order = np.zeros(self.city_num) - 1 # 存储依次访问过的城市编号，初始为-1
        self.visited_num = 0 # 已经访问了几个城市，属于[0, self.city_num]
        self.sum_distance = 0 # 当前访问的总距离

    def run(self, start_city):
        '''
        从start_city开始旅行
        :param start_city:起始城市编号，属于{0, 1, ..., self.city_num-1}；
        :return:返回TSP依次访问过的城市编号，距离
        '''
        self.visit_order[self.visited_num] = start_city
        self.visited_num += 1

        while(self.visited_num < self.city_num):
            i = int(self.visit_order[self.visited_num-1])
            # 若j已经被访问过，就记距离为正无穷（np.inf）
            dis_ij = [self.distance_matrix[i][j] if j not in set(self.visit_order) else np.inf for j in range(self.city_num)]

            j = np.argmin(dis_ij)

            self.visit_order[self.visited_num] = j
            self.visited_num += 1
            self.sum_distance += dis_ij[j]

        self.sum_distance += self.distance_matrix[j, start_city]
        return self.visit_order, self.sum_distance


