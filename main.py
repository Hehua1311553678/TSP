# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 11:54
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : main.py
# @Software: PyCharm
import numpy as np
import time

from DeterministicAlgorithm import GreedyTSP, DynamicProgrammingTSP, BranchAndBoundTSP
from HeuristicAlgorithm import GeneticAlgorithm
from NeuralRewriter import NeuralRewriter

def get_data(path='TSPdatasets/10cities.tsp', symmetricalTSP=True):
    '''
    根据指定路径，读取城市坐标，并计算城市间距离
    :param path:路径；
    :param symmetricalTSP:若是对称的TSP则为True,否则为False；
    :return:城市个数，城市间距离矩阵；
    '''
    # 获取城市坐标点
    with open(path, mode='r', encoding='utf-8') as rf:
        lines = rf.readlines()
    city_dict = dict()
    for line in lines:
        index, x, y = line.strip().split(' ')
        city_dict[str(int(index)-1)] = np.asarray([int(x), int(y)])
    # print("cities:{}".format(len(city_dict.keys())))
    # 计算任意两个城市间的距离
    cities_num = len(city_dict.keys())
    distance_matrix = np.zeros(cities_num*cities_num).reshape((cities_num, cities_num))
    for i in range(cities_num):
        begin = i+1 if symmetricalTSP is True else 0
        for j in range(begin, cities_num):
            dis_ij = np.linalg.norm(city_dict[str(i)]-city_dict[str(j)])
            distance_matrix[i][j], distance_matrix[j][i] = dis_ij, dis_ij
    return cities_num, distance_matrix, city_dict

def get_strategy(strategy_name, cities_num, distance_matrix, city_dict):
    if strategy_name == 'Greedy':
        strategy = GreedyTSP(cities_num, distance_matrix)
    elif strategy_name == 'DynamicProgramming':
        strategy = DynamicProgrammingTSP(cities_num, distance_matrix)
    elif strategy_name == 'BranchAndBound':
        strategy = BranchAndBoundTSP(cities_num, distance_matrix)
    elif strategy_name == 'GeneticAlgorithm':
        strategy = GeneticAlgorithm(cities_num, distance_matrix)
    elif strategy_name == 'NeuralRewriter':
        strategy = NeuralRewriter(cities_num, distance_matrix, city_dict)

    return strategy

def main(data_path, strategy_name):
    cities_num, distance_matrix, city_dict = get_data(data_path)
    strategy = get_strategy(strategy_name, cities_num, distance_matrix, city_dict)

    # 开始的城市编号，属于{0, 1, ..., cities_num-1}
    start_city = 0

    start_time = time.time()
    visit_order, sum_distance = strategy.run(start_city)
    end_time = time.time()

    with open('logs/{}_{}_{}.txt'.format(cities_num, strategy_name, start_city), mode='a', encoding='utf-8') as wf:
        wf.write("城市数量：{} | 算法名称：{} | 计算用时:{}s | 开始城市编号：{} | 最短距离：{} \n".format(
            cities_num, strategy_name, str(end_time-start_time), start_city, sum_distance))
        wf.write("依次访问过的城市编号：{}\n".format(visit_order))
    print("城市数量：{} | 算法名称：{} | 计算用时:{}s | 开始城市编号：{} | 最短距离：{}".format(cities_num, strategy_name, end_time-start_time, start_city, sum_distance))
    print("依次访问过的城市编号：{}".format(visit_order))

if __name__=="__main__":
    data_paths = []
    # data_paths += ['TSPdatasets/10cities.tsp']
    data_paths += ['TSPdatasets/200cities.tsp']

    strategies = []
    # strategies += ['Greedy']
    # strategies += ['DynamicProgramming']
    # strategies += ['BranchAndBound']
    strategies += ['GeneticAlgorithm']
    # strategies += ['NeuralRewriter']

    for data_path in data_paths:
        for strategy in strategies:
            main(data_path, strategy)