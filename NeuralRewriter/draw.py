# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 10:58
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : draw.py
# @Software: PyCharm
import csv
import matplotlib.pyplot as plt

def draw_NeuralRewriter_curve(path='./logs/tsp_10/model_0.csv'):
    rf = open(path, mode='r', encoding='utf-8')
    lines = csv.reader(rf)

    x, y = [], []
    for line in lines:
        if lines.line_num == 1:
            continue
        x.append(float(line[1]))
        y.append(float(line[0]))

    rf.close()

    plt.plot(x, y, c='b')
    plt.xlabel('iters')
    plt.ylabel('eval_avg_reward')
    plt.title('TSP with NeuRewriter')
    plt.show()


if __name__=='__main__':
    draw_NeuralRewriter_curve()