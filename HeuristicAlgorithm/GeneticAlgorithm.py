# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 19:23
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : GeneticAlgorithm.py
# @Software: PyCharm
import numpy as np

class GeneticAlgorithm:
    '''遗传算法处理TSP
    参考B站视频：
    2:18 [https://www.bilibili.com/video/BV17Z4y1w7qF?from=search&seid=5545025040067515449]
    https://www.bilibili.com/video/BV1zp4y1U7Ti?from=search&seid=5545025040067515449
    参考书籍：遗传算法原理及应用
    http://img.sslibrary.com/n/slib/book/slib/10190716/050b23598b994e0198631fe9d0e06420/57547a3e911456a493f0d9d87d4c4c15.shtml?dxbaoku=true&deptid=1173&fav=http%3A%2F%2Fwww.sslibrary.com%2Freader%2Fpdg%2Fdxpdgreader%3Fd%3Dd4196b013af989b9b62e0fbd41a97781%26enc%3D073c9c701bdb2fa8c4c5ee87c6189605%26ssid%3D10190716%26did%3D1173%26username%3Ddx2baoku_157.0.78.102_1173&fenlei=18170206&spage=1&t=5&username=dx2baoku_157.0.78.102_1173&view=-1
    （18：57）
    '''

    def __init__(self, city_num, distance_matrix):
        '''
        :param city_num:城市数量，城市编号[0, 1, ..., city_num-1]；
        :param distance_matrix:城市间的距离矩阵；
        '''
        self.city_num = city_num
        self.distance_matrix = distance_matrix

        if self.city_num <= 10:
            self.T = 500 # T遗传算法的中止进化代数
            self.M = 60 # 集群大小
            self.Pc = 0.9 # 交叉概率 0.8
            self.Pm = 0.9 # 变异概率 0.05
        else:
            self.T = 500000  # T遗传算法的中止进化代数
            self.M = 1500  # 集群大小
            # self.M = 1200  # 集群大小
            self.Pc = 0.9  # 交叉概率 0.8
            self.Pm = 0.9  # 变异概率 0.05

        self.crossover_name = 'PMX' # 使用的交叉算子名称
        self.mutation_name = 'exchange' # 使用的变异算子名称

    def run(self, start_city):
        '''
        从start_city开始旅行(在遗传算法中无用，只是为了统一调用不同算法的run函数)
        :param start_city:起始城市编号，属于{0, 1, ..., self.city_num-1}；
        :return:返回TSP依次访问过的城市编号，距离；
        '''
        # 1. 随机生成初始种群
        population = []
        cities_number = np.arange(self.city_num)
        for i in range(self.M):
            np.random.shuffle(cities_number)
            population.append(cities_number.copy())

        n = 0
        while n < self.T:
            # 2. 计算适应度
            fitness_s = np.asarray([self.calculate_fitness(population[i]) for i in range(len(population))])
            # 3. 选择操作
            select_times = self.selection(fitness_s)
            population_new = []
            for i in range(len(select_times)):
                for j in range(int(select_times[i])):
                    population_new.append(population[i].copy())
            # 4. 交叉操作
            index = np.arange(self.M)
            np.random.shuffle(index)  # 两两随机配对
            index = list(index)
            while (len(index) > 1):
                Tx_index = index.pop()
                Ty_index = index.pop()

                prob_crossover = np.random.rand()
                if prob_crossover > self.Pc:
                    # 以一定的概率进行交叉
                    Tx_new, Ty_new = self.crossover(population_new[Tx_index].copy(), population_new[Ty_index].copy())
                    population_new.append(Tx_new)
                    population_new.append(Ty_new)
                    # population_new[Tx_index], population_new[Ty_index] = Tx_new, Ty_new
            # 5. 变异操作
            for i in range(self.M):
                prob_mutaion = np.random.rand()
                if prob_mutaion > self.Pm:
                    # 以一定的概率进行变异
                    Individual_new = self.mutation(population_new[i].copy())
                    # population_new[i] = Individual_new
                    population_new.append(Individual_new)
            # 6. 子代父代结合生成新种群
            population = population_new.copy()

            # 补充：计算最优值输出
            best_Individual, best_distance = self.get_best_Individual(population)
            # print("iter:{}/{}; best_distance:{}; best_Individual:{}".format(n, self.T, best_distance, best_Individual))
            with open("./logs/{}_GeneticAlgorithm_process_log.txt".format(self.city_num), mode='a', encoding='utf-8') as wf:
                # wf.write("iter:{}/{}; best_distance:{}; best_Individual:{}\n".format(n, self.T, best_distance, best_Individual))
                wf.write("iter:{}/{}; best_distance:{};\n".format(n, self.T, best_distance))

            n += 1
            # if n%100 == 0:
            #     new_pc = self.Pc * 1.05
            #     if new_pc < 1:
            #         self.Pc = new_pc
            #     new_pm = self.Pm * 1.05
            #     if new_pm < 1:
            #         self.Pm = new_pm
            #     print("n:{}, Pc:{}, Pm:{}".format(n, self.Pc, self.Pm))

        return best_Individual, best_distance

    def calculate_fitness(self, Individual):
        '''
        计算个体Individual的适应度：F(T) = n/length(T)，其中length(T)表示巡回路线T的路线长度，n是城市数量；
        :param Individual:个体，这里指巡回路线；
        :return:个体适应度；
        '''
        dis = np.asarray([self.distance_matrix[Individual[i]][Individual[i+1]] for i in range(len(Individual)-1)])
        dis = np.sum(dis)
        dis += self.distance_matrix[Individual[-1]][Individual[0]]
        # return self.city_num/dis
        return 1/dis

    def selection(self, fitness_s):
        '''
        选择算子（赌盘选择、比例选择算子）
        参考博客：https://www.cnblogs.com/adelaide/articles/5679475.html
        :param fitness_s:所有个体的适应度；
        :return:每个个体的选择次数；
        '''
        # 计算每个个体被遗传到下一个种群的概率
        p = fitness_s/np.sum(fitness_s)
        # 计算每个个体的累积概率
        q = [np.sum(p[:i+1]) for i in range(len(fitness_s))]

        select_times = np.zeros(len(fitness_s))
        for i in range(self.M):
            # 在[0,1)区间内产生一个均匀分布的随机数r
            r = np.random.rand()
            # 若 r < q[1],则选择个体1，否则，选择个体k，使得：q[k-1]<r≤q[k] 成立
            for k in range(len(fitness_s)):
                if q[k] >= r:
                    break
            select_times[k] += 1
        return select_times

    def crossover(self, Tx, Ty):
        '''
        交叉算子
        :param Tx:巡回路线；
        :param Ty:巡回路线；
        :return:交叉后的两条新巡回路线Tx, Ty
        '''
        if self.crossover_name == 'PMX':
            return self.crossover_PMX(Tx, Ty)
        else:
            print("Error: 没有对应的交叉算子！")

    def crossover_PMX(self, Tx, Ty):
        '''
        部分匹配交叉（Partially Matched Crossover）
        :param Tx: 巡回路线；
        :param Ty: 巡回路线；
        :return:交叉后的两条新巡回路线Tx, Ty
        '''
        # 随机选取两个基因座i, j, 交叉区域：[i+1, j];
        i, j = 0, 0
        while i == j:
            i, j = np.random.randint(low=0, high=self.city_num, size=2)

        if i > j: i, j = j, i

        # 对交叉区域的每个基因座p，找到对应基因并交换【参考：《遗传算法原理及应用》P147】
        for p in range(i+1, j+1):
            tx_p, ty_p = Tx[p], Ty[p]
            q = np.argwhere(Tx == ty_p)[0][0]
            r = np.argwhere(Ty == tx_p)[0][0]
            Tx[p], Tx[q] = Tx[q], Tx[p]
            Ty[p], Ty[r] = Ty[r], Ty[p]
        return Tx, Ty

    def mutation(self, Individual):
        '''
        变异算子
        :param Individual:个体，这里指巡回路线；
        :return:变异后的新个体；
        '''
        if self.mutation_name == 'exchange':
            return self.mutation_exchange(Individual)
        else:
            print("Error: 没有对应的变异算子！")

    def mutation_exchange(self, Individual):
        '''
        交换变异
        :param Individual:个体，这里指巡回路线；
        :return:交换变异后的新个体；
        '''
        i, j = 0, 0
        while i == j:
            i, j = np.random.randint(low=0, high=self.city_num, size=2)

        Individual[i], Individual[j] = Individual[j], Individual[i]
        return Individual

    def get_best_Individual(self, population):
        '''
        从种群population中挑出最好的一个个体返回
        :param population:种群；
        :return:最好的个体（回路距离最短的个体），距离；
        '''
        individual_dis = []
        for Individual in population:
            dis = np.asarray([self.distance_matrix[Individual[i]][Individual[i+1]] for i in range(len(Individual)-1)])
            dis = np.sum(dis)
            dis += self.distance_matrix[Individual[-1]][Individual[0]]
            individual_dis.append(dis)

        best_index = np.argmin(individual_dis)
        return population[best_index], individual_dis[best_index]