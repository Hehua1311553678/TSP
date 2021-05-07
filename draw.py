# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 22:19
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : draw.py
# @Software: PyCharm
import re
import matplotlib.pyplot as plt
import numpy as np

def draw_GeneticAlgorithm_curve(path='./logs/10_GeneticAlgorithm_process_log.txt'):
    with open(path, mode='r', encoding='utf-8') as rf:
        lines = rf.readlines()
    # pattern = re.compile(".best_distance:(\d*\.\d*); best_Individual.*?")
    pattern = re.compile(".best_distance:(\d*\.\d*);")
    results = [float(re.findall(pattern, line)[0]) for line in lines]

    x = np.arange(len(results)) + 1
    plt.plot(x, results, c='b')
    plt.xlabel('iters')
    plt.ylabel('best_value')
    plt.title('TSP with Genetic Algorithm')
    plt.show()

def get_coordinate(path='TSPdatasets/10cities.tsp'):
    # 获取城市坐标点
    with open(path, mode='r', encoding='utf-8') as rf:
        lines = rf.readlines()
    city_dict = dict()
    for line in lines:
        index, x, y = line.strip().split(' ')
        city_dict[str(int(index) - 1)] = np.asarray([int(x), int(y)])
    return city_dict


# def draw_TSP(path='./logs/10_GeneticAlgorithm_process_log.txt', dict_coordinates=None):
#     with open(path, mode='r', encoding='utf-8') as rf:
#         lines = rf.readlines()
#     lines = lines[1:]
#     cities = []
#     for line in lines:
#         if '[' in line:
#             cities_i = line.strip('依次访问过的城市编号：[').strip(']').strip().split(' ')
#         elif ']' in line:
#             if '.' in line: cities_i = line.strip(']').strip().split('. ')
#             else: cities_i = line.strip(']').strip().split(' ')
#         elif '.' in line:
#             cities_i = line.strip().split('. ')
#         cities += cities_i

def get_tour(title = 'Greedy', tour_str='0. 9. 8. 5. 3. 2. 1. 7. 4. 6.'):
    # str->list
    if title == "Greedy":
        tour_list = tour_str.split('. ')
    else:
        tour_list = tour_str.split()
    return tour_list

def draw_TSP(title='Greedy', tour=None, dict_coordinates=None):
    # 1. 画城市坐标点
    for index in dict_coordinates.keys():
        coord = dict_coordinates[index]
        plt.scatter(coord[0], coord[1], s=10, c='b')

    # 画边
    for i in range(len(tour) - 1):
        start = dict_coordinates[tour[i].strip()]
        end = dict_coordinates[tour[i+1].strip()]
        plt.plot([start[0], end[0]], [start[1], end[1]], c='b')
    start = dict_coordinates[tour[0].strip()]
    end = dict_coordinates[tour[-1].strip()]
    plt.plot([start[0], end[0]], [start[1], end[1]], c='b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.plot()
    plt.show()

if __name__=='__main__':
    # 画收敛曲线
    # path = './logs/10_GeneticAlgorithm_process_log.txt'
    path = './logs/200_GeneticAlgorithm_process_log.txt'
    draw_GeneticAlgorithm_curve(path)

    # 画TSP tour
    # path='TSPdatasets/10cities.tsp'
    # path='TSPdatasets/200cities.tsp'

    # title = 'Greedy'
    # tour_str = '0. 9. 8. 5. 3. 2. 1. 7. 4. 6'

    # title = 'DynamicProgramming'
    # tour_str = '0 4 6 7 1 3 2 5 8 9'

    # title = 'BranchAndBound'
    # tour_str = '0 4 6 7 1 3 2 5 8 9'

    # title = 'GeneticAlgorithm'
    # tour_str = '6 4 0 9 8 5 2 3 1 7'

 #    title = 'Greedy'
 #    tour_str = '0.  52. 114. 116. 110. 131.  84. 144. 190. 197.  26. 122.  14.  12.\
 #  78. 159. 161.  63.  19.  54.  41. 134. 185. 126. 111. 119.  46.  30.\
 #  66. 176.  64.  79. 160. 124. 180.   1.  34. 168.  67.  29.  88.  40.\
 #  58.   2.  72. 188.  68. 141. 130. 179. 155.  99.  32.  44. 196.  80.\
 #  96. 103. 164. 165.  95. 125.  86.  51.  10.  83.  47. 169. 121. 115.\
 # 187.  43.  62.  15. 117. 123. 137.   8.  77.  81. 198.  25. 135.  60.\
 #  31.  23. 158. 173. 120.  45. 171.  28. 109.  17.  48. 189. 148. 105.\
 #  92. 162.   3. 100.  59. 127. 192. 157.  76. 150. 186.   5. 108. 106.\
 # 156.  53.  74. 182. 154.   7.  21. 133. 128. 145. 102. 142.  89.  33.\
 #  24.  16. 113.  97.  87. 147.  27.  38.  37.  70. 129.  71.  82.  61.\
 # 184. 167. 172.  22. 143.  69.  75.  90. 149.  93.  94.  49. 138.  85.\
 #   4. 104.  42. 136. 177. 151.  55. 195. 199. 170.  57. 140. 132. 175.\
 # 112. 194. 181. 101.  20. 139. 163. 153. 166. 107. 191.  13.  35.  56.\
 #  73. 174.   9.  91.  98.  18. 118.  65. 152. 178.  50. 193.  36. 183.\
 #  11. 146.  39.   6'

 #    title = 'GeneticAlgorithm'
 #    tour_str = '139 153 99 155 32 56 44 196 80 96 91 174 9 118 152  65  43  62\
 #  15 117  50 115 187 178 189 105   3 192 127  69 143 172 149 132 175 112\
 # 194 181  75  93  90  22 163  88  58 188 179 130  68 107 191 162 109  48\
 #  28 183  45 171 137  31  25 198   6  81  77   8  60 135  23 158 173 120\
 # 110  52 116   0 122 197 144  84 190  14  12  64  79 157  76 160 180  34\
 # 167 138 195  70 104  27  87 113 140 170 199  57  97  37  55   4 136  42\
 # 177 151 147  38 129  82 102 142 145  16  24  33  89 128   7  21 133 186\
 # 108  46  30 150   1 124  29  40 166  59 100 148  18  98  92  17 123  36\
 # 193 121 169  10  51 164  95  86 125 165  83  47 103  35  73  13 141  72\
 #   2 168  67  53  74 182 154 111 134 126  54  41 185  19  63  66 176 161\
 #  26 146  11  39 114 131 159  78 119 156   5 106 184  71  85  61  49  94\
 # 101  20'
 #
 #    city_dict = get_coordinate(path)
 #    tour_list = get_tour(title, tour_str)
 #    draw_TSP(title, tour_list, city_dict)

