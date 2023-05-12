"""
交叉：按照概率Pc从selcet函数选出的种群中选择染色体进行交配
具体过程为：
（1）从区间[1，N]中随机产生两个整数a,b
（2）对父代个体sa和sb进行算术杂交，得到两个子代个体
（3）重复上述过程，直到得到N个子个体
"""
import random
import numpy as np
import copy

def cross(pop,pc):
    #pop : 原种群
    #newpop : 交叉后产生的新种群
    #pc : 交叉概率
    #t ： 交叉次数
    t = 0
    newpop = copy.deepcopy(pop)  #初始化
    while t <= len(pop):
        rd = random.uniform(0,1)
        if rd>pc:   #不交叉
            t += 1
        else:       #交叉
            flag = 0
            while flag == 0:
                a = random.randint(0,len(pop)-1)
                b = random.randint(a,len(pop)-1)
                # print(pop[a].chrom,pop[b].chrom)
                if any(pop[a].chrom != pop[b].chrom):  #只有父代的染色体不同时，交叉才有意义
                    crosspoint = random.sample(range(0,pop[0].L),2)    #交叉点
                    startpoint = min(crosspoint)     #交叉起点
                    endpoint = max(crosspoint)      #交叉终点
                    F = random.uniform(0,1)
                    newpop[b].chrom[startpoint:endpoint] = copy.deepcopy(F*pop[a].chrom[startpoint:endpoint]) + copy.deepcopy((1-F)*pop[b].chrom[startpoint:endpoint])
                    newpop[a].chrom[startpoint:endpoint] = copy.deepcopy(F*pop[b].chrom[startpoint:endpoint]) + copy.deepcopy((1-F)*pop[a].chrom[startpoint:endpoint])
                    flag = 1
            t += 1
    return copy.deepcopy(newpop)
