"""
变异：
种群的每个个体基因变异的概率为pm
根据当前迭代次数确定基因变异的位数:随着迭代次数增大，变异位数减小
"""
import math
import random
import copy

def mutate(pop,pm,amax,amin,G,Gmax):
    #pop : 原种群
    #newpop : 新种群
    #pm : 变异概率
    #amax : 基因变异的上界值
    #amin : 基因变异的下界值
    #G : 当前迭代次数
    #Gmax : 最大迭代次数
    #t ： 变异位数
    Lmax = 10
    t = math.ceil(Lmax*(1-G/Gmax))    #确定变异位数
    for i in range(len(pop)):
        ind = pop[i]
        rd = random.uniform(0,1)
        if rd < pm:    # 变异
            positions = random.sample(range(0,ind.L),t)    #随机选出变异的位置
            for position in positions:
                r = random.uniform(0,1)           #r控制增大或者减小的步长
                if random.uniform(0,1) > 0.5:     #基因值以相等的概率增大或减小
                    ind.chrom[position] = ind.chrom[position] + r*(amax - ind.chrom[position])*(1-G/Gmax)
                else:
                    ind.chrom[position] = ind.chrom[position] + r * (amin - ind.chrom[position] ) * (1 - G / Gmax)
        pop[i] = ind
