"""
选择：
用轮盘赌法从原种群中选出一些个体组成新种群
保证新种群和原种群的个体数相等
"""
import random
import copy

def select(pop):
    #pop : 原种群
    #newpop : 新种群
    #p : 每个染色体被选择的概率
    #sump : 赌盘，存放每个染色体的累积选择概率

    sump = []
    fit_list = [ind.fit() for ind in pop]
    sumfit = sum(fit_list)
    sump.append(0)

    #构造赌盘
    s = 0  #累加变量
    for i in range(0,len(pop)):
        p = fit_list[i]/sumfit
        s += p
        sump.append(s)
    #选出新种群
    newpop = []
    for j in range(len(pop)):
        rd = random.uniform(0,1)
        for k in range(len(sump)-1):
            if sump[k] <= rd and rd < sump[k+1]:
                newpop.append(copy.deepcopy(pop[k]))
    return copy.deepcopy(newpop)
