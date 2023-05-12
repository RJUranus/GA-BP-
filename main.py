"""
主程序部分，用给定函数产生数据集，并用GABP算法对神经网络的权值进行优化，即：
先用遗传算法对神经网络的权值进行优化，
得到的最优个体作为BP算法的初始解，
再利用BP算法对神经网络进一步优化。
注：为了避免一次性输出太多图像，部分代码改为注释，将代码的注释去掉，可以正常输出相应图像
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import math
import random

from selection import select
from crossover import cross
from mutation import mutate
import copy
from bpalgorithm import BP
from plot_standard import plot_standardimage
from plot_fitting import plot_fittingimage

##超参数
popsize =100  #种群规模
Gmax = 1000    #最大迭代次数
pc=0.8        #交叉概率
pm=0.1        #变异概率
amax = 15     #染色体基因值的上界
amin = -15    #基因值下界

inputnum , outputnum = 2, 1  # 输入神经元，输出神经元个数

#根据经验选择隐层节点数
hiddennum =20
print(hiddennum)

num = 1000  # 数据总数
trainnum =int(0.9*num)  #其中十分之九用于训练
testnum =int(0.1*num)  #其中十分之一用于测试

# 用函数f(x)=2*x*x+sin(y+pi/4)生成数据集，其中自变量x，y的取值范围是(-2π，2π)
input_data = np.random.uniform(low=-2 * pi, high=2 * pi, size=[num, 2])  # 第一列作为x，第二列作为y
output_data = 2 * np.multiply(input_data[:, 0], input_data[:, 0]) + np.sin(input_data[:, 1] + pi / 4)

# 归一化处理,将输入的范围调整到-1到1之间,输出范围调整到0到1之间
output_max = np.max(output_data)
output_min = np.min(output_data)

input_datan = input_data / (2 * pi)
output_datan = (output_data - output_min) / (output_max - output_min)

# 9:1的比例划分训练集和测试集
input_train = input_datan[0:int(0.9 * num) , :]
input_test = input_datan[int(0.9 * num):num , :]
out_train = output_datan[0:int(0.9 * num) ]
out_test = output_data[int(0.9 * num):num ]    #测试输出不需要归一化

#对染色体进行解码得到权值和阈值
def decode(chrom):
    # 输入层到隐层的权值
    w_hide = chrom[:inputnum * hiddennum].reshape(inputnum, hiddennum)
    # 隐层神经元的阈值
    bias_hide = chrom[inputnum * hiddennum: inputnum * hiddennum + hiddennum]
    # 隐层到输出层的权值
    w_out = chrom[inputnum * hiddennum + hiddennum: inputnum * hiddennum + hiddennum + hiddennum * outputnum] \
        .reshape(hiddennum, outputnum)
    # 输出层的权值
    bias_out = chrom[inputnum * hiddennum + hiddennum + hiddennum * outputnum:]
    return w_hide , bias_hide , w_out , bias_out

# 定义个体类，可以调用方法直接计算个体的适应度
class indivdual:
    def __init__(self):
        self.L = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum #编码长度
        self.chrom = np.zeros(self.L, np.float)  # 染色体初始化
        self.fitness = 0  # 适应度初始化

    #适应度计算
    def fit(self):
        w_hide, bias_hide, w_out, bias_out =decode(self.chrom)
        hide_in = np.matmul(input_train , w_hide)
        hide_out = 1 / (1 + np.exp(-(hide_in - bias_hide)))
        out_in = np.matmul(hide_out, w_out)
        y = 1 / (1 + np.exp(-(out_in - bias_out)))  #网络实际输出
        y = y.reshape(1,-1)   #列向量转成行向量
        cost = np.abs(y - out_train)
        sumcost = np.sum(cost)     #损失函数
        fitness = 1/sumcost        #取损失函数的倒数作为适应度`
        return fitness

# 初始化种群
def initPopulation(pop , popsize):
    #pop : 种群
    #popsize : 种群大小

    for i in range(popsize):
        ind = indivdual()
        ind.chrom = copy.deepcopy(np.random.uniform(low=amin, high=amax, size=ind.L))  #对每个个体的权值和阈值进行初始化
        #ind.chrom = np.ones(ind.L)*i
        pop.append(ind)

# 寻找种群中的最优个体
def findbest(pop):
    fit_list = [ind.fit() for ind in pop]
    bestindex = fit_list.index(max(fit_list))
    return pop[bestindex]

#以散点图的形式画出神经网络的预测结果，同时计算平均绝对误差
def picture(w_hide , bias_hide , w_out , bias_out , str):
    #MAD : 平均绝对误差
    #output : 神经网络得到的输出
    #outputun : 反归一化后的输出
    #str : 区别GA和BPGA的字符串
    output = []
    for m in range(testnum):
        hide_in=np.matmul(input_test[m], w_hide)
        hide_out=(1/(1+(np.exp(-hide_in+bias_hide))))
        out_in=np.matmul(hide_out, w_out)
        c= 1/(1+(np.exp(-out_in+bias_out)))
        output.append(c[0])
    #计算误差
    outputun = (output_max - output_min)*np.array(output) + output_min  #反归一化
    MAD = np.sum(abs(outputun-out_test))/testnum
    print("\n"+str+"算法测试的平均绝对误差为：",MAD)
    if str == "GA":
        plot_standardimage()   #做出标准函数图像
    #作散点图
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.scatter3D(2*pi*input_test[:, 0], 2*pi*input_test[:, 1], outputun, 'binary')
    plt.title("The test result of " + str+"\nTurn off automatically after 5 seconds")
    plt.pause(5)  # 显示5秒，5秒后自动关闭并继续运行
    plt.close()

#主程序
pop = []
initPopulation(pop,popsize)

#精英策略，保留全局最优个体
ind_best_global = findbest(pop)
best_fit_iteration = []
best_fit_iteration.append(ind_best_global.fit())

for G in range(1,Gmax+1):
    print("--------------第"+str(G)+"次迭代--------------")
    pop = select(pop)
    pop = cross(pop,pc)
    mutate(pop,pm,amax,amin,G,Gmax)
    ind_best_now = findbest(pop)
    if ind_best_now.fit() > ind_best_global.fit():
        ind_best_global = copy.deepcopy(ind_best_now)
    print("当前最优适应度：",ind_best_now.fit())
    print("全局最优适应度：",ind_best_global.fit())
    best_fit_iteration.append(ind_best_global.fit())

w_hide , bias_hide , w_out , bias_out =decode(ind_best_global.chrom)

picture(w_hide , bias_hide , w_out , bias_out,"GA")

print("BP算法迭代中……")
#BP函数中权值数组的形状与主程序定义的数组形状互为转置
# 因此传入参数前需要将两个权值数组转置

w_hide_bp, bias_hide_bp, w_out_bp, bias_out_bp = BP(input_train, out_train, w_hide.transpose(), bias_hide, w_out.transpose()[0], bias_out, hiddennum, trainnum)

picture(w_hide_bp.transpose(), bias_hide_bp, w_out_bp.transpose(), bias_out_bp,"GABP")

print("\n画出拟合得到的函数图像，请稍候……")
plot_fittingimage(w_hide_bp.transpose(), bias_hide_bp, w_out_bp.transpose(), bias_out_bp,output_max,output_min,"GABP")

