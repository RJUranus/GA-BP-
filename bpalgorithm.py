"""
用BP算法（即梯度下降法）对神经网络进行训练
当总误差小于设定值或者训练代数大于设定值代时停止训练
单独运行该程序，则可以测试BP算法训练神经网络的效果；
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import math
from math import pi
from plot_standard import plot_standardimage
from plot_fitting import plot_fittingimage


def sg(x):  # sigmoid函数定义
    return 1 / (1 + np.exp(-x))


# 用BP算法训练网络的函数
def BP(input_train, out_train, w_hide, bias_hide, w_out, bias_out, hiddennum, trainnum):
    # 重要变量和参数说明：
    # M : 训练轮次计数器
    # Eaverage : 平均绝对误差
    # step ：学习率
    # iternum_max : 最大迭代次数
    # Eaverage_min : 最小可接受误差za
    step = 0.5
    iternum_max = 300
    Eaverage_min = 0.01
    M = 0
    Eaverage = 1
    hidout = np.zeros(hiddennum)
    y_delta = np.zeros(hiddennum)
    M_iteration=[]
    Eaverage_iteration = []
    while (M < iternum_max and Eaverage > Eaverage_min):
        Eall = 0

        for i in range(trainnum):
            out = 0
            for j in range(hiddennum):
                hidout[j] = sg(w_hide[j, 0] * input_train[i, 0] + w_hide[j, 1] * input_train[i, 1] - bias_hide[j])
                out += hidout[j] * w_out[j]
            out = sg(out - bias_out)
            E = abs(out_train[i] - out)
            Eall += E  # 损失函数
            delta = (out_train[i] - out) * out * (1 - out)
            for m in range(hiddennum):
                y_delta[m] = delta * w_out[m] * hidout[m] * (1 - hidout[m])
            bias_out -= step * delta
            for m in range(hiddennum):
                w_out[m] += step * delta * hidout[m]
                bias_hide[m] -= step * y_delta[m]
                w_hide[m, 0] += step * y_delta[m] * input_train[i, 0]
                w_hide[m, 1] += step * y_delta[m] * input_train[i, 1]
        Eaverage = Eall / (trainnum)
        M += 1
        print("BP算法第", M, "次训练，归一化平均绝对误差为：", Eaverage)
        M_iteration.append(M)
        Eaverage_iteration.append(Eaverage)
    # 平均绝对误差随迭代次数变化情况的散点图
    M_iteration = np.array(M_iteration)
    Eaverage_iteration = np.array(Eaverage_iteration)
    plt.scatter(M_iteration,Eaverage_iteration,s=0.2)
    plt.show()
    return w_hide, bias_hide, w_out, bias_out


# 测试单独使用bp算法的效果
if __name__ == "__main__":

    num = 1000  # 数据总数
    trainnum = int(0.9 * num)
    testnum = int(0.1 * num)  # 其中十分之一用测试

    # 用函数f(x)=2*x*x+sin(y+pi/4)生成数据集，其中自变量x，y的取值范围是(-2π，2π)
    input_data = np.random.uniform(low=-2 * pi, high=2 * pi, size=[num, 2])  # 第一列作为x，第二列作为y
    output_data = 2 * np.multiply(input_data[:, 0], input_data[:, 0]) + np.sin(input_data[:, 1] + pi / 4)

    # 归一化处理,将输入的范围调整到-1到1之间,输出范围调整到0到1之间
    output_max = np.max(output_data)
    output_min = np.min(output_data)
    input_datan = input_data / (2 * pi)
    output_datan = (output_data - output_min) / (output_max - output_min)

    # 9:1的比例划分训练集和测试集
    input_train = input_datan[0:int(0.9 * num), :]
    input_test = input_datan[int(0.9 * num):num, :]
    out_train = output_datan[0:int(0.9 * num)]
    out_test = output_datan[int(0.9 * num):num]
    out_testun = output_data[int(0.9 * num):num]  # 未归一化的测试输出
    inputnum, outputnum = 2, 1  # 输入神经元，输出神经元个数

    # 根据经验选择隐层节点数
    hiddennum = 20
    # 网络初始化
    w_hide = 2 * np.c_[np.random.random(hiddennum), np.random.random(hiddennum)] - 1  # 第i行j列表示第i个节点对第j个输入的权值
    bias_hide = 2 * np.random.random(hiddennum) - 1
    w_out = 2 * np.random.random(hiddennum) - 1
    bias_out = random.uniform(-1, 1)
    hidout = np.zeros(hiddennum)
    y_delta = np.zeros(hiddennum)
    # 得到训练后的神经网络
    w_hide, bias_hide, w_out, bias_out = BP(input_train, out_train, w_hide, bias_hide, w_out, bias_out, hiddennum,
                                            trainnum)

    # 计算测试集的误差
    E = 0
    outun = np.zeros(testnum)
    for i in range(testnum):
        out = 0
        for j in range(hiddennum):
            hidout[j] = sg(w_hide[j, 0] * input_test[i, 0] + w_hide[j, 1] * input_test[i, 1] - bias_hide[j])
            out += hidout[j] * w_out[j]
        out = sg(out - bias_out)
        outun[i] = (output_max - output_min) * out + output_min  # 反归一化
        E += abs(out_test[i] - out)
    Etest_average = E / testnum
    print("\n用测试集测试")
    print("归一化平均绝对误差为：", Etest_average)

    # 作散点图
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.scatter3D(2 * pi * input_test[:, 0], 2 * pi * input_test[:, 1], outun, 'binary')
    plt.title("The test result of BP" + "\nTurn off automatically after 5 seconds")
    plt.pause(5)  # 显示5秒，5秒后自动关闭并继续运行
    plt.close()
    print("\n画出拟合得到的函数图像，请稍候……")
    plot_fittingimage(w_hide.transpose(), bias_hide, w_out.transpose(), bias_out,output_max,output_min,"BP")
