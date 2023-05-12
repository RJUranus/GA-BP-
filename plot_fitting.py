"""
做出拟合得到的函数图像
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from math import pi

def plot_fittingimage(w_hide , bias_hide , w_out , bias_out, output_max, output_min, str):
    N = 500 #点的密度
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    # 产生隔点矩阵
    x, y = np.meshgrid(x, y)
    z = np.zeros((N,N),np.float)
    for i in range(N):
        for j in range(N):
            input = np.array([x[i][j],y[i][j]])
            hide_in = np.matmul(input, w_hide)
            hide_out = (1 / (1 + (np.exp(-hide_in + bias_hide))))
            out_in = np.matmul(hide_out, w_out)
            z[i][j] = 1 / (1 + (np.exp(-out_in + bias_out)))
            z[i][j] = (output_max - output_min)*z[i][j] + output_min   #反归一化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(2 * pi*x, 2 * pi*y, z)    #x,y需要反归一化到（-2π，2π）之间
    plt.title("The fitting function image by "+str)
    plt.show()
