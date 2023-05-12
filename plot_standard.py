"""
做出标准的函数图像，方便与测试结果对比
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
from math import pi

def plot_standardimage():
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-2 * pi, 2 * pi, 2000)
    y = np.linspace(-2 * pi, 2 * pi, 2000)
    # 产生隔点矩阵
    x, y = np.meshgrid(x, y)
    z = 2 * x * x + np.sin(y + pi / 4)
    ax.plot_surface(x, y, z)
    plt.title("The standard function image\nTurn off automatically after 3 seconds")
    plt.pause(3)  # 显示5秒，5秒后自动关闭并继续运行
    plt.close()

if __name__=="__main__":
    plot_standardimage()
