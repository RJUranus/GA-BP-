import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

# 定义激活函数sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义BP神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.b1 = np.zeros(hidden_size)  # 隐藏层的偏置
        self.W2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重
        self.b2 = np.zeros(output_size)  # 输出层的偏置

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1  # 隐藏层的加权输入
        self.a1 = sigmoid(self.z1)  # 隐藏层的输出
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 输出层的加权输入
        self.a2 = sigmoid(self.z2)  # 输出层的输出

    def backward(self, X, y, learning_rate):
        # 反向传播
        m = X.shape[0]  # 样本数量

        # 计算输出层的误差
        delta2 = (self.a2 - y) * self.a2 * (1 - self.a2)

        # 计算隐藏层的误差
        delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1)

        # 更新权重和偏置
        self.W2 -= learning_rate * np.dot(self.a1.T, delta2) / m
        self.b2 -= learning_rate * np.mean(delta2, axis=0)
        self.W1 -= learning_rate * np.dot(X.T, delta1) / m
        self.b1 -= learning_rate * np.mean(delta1, axis=0)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        self.forward(X)
        return self.a2

# 生成数据集
num = 1000
train_num = 900
test_num = 100
input_data = np.random.uniform(low=-2 * pi, high=2 * pi, size=(num, 2))
output_data = 2 * input_data[:, 0]**2 + np.sin(input_data[:, 1] + pi / 4)

# 归一化处理
input_data_normalized = input_data / (2 * pi)
output_data_normalized = (output_data - np.min(output_data)) / (np.max(output_data) - np.min(output_data))

# 划分训练集和测试集
input_train = input_data_normalized[:train_num]
output_train = output_data_normalized[:train_num].reshape(-1, 1)
input_test = input_data_normalized[train_num:]
output_test = output_data_normalized[train_num:]

# 创建BP神经网络实例
neural_network = NeuralNetwork(input_size=2, hidden_size=20, output_size=1)

# 训练神经网络
epochs = 30000
learning_rate = 20
neural_network.train(input_train, output_train, epochs, learning_rate)

# 在测试集上进行预测
predictions = neural_network.predict(input_test)

# 反归一化处理
output_test_denormalized = output_test * (np.max(output_data) - np.min(output_data)) + np.min(output_data)
predictions_denormalized = predictions * (np.max(output_data) - np.min(output_data)) + np.min(output_data)

# 计算平均绝对误差
mae = np.mean(np.abs(output_test_denormalized - predictions_denormalized))
print("平均绝对误差：", mae)

# 绘制预测结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(input_test[:, 0], input_test[:, 1], output_test_denormalized, c='b', label='真实值')
ax.scatter(input_test[:, 0], input_test[:, 1], predictions_denormalized, c='r', label='预测值')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.legend()
plt.show()
