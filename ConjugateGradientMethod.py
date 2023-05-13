"""
在BP神经网络中应用非线性共轭梯度法求解
"""
import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
# 定义函数 f(x) = 2*x*x + sin(y + pi/4)
from sklearn.model_selection import train_test_split


def target_function(x, y):
    return 2 * x * x + np.sin(y + np.pi / 4)

# 生成数据集
def generate_dataset(n_samples):
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(n_samples, 2))
    y = target_function(x[:, 0], x[:, 1])
    y = np.reshape(y, (n_samples, 1))  # 将输出标签调整为 (n_samples, 1) 的形状
    return x, y

# 定义激活函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = 0.5

        # 初始化权重
        self.weights1 = np.random.uniform(-0.5, 0.5, size=(n_input, n_hidden))
        self.weights2 = np.random.uniform(-0.5, 0.5, size=(n_hidden, n_output))

    def forward_propagation(self, X):
        # 前向传播计算输出
        self.hidden_layer = sigmoid(np.dot(X, self.weights1))
        self.output_layer = np.dot(self.hidden_layer, self.weights2)
        return self.output_layer

    def calculate_loss(self, X, y):
        # 计算损失函数值
        predicted_output = self.forward_propagation(X)
        loss = np.mean((predicted_output - y) ** 2)
        return loss

    def gradient(self, X, y):
        # 计算权重的梯度
        n_samples = X.shape[0]

        # 前向传播
        self.forward_propagation(X)

        # 反向传播计算梯度
        delta_output = 2 * (self.output_layer - y) / n_samples
        delta_hidden = np.dot(delta_output, self.weights2.T) * sigmoid_derivative(self.hidden_layer)

        grad_weights2 = np.dot(self.hidden_layer.T, delta_output)
        grad_weights1 = np.dot(X.T, delta_hidden)

        return grad_weights1, grad_weights2

    def train(self, X, y):
        def cost_function(params):
            # 将参数转化为权重矩阵
            self.weights1 = np.reshape(params[:self.n_input * self.n_hidden], (self.n_input, self.n_hidden))
            self.weights2 = np.reshape(params[self.n_input * self.n_hidden:], (self.n_hidden, self.n_output))
            # 计算损失函数值
            loss = self.calculate_loss(X, y)
            return loss

        def gradient_function(params):
            # 将参数转化为权重矩阵
            self.weights1 = np.reshape(params[:self.n_input * self.n_hidden], (self.n_input, self.n_hidden))
            self.weights2 = np.reshape(params[self.n_input * self.n_hidden:], (self.n_hidden, self.n_output))
            # 计算权重的梯度
            grad_weights1, grad_weights2 = self.gradient(X, y)
            # 更新权重矩阵
            self.weights1 -= self.learning_rate * grad_weights1
            self.weights2 -= self.learning_rate * grad_weights2
            # 将梯度展开为一维向量
            grad = np.concatenate((grad_weights1.ravel(), grad_weights2.ravel()))
            return grad

        # 初始化权重参数
        initial_params = np.concatenate((self.weights1.ravel(), self.weights2.ravel()))

        # 使用非线性共轭梯度法进行训练
        optimized_params = fmin_cg(cost_function, initial_params, fprime=gradient_function, disp=False)

        # 将优化后的参数重新转化为权重矩阵
        self.weights1 = np.reshape(optimized_params[:self.n_input * self.n_hidden], (self.n_input, self.n_hidden))
        self.weights2 = np.reshape(optimized_params[self.n_input * self.n_hidden:], (self.n_hidden, self.n_output))

        # 返回训练后的模型参数
        return self.weights1, self.weights2

    def predict(self, X):
        # 使用训练好的模型进行预测
        predicted_output = self.forward_propagation(X)
        return predicted_output

# 生成数据集
X, y = generate_dataset(1000)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 创建BP神经网络模型
n_input = 2
n_hidden = 20
n_output = 1


# 计算平均绝对误差
# 训练模型并记录平均绝对误差
max_iterations = 300
mae_history = []
model = NeuralNetwork(n_input, n_hidden, n_output)
for i in range(max_iterations):
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = np.mean(np.abs(y_pred - y_test))
    mae_history.append(mae)

    # 打印每次迭代的平均绝对误差
    print(f"Iteration: {i+1}, MAE: {mae}")

# 绘制平均绝对误差随迭代次数变化的散点图
iterations = range(1, max_iterations + 1)
plt.scatter(iterations, mae_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Absolute Error')
plt.title('Scatter Plot of MAE vs Iteration')
plt.show()


# 绘制预测结果的三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(2 * np.pi * x_test[:, 0], 2 * np.pi * x_test[:, 1], y_test.flatten(), 'binary')
#ax.scatter(x_test[:, 0], x_test[:, 1], y_pred.flatten(), c='r', label='Predicted Output')
#ax.scatter(x_test[:, 0], x_test[:, 1], y_test.flatten(), c='b', label='True Output')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Output')
ax.set_title('Scatter Plot of Predicted Results')
ax.legend()
plt.show()
