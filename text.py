import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义适应度函数
def evaluate(individual):
    hidden_layer_size = (individual[0] + 1) * 10
    activation_func = ['relu', 'tanh', 'sigmoid'][individual[1]]
    alpha_value = 10 ** (-individual[2])

    # 构建神经网络模型
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation=activation_func, alpha=alpha_value)

    # 在训练集上训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率作为适应度
    fitness_score = accuracy_score(y_test, y_pred)

    return fitness_score,

# 创建遗传算法的基本元素
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", np.random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 遗传算法参数设置
population_size = 20
num_generations = 50

population = toolbox.population(n=population_size)

# 迭代优化过程
for generation in range(num_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# 选择最优个体
best_individual = tools.selBest(population, k=1)[0]

# 输出最优参数
hidden_layer_size = (best_individual[0] + 1) * 10
activation_func = ['relu', 'tanh', 'sigmoid'][best_individual[1]]
alpha_value = 10 ** (-best_individual[2])
print("Best Parameters:")
print("Hidden Layer Sizes:", hidden_layer_size)
print("Activation Function:", activation_func)
print("Alpha:", alpha_value)
