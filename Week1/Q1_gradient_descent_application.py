'''

假设我们有以下数据集，它包含一些点，我们需要找到一条直线来近似这些点：

* 数据点 (x, y): (1, 2), (2, 3), (3, 5), (4, 6)

我们想要拟合一个模型 y=wx+b，其中 w 是权重（斜率），b 是偏置（y轴截距）。

'''


import numpy as np

# 数据点
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 6])

# 初始化参数
w = 0.0
b = 0.0

# 学习率
alpha = 0.01

# 迭代次数
iterations = 1000

# 梯度下降迭代
for i in range(iterations):
    y_pred = w * x + b
    error = y - y_pred
    gradient_w = -2 * np.dot(x, error) / len(x)
    gradient_b = -2 * np.sum(error) / len(x)
    
    w = w - alpha * gradient_w
    b = b - alpha * gradient_b

    if i % 100 == 0:
        print(f"Iteration {i}: w = {w}, b = {b}, Loss = {np.mean(error ** 2)}")

# 最终结果
print(f"Final parameters: w = {w}, b = {b}")
