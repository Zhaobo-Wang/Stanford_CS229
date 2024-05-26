import numpy as np

# 初始化权重和偏置
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# ReLU激活函数及其导数
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 前向传播
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# 计算损失（包含正则化）
def compute_loss(A2, Y, W1, W2, lambd):
    m = Y.shape[1]
    cross_entropy_loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    L2_regularization_cost = lambd/(2*m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss = cross_entropy_loss + L2_regularization_cost
    return loss

# 反向传播（包含正则化）
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, lambd):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T) + (lambd/m) * W2
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T) + (lambd/m) * W1
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# 参数更新
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

# 训练神经网络
def train_neural_network(X, Y, input_size, hidden_size, output_size, num_iterations, learning_rate, lambd):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(A2, Y, W1, W2, lambd)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, lambd)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    return W1, b1, W2, b2

# 示例数据
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # 输入特征
Y = np.array([[0, 1, 1, 0]])  # 输出标签

# 训练神经网络
input_size = X.shape[0]
hidden_size = 2
output_size = 1
num_iterations = 10000
learning_rate = 0.1
lambd = 0.7  # 正则化参数

W1, b1, W2, b2 = train_neural_network(X, Y, input_size, hidden_size, output_size, num_iterations, learning_rate, lambd)

# 测试神经网络
def predict(X, W1, b1, W2, b2):
    _, A1, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = A2 > 0.5
    return predictions

predictions = predict(X, W1, b1, W2, b2)
print("Predictions:", predictions)
