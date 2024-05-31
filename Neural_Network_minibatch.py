import numpy as np

# 9个训练数据点和标签
X = np.array([
    [1, 2], 
    [3, 4], 
    [5, 6], 
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14],
    [15, 16],
    [17, 18]
])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 模型参数初始化
np.random.seed(1)
w1 = np.random.randn(2, 4)  # 第一层参数
w2 = np.random.randn(4, 1)  # 第二层参数

# 超参数设置
learning_rate = 0.01
batch_size = 3  # 设置批量大小
num_epochs = 100
lambda_reg = 0.01  # 正则化参数

def forward_propagation(X, w1, w2):
    Z1 = np.dot(X, w1)
    A1 = np.maximum(0, Z1)  # ReLU激活函数
    Z2 = np.dot(A1, w2)
    A2 = Z2  # 假设输出层为线性激活函数
    return Z1, A1, Z2, A2

def compute_cost(A2, y, w1, w2, lambda_reg):
    m = y.shape[0]
    loss = np.mean((A2 - y) ** 2)
    reg_term = (lambda_reg / (2 * m)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    cost = loss + reg_term
    return cost

def backpropagation(X, y, Z1, A1, Z2, A2, w1, w2, lambda_reg):
    m = y.shape[0]
    
    dZ2 = A2 - y
    dw2 = (1/m) * np.dot(A1.T, dZ2) + (lambda_reg / m) * w2
    
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * (Z1 > 0)  # ReLU的导数
    dw1 = (1/m) * np.dot(X.T, dZ1) + (lambda_reg / m) * w1
    
    return dw1, dw2

# Mini-Batch梯度下降法
for epoch in range(num_epochs):
    # 打乱数据集
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    
    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # 前向传播
        Z1, A1, Z2, A2 = forward_propagation(X_batch, w1, w2)
        
        # 计算损失
        cost = compute_cost(A2, y_batch, w1, w2, lambda_reg)
        
        # 反向传播
        dw1, dw2 = backpropagation(X_batch, y_batch, Z1, A1, Z2, A2, w1, w2, lambda_reg)
        
        # 更新参数
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
    
    if epoch % 10 == 0:
        Z1, A1, Z2, A2 = forward_propagation(X, w1, w2)
        cost = compute_cost(A2, y, w1, w2, lambda_reg)
        print(f'Epoch {epoch}, Cost: {cost}')

print(f'Final parameters w1: {w1}')
print(f'Final parameters w2: {w2}')
