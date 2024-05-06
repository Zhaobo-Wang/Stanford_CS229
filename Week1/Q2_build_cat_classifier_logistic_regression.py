import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage

# 数据集路径
dataset_path = 'D:/McMaster/2024_Summer/CS229_MachineLearning/吴恩达深度学习作业/01.机器学习和神经网络/2.第二周 神经网络基础/编程作业/datasets/'

def load_dataset():
    # 加载训练数据集
    train_dataset = h5py.File(dataset_path + 'train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集标签

    # 加载测试数据集
    test_dataset = h5py.File(dataset_path + 'test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集标签

    classes = np.array(test_dataset["list_classes"][:])  # 类别列表
    
    # 调整标签的维度
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 显示训练集中的一个图像
index = 7
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# 获取训练集和测试集的数量
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]  # 图像的尺寸

# 将每张图片从 (num_px, num_px, 3) 转换成一维数组
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# 归一化数据
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Sigmoid 激活函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# 初始化参数
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))  # 创建全零向量作为权重
    b = 0  # 偏差初始化为0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# 正向和反向传播

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # 计算激活值
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # 计算成本

    # 反向传播，计算梯度
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    
    return grads, cost



# 优化函数，执行梯度下降算法
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)  # 计算成本和梯度
        dw = grads["dw"]
        db = grads["db"]
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 每100次迭代记录一次成本
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

# 预测函数
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction

# 完整的模型函数
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    #调用 initialize_with_zeros 函数来初始化权重 w 和偏差 b。
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    '''
    调用 optimize 函数进行梯度下降优化。
    这个函数执行指定次数的梯度下降步骤来更新权重和偏差，以最小化成本函数。
    该函数返回更新后的参数 (parameters)、每步的梯度 (grads) 和每100步的成本列表 (costs)。
    '''
    w = parameters["w"]
    b = parameters["b"]
    #从 parameters 字典中提取最终优化后的权重 w 和偏差 b。
    Y_prediction_test = predict(w, b, X_test)   
    Y_prediction_train = predict(w, b, X_train)
    '''
    使用优化后的参数对训练集和测试集进行预测。
    predict 函数根据权重 w、偏差 b 和给定的输入数据 X 计算预测值。它返回一个包含预测结果(0 或 1)的数组。
    '''
    # 打印训练和测试的准确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    '''
    创建并返回一个包含模型详细信息的字典 d。
    这包括训练过程中记录的成本 (costs)、测试集和训练集的预测结果 (Y_prediction_test, Y_prediction_train)、
    最终的权重和偏差 (w, b) 以及模型的学习率和迭代次数。这些信息可以用于进一步的分析和验证模型的表现。    
    '''
    return d

# 运行模型
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# 绘制成本下降图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (hundreds times)')
plt.title("Learning Rate =" + str(d["learning_rate"]))
plt.show()
