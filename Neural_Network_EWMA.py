import numpy as np
import matplotlib.pyplot as plt

# 计算EWMA
def calculate_ewma(data, alpha):
    ewma = np.zeros(len(data))
    ewma[0] = data[0]
    for t in range(1, len(data)):
        ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t - 1]
    return ewma


data = np.array([100, 102, 101, 105, 107, 108, 110, 111, 115, 117])
alpha = 0.3  # 平滑因子

# 计算EWMA
ewma = calculate_ewma(data, alpha)

# 预测下一个数据点值
predicted_next_value = ewma[-1]
print("预测的下一个数据点值为：", predicted_next_value)
print("EWMA序列：", ewma)


# 绘制原始数据和EWMA值
plt.plot(data, label='Original Data', marker='o')
plt.plot(ewma, label='EWMA', linestyle='--', marker='x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Data vs EWMA')
plt.legend()
plt.show()