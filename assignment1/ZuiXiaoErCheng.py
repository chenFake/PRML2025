import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#读取数据
df = pd.read_excel('data.xlsx')
x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values
x_test = df.iloc[:, 2].values
y_test = df.iloc[:, 3].values
#设计矩阵
X_train = np.vstack([np.ones(len(x_train)), x_train]).T
X_test = np.vstack([np.ones(len(x_test)), x_test]).T


# 最小二乘法
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(theta)
# 计算训练MSE
y_train_pred = X_train @ theta
mse_train = np.mean((y_train_pred - y_train)**2)
print('mse_train', mse_train)
# 计算测试MSE
y_test_pred = X_test @ theta
mse_test = np.mean((y_test_pred - y_test)**2)
print('mse_test', mse_test)

# 结果可视化
plt.figure(figsize=(10, 6))
x_min = min(x_train.min(), x_test.min())
x_max = max(x_train.max(), x_test.max())
x_line = np.linspace(x_min, x_max, 100)
X_line = np.vstack([np.ones(100), x_line]).T
y_line = X_line @ theta

plt.scatter(x_train, y_train, c='b', label='Train Data')
plt.scatter(x_test, y_test, c='r', marker='x', label='Test Data')
plt.plot(x_line, y_line, 'g-', label=f'Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'OLS')
plt.legend()
plt.grid(True)
plt.show()
#
# # 打印结果
# print(f"模型参数 theta: {theta}")
# print(f"训练集MSE: {mse_train:.4f}")
# print(f"测试集MSE: {mse_test:.4f}")