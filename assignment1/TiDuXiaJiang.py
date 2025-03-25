import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取数据
df = pd.read_excel('data.xlsx')

x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values
# print(len(y_train))
x_test = df.iloc[:, 2].values
y_test = df.iloc[:, 3].values
#设计矩阵
X_train = np.vstack([np.ones(len(x_train)), x_train]).T
X_test = np.vstack([np.ones(len(x_test)), x_test]).T

# 初始化参数
theta = np.zeros(2)
learning_rate = 0.01
epsilon = 1e-6
max_iterations = 10000
m = len(y_train)

# 梯度下降
loss_history = []
for i in range(max_iterations):
    y_pred = X_train @ theta
    error = y_pred - y_train

    gradient = (1 / m) * X_train.T @ error

    theta = theta - learning_rate * gradient

    current_loss = np.mean(error ** 2)
    loss_history.append(current_loss)

    # 收敛条件判断
    if i > 0 and abs(prev_loss - current_loss) < epsilon:
        print(f'{len(loss_history)}')
        break

    prev_loss = current_loss


y_train_pred = X_train @ theta
mse_train = np.mean((y_train_pred - y_train) ** 2)

y_test_pred = X_test @ theta
mse_test = np.mean((y_test_pred - y_test) ** 2)

plt.figure(figsize=(10, 6))
#回归拟合结果
x_min = min(x_train.min(), x_test.min())
x_max = max(x_train.max(), x_test.max())
x_line = np.linspace(x_min, x_max, 100)
y_line = theta[0] + theta[1] * x_line

plt.scatter(x_train, y_train, c='b', label='Train Data')
plt.scatter(x_test, y_test, c='r', marker='x', label='Test Data')
plt.plot(x_line, y_line, 'g-', label=f'Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'GD')
plt.legend()
plt.grid(True)
#
#损失函数下降曲线
# plt.plot(range(len(loss_history)), loss_history, 'r-')
# plt.xlabel('Iterations')
# plt.ylabel('MSE')
# plt.title('Loss Convergence Process')
# plt.grid(True)
#

plt.show()

# 打印结果
print(f"最终模型参数 theta: {theta}")
print(f"训练集MSE: {mse_train:.4f}")
print(f"测试集MSE: {mse_test:.4f}")
print(f"初始学习率: {learning_rate}")