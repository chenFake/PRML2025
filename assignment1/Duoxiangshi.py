import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 读取数据
df = pd.read_excel('data.xlsx')
x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values
# print(y_train)
x_test = df.iloc[:, 2].values
y_test = df.iloc[:, 3].values

x_train_data=np.array(x_train).reshape(-1,1)
y_train_data=np.array(y_train)
x_test_data = np.array(x_test).reshape(-1, 1)
y_test_data = np.array(y_test)

degree=16
poly = PolynomialFeatures(degree=degree)
# print(poly.fit_transform(x_train_data))
x_poly_train = poly.fit_transform(x_train_data)
x_poly_test = poly.transform(x_test_data)


model = LinearRegression()
model.fit(x_poly_train, y_train_data)

# 训练集预测
y_train_pred = model.predict(x_poly_train)
# 测试集预测
y_test_pred = model.predict(x_poly_test)

mse_train = np.mean((y_train_data - y_train_pred) ** 2)
mse_test = np.mean((y_test_data - y_test_pred) ** 2)
print('mse_train:', mse_train)
print('mse_test:', mse_test)

x_fit = np.linspace(min(x_train_data), max(x_train_data), 100).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)
# print(len(y_fit))
mse_train = np.mean((y_fit - y_train) ** 2)

plt.figure(figsize=(10, 6))
#回归拟合结果
x_min = min(x_train.min(), x_test.min())
x_max = max(x_train.max(), x_test.max())

plt.scatter(x_train, y_train, c='b', label='Train Data')
plt.scatter(x_test, y_test, c='r', marker='x', label='Test Data')
plt.plot(x_fit, y_fit, color='green', label='Fitted Curve', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()


coefficients = model.coef_
intercept = model.intercept_
print(degree)
print("多项式系数:", coefficients)
print("截距:", intercept)
