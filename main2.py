import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('LSTM-Multivariate_pollution.csv')
data['datetime'] = pd.to_datetime(data['date'])
data.set_index('datetime', inplace=True)
data.drop(['date'], axis=1, inplace=True)

# 查看数据
print(data.head())

# 特征选择
features = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data = data[features]

# 填充缺失值
data = data.fillna(method='pad')
data = data.fillna(0)

# 数据预处理
numeric_features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
categorical_features = ['wnd_dir']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 将数据转换为数值型数组
data_processed = preprocessor.fit_transform(data)

# 创建时间序列数据
def create_sequences(data_processed, window_size):
    X, y = [], []
    for i in range(len(data_processed) - window_size):
        X.append(data_processed[i:i+window_size, :-len(preprocessor.named_transformers_['cat'].categories_[0])])
        y.append(data_processed[i+window_size, 0])  # 假设目标值是第一个特征（pollution）
    return np.array(X), np.array(y)

window_size = 24  # 使用过去24小时的数据来预测下一小时的PM2.5浓度
X, y = create_sequences(data_processed, window_size)

# 拆分训练集和测试集
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(window_size, X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 添加MAE作为评价指标

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=False
)

# 评估与预测
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss[0]:.4f}')  # MSE
print(f'Test MAE: {loss[1]:.4f}')   # MAE

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 使用训练好的模型进行预测
y_pred = model.predict(X_test)

# 绘制预测值与真实值的对比图（整体数据）
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values')
plt.plot(y_pred, label='Predicted Values')
plt.title('True vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()

# 绘制预测值与真实值的对比图（截取前500个时间步）
plt.figure(figsize=(12, 6))
plt.plot(y_test[:500], label='True Values')
plt.plot(y_pred[:500], label='Predicted Values')
plt.title('True vs Predicted Values (First 500 Time Steps)')
plt.xlabel('Time Steps')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()

# 计算其他评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")