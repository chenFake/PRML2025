import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# 生成数据
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 第三维添加正弦变化

    # 拼接正负月形数据并添加偏移和噪声
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y
# 可视化
def plot_3d_decision_boundary(X, y, clf, title):
    # 创建3D网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    # 生成网格点
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50),
                             np.linspace(z_min, z_max, 50))

    # 预测网格点的类别
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = clf.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    # 绘制3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制决策边界
    ax.plot_surface(xx[:, :, 25], yy[:, :, 25], zz[:, :, 25],
                    facecolors=plt.cm.viridis(Z[:, :, 25] / 2), alpha=0.3)

    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o')

    # 添加图例和标签
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()

# 生成训练数据和测试数据
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)  # 生成测试数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 决策树
dt_params = {'max_depth': [3, 5, 7, 10]}
dt_clf = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
dt_clf.fit(X_train_scaled, y_train)
dt_pred = dt_clf.predict(X_test_scaled)
print("决策树分类器:")
print(f"最佳参数: {dt_clf.best_params_}")
print(f"准确率: {accuracy_score(y_test, dt_pred):.4f}")
print(classification_report(y_test, dt_pred))

# AdaBoost
ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
ada_clf = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3)),
                       ada_params, cv=5)
ada_clf.fit(X_train_scaled, y_train)
ada_pred = ada_clf.predict(X_test_scaled)
print("\nAdaBoost分类器:")
print(f"最佳参数: {ada_clf.best_params_}")
print(f"准确率: {accuracy_score(y_test, ada_pred):.4f}")
print(classification_report(y_test, ada_pred))

# SVM分类器 - 线性核
svm_linear = GridSearchCV(SVC(kernel='linear'), {'C': [0.1, 1, 10]}, cv=5)
svm_linear.fit(X_train_scaled, y_train)
svm_linear_pred = svm_linear.predict(X_test_scaled)
print("\nSVM (线性核):")
print(f"最佳参数: {svm_linear.best_params_}")
print(f"准确率: {accuracy_score(y_test, svm_linear_pred):.4f}")
print(classification_report(y_test, svm_linear_pred))

# SVM分类器 - 多项式核
svm_poly = GridSearchCV(SVC(kernel='poly'), {'C': [0.1, 1, 10], 'degree': [2, 3]}, cv=5)
svm_poly.fit(X_train_scaled, y_train)
svm_poly_pred = svm_poly.predict(X_test_scaled)
print("\nSVM (多项式核):")
print(f"最佳参数: {svm_poly.best_params_}")
print(f"准确率: {accuracy_score(y_test, svm_poly_pred):.4f}")
print(classification_report(y_test, svm_poly_pred))

# SVM分类器 - RBF核
svm_rbf = GridSearchCV(SVC(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}, cv=5)
svm_rbf.fit(X_train_scaled, y_train)
svm_rbf_pred = svm_rbf.predict(X_test_scaled)
print("\nSVM (RBF核):")
print(f"最佳参数: {svm_rbf.best_params_}")
print(f"准确率: {accuracy_score(y_test, svm_rbf_pred):.4f}")
print(classification_report(y_test, svm_rbf_pred))


plot_3d_decision_boundary(X_test, y_test, dt_clf, "Decision Trees")
plot_3d_decision_boundary(X_test, y_test, ada_clf, "AdaBoost")
plot_3d_decision_boundary(X_test, y_test, svm_linear, "SVM (Linear)")
plot_3d_decision_boundary(X_test, y_test, svm_poly, "SVM (Plot)")
plot_3d_decision_boundary(X_test, y_test, svm_rbf, "SVM (RBF)")