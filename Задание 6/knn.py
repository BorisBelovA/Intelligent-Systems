# Математика
import numpy as np
# Функция, которая считает расстояния между парами точек из двух массивов, подающихся ей на вход
from scipy.spatial.distance import cdist
# Визуализация
import matplotlib.pyplot as plt

# Инициализация точек
X = np.zeros((8, 2))
X[0] = 1, 3
X[1] = 3, 3
X[2] = 4, 3
X[3] = 5, 3
X[4] = 1, 2
X[5] = 4, 2
X[6] = 1, 1
X[7] = 2, 1

# Инициализация плоскости
plt.figure(figsize=(5, 5))
plt.plot(X[:, 0], X[:, 1], 'bo');
plt.figure(figsize=(5, 5))
plt.plot(X[:, 0], X[:, 1], 'bo');
plt.show()

# Инициализация центроидов
np.random.seed(seed=42)
centroids = np.random.normal(loc=0.0, scale=1., size=4)
centroids = centroids.reshape((2, 2))
cent_history = [centroids]

# for i in range(2):
#     # Считаем расстояния от наблюдений до центроид
#     distances = cdist(X, centroids)
#     # Смотрим, до какой центроиде каждой точке ближе всего
#     labels = distances.argmin(axis=1)
#
#     # Положим в каждую новую центроиду геометрический центр её точек
#     centroids = centroids.copy()
#     centroids[0, :] = np.mean(X[labels == 0, :], axis=0)
#     centroids[1, :] = np.mean(X[labels == 1, :], axis=0)
#
#     cent_history.append(centroids)
