import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

m = 10000 # 矩阵行数
n = 36 # 矩阵列数

# 创建一个m行n列的零矩阵
matrix = np.zeros((m, n))

# 设置第一行的元素为1
matrix[0, :] = 1

# 设置第二行的前28个元素为1
matrix[1, :28] = 1

# 设置奇数行和第一行一样，偶数行和第二行一样
for i in range(2, m, 2):
    matrix[i, :] = matrix[0, :]
for i in range(3, m, 2):
    matrix[i, :] = matrix[1, :]

matrix = matrix.astype(int)

# 将稀疏矩阵转换为COO格式
coo = coo_matrix(matrix)

# 将COO格式的矩阵保存为Matrix Market格式的文件
mmwrite("matrix1.mtx", coo)
