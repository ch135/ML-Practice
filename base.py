import numpy as np

# 生成一维数组
print(np.arange(4))
print(np.arange(4, 20, 2))

# 生成数据在[0,1)之间的随机数组
print(np.random.rand(4, 4))
# 生成数据在9-1,1)之间的正则化随机数组
print(np.random.randn(4, 4))
# 功能和rand一样；区别：rand传入多个参数，random传入单个参数
print(np.random.random((3, 4)))

# 数组转化为矩阵
matrix = np.mat(np.random.randn(3, 4))
print(matrix)
# 矩阵求逆
print(matrix.I)
# 矩阵转置
print(matrix.T)
# 创建单位矩阵
print(np.eye(4))

