"""
任务3.1.1：创建 3D 张量 T，形状为 (2, 3, 4)，用简单序列填充（如 np.arange(24).reshape(2,3,4)）。对 T 沿最后一个轴进行简单加权求和：先构造一个长度为 4 的权重向量 w（例如 [1, 2, -1, 0.5]），然后用 np.einsum('ijk,k->ij', T, w) 计算 2D 结果 M
"""
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建 3D 张量 T，形状 (2, 3, 4)
T = np.arange(24).reshape(2, 3, 4)
print("张量 T 的形状:", T.shape)
print("T 的内容:\n", T)

# 2. 构造权重向量 w（长度 4）
w = np.array([1, 2, -1, 0.5])
print("权重向量 w:", w)

# 3. 爱因斯坦求和：沿最后一个轴 (k) 加权求和，得到形状 (2,3) 的矩阵 M
#    公式: M[i,j] = Σ_k T[i,j,k] * w[k]
M = np.einsum('ijk,k->ij', T, w)
print("结果矩阵 M 的形状:", M.shape)
print("M 的内容:\n", M)
