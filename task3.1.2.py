"""
任务3.1.2：在区间 ([0,1]) 上取 (n=50) 个等距内点（不包含两端点），定义步长 (h=1/(n+1)) 以及对应的坐标 (x_i=ih)。根据给定规则构造一个三对角稀疏矩阵 (A)，其主对角线元素为 (2/h^2)，上下相邻对角线元素为 (-1/h^2)，同时构造右端向量 (b)，其中 (b_i=x_i(1-x_i))。使用 SciPy 的稀疏线性求解器 spsolve 求解线性方程组 (Au=b)，得到解向量 (u)，并将内点坐标 (x_i) 与对应的解 (u_i) 绘制为折线图以观察解的整体结构。最后取 (n=3) 的小规模情形，写出对应的矩阵和向量，通过手工计算或借助计算器求解线性方程组，并将结果与程序输出进行对比，以验证数值实现的正确性。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ========== 数值求解 n=50 ==========
n = 50
h = 1.0 / (n + 1)
x = np.linspace(h, 1 - h, n)          # 内点坐标

# 构造三对角矩阵 A (CSR格式)
main_diag = 2.0 / h**2 * np.ones(n)
off_diag = -1.0 / h**2 * np.ones(n - 1)
A = diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr')

# 右端项 b
b = x * (1 - x)

# 求解线性方程组 Au = b
u = spsolve(A, b)

# 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(x, u, 'o-', color='crimson', markersize=4, label='数值解 u(x)')
plt.title(f"方程的解 (n={n} 个内点)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# ========== 手工验证 n=3 小规模 ==========
print("\n" + "="*50)
print("手工验证 (n=3)")
print("="*50)

n_small = 3
h_small = 1.0 / (n_small + 1)          # h = 0.25
x_small = np.linspace(h_small, 1 - h_small, n_small)   # [0.25, 0.5, 0.75]

# 构造稠密矩阵 A_small
A_small = (2.0 / h_small**2) * np.eye(n_small) + \
          (-1.0 / h_small**2) * (np.eye(n_small, k=1) + np.eye(n_small, k=-1))

b_small = x_small * (1 - x_small)

print(f"h = {h_small}")
print(f"内点 x = {x_small}")
print("矩阵 A (手工形式，稠密):\n", A_small)
print("右端项 b =", b_small)

# 手工求解（使用 numpy.linalg.solve）
u_small = np.linalg.solve(A_small, b_small)
print("手工计算得到的解 u =", u_small)

# 与稀疏求解器结果对比
A_small_sparse = diags([2/h_small**2, -1/h_small**2, -1/h_small**2], [0, -1, 1], shape=(3,3), format='csr')
u_small_sparse = spsolve(A_small_sparse, b_small)
print("稀疏求解器得到的解 u =", u_small_sparse)

# 验证一致性
if np.allclose(u_small, u_small_sparse):
    print("\n 手工验证通过：稠密求解与稀疏求解结果一致。")
else:
    print("\n 验证失败，请检查代码。")
