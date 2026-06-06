"""
任务3.2.3：用 FEALPy 的 Sparse 模块，分别创建 1000、10000、100000、1000000 阶的稀疏矩阵（可用随机结构或对角结构），再创建同阶随机向量，考察不同后端下稀疏矩阵-向量乘法的效率。
"""
import time
import pandas as pd
from fealpy.backend import backend_manager as bm
from fealpy.sparse import coo_matrix, csr_matrix
from fealpy.utils import timer

# ========== 公共参数 ==========
sizes = [1000, 10000, 100000, 1000000]   # 矩阵阶数
nz_per_row = 10                          # 每行非零元个数
repeats = 10                             # 每个规模重复计算次数（取平均）

#使用 bm.repeat 和 bm.random.randint 一次性生成所有索引，构建稀疏矩阵的时间从 O(N) 降低到 O(1) 的底层向量化操作,消除了python循环
def create_sparse_matrix_and_vector(N, nz_per_row):
    nnz = N * nz_per_row
    # 行索引：每行重复 nz_per_row 次
    row = bm.repeat(bm.arange(N, dtype=bm.int32), nz_per_row)
    # 列索引：一次性随机生成，可能同一行内有重复列（但对性能测试影响很小）
    col = bm.random.randint(0, N, size=(nnz,), dtype=bm.int32)
    # 数值
    data = bm.random.randn(nnz)
    # 构建 COO 并转换为 CSR
    coo = coo_matrix((data, (row, col)), shape=(N, N))
    A = coo.tocsr()
    x = bm.random.randn(N)
    return A, x

results = []   # 存储所有结果

# ================= 1. 测试 NumPy 后端 (CPU) =================
print("="*50)
print("开始测试 NumPy 后端 (CPU)")
bm.set_backend('numpy')

for N in sizes:
    print(f"  生成 {N}×{N} 矩阵 (每行 {nz_per_row} 非零元) ...")
    A, x = create_sparse_matrix_and_vector(N, nz_per_row)

    # 预热
    _ = A @ x

    # 使用 timer 记录标签（但不依赖其返回值）
    tmr = timer()
    next(tmr)                       # 启动计时器（记录一个时间点）
    start = time.perf_counter()     # 高精度计时起点（秒）

    for _ in range(repeats):
        y = A @ x

    end = time.perf_counter()
    elapsed = end - start           # 总耗时（秒）
    tmr.send(f"numpy_{N}")          # 用于记录标签，不影响计算结果

    avg_time = elapsed / repeats
    results.append({
        'Size': N,
        'Backend': 'NumPy (CPU)',
        'Avg Time (ms)': avg_time * 1000
    })
    print(f"    平均耗时: {avg_time*1000:.4f} ms")

# ================= 2. 测试 PyTorch 后端 (CPU) =================
print("\n" + "="*50)
print("开始测试 PyTorch 后端 (CPU)")
bm.set_backend('pytorch')
bm.set_default_device('cpu')

for N in sizes:
    print(f"  生成 {N}×{N} 矩阵 (每行 {nz_per_row} 非零元) ...")
    A, x = create_sparse_matrix_and_vector(N, nz_per_row)

    # 预热
    _ = A @ x

    tmr = timer()
    next(tmr)
    start = time.perf_counter()

    for _ in range(repeats):
        y = A @ x

    end = time.perf_counter()
    elapsed = end - start
    tmr.send(f"pytorch_{N}")

    avg_time = elapsed / repeats
    results.append({
        'Size': N,
        'Backend': 'PyTorch (CPU)',
        'Avg Time (ms)': avg_time * 1000
    })
    print(f"    平均耗时: {avg_time*1000:.4f} ms")

# ================= 输出对比表格 =================
df = pd.DataFrame(results)
pivot = df.pivot(index='Size', columns='Backend', values='Avg Time (ms)')
print("\n\n========== 性能对比表 (单位: ms) ==========")
print(pivot.round(4))
