"""
任务3.2.2：用 FEALPy 的后端管理器 bm，编写一个“数值微分”函数（如 instruction.md 示例），分别在 NumPy 和 PyTorch 后端下运行，比较输出结果。
"""
import numpy as np
from fealpy.backend import backend_manager as bm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def central_derivative(f, x0, h=1e-6):
    x0_tensor = bm.array(x0, dtype=bm.float64)
    x1 = x0_tensor + h
    x2 = x0_tensor - h
    return (f(x1) - f(x2)) / (2 * h)

def compare_backends():
    test_points = [0.5, 1.0, 1.23, 2.0, 3.14159]
    func = bm.sin
    results = {}

    # NumPy 后端
    bm.set_backend('numpy')
    logger.info("当前后端: NumPy")
    results['numpy'] = central_derivative(func, test_points)

    # PyTorch 后端 (CPU)
    bm.set_backend('pytorch')
    bm.set_default_device('cpu')
    logger.info("当前后端: PyTorch (CPU)")
    results['pytorch_cpu'] = central_derivative(func, test_points)

    # 输出结果
    print("\n" + "=" * 60)
    print("中心差分结果对比 (导数近似值)")
    print("=" * 60)
    print(f"测试点: {test_points}\n")
    for backend, vals in results.items():
        if vals is None:
            continue
        if hasattr(vals, 'numpy'):
            vals = vals.numpy()
        print(f"{backend:15s}: {vals}")

    print("\n" + "-" * 60)
    print("差异分析 (相对于 NumPy 结果)")
    numpy_vals = results['numpy']
    for backend, vals in results.items():
        if backend == 'numpy' or vals is None:
            continue
        if hasattr(vals, 'numpy'):
            vals = vals.numpy()
        abs_diff = np.abs(vals - numpy_vals)
        max_abs = np.max(abs_diff)
        print(f"{backend:15s} -> 最大绝对误差: {max_abs:.2e}")

if __name__ == "__main__":
    compare_backends()
