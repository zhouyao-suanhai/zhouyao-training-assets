"""
任务3.1.3：将积分验证案例封装成类
目标函数：u(x) = exp(-x) * cos(2x)，区间 [0, 2π]
可视化：只填充 u(x) > 0 的区域，并标注积分值与误差
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from sympy import symbols, integrate, exp, cos, pi

class IntegralVisualizer:
    """
    封装符号积分、数值积分、可视化功能的类
    """
    def __init__(self, func_expr, x_symbol, lower, upper, numeric_func):
        """
        参数:
            func_expr : SymPy 表达式，例如 exp(-x)*cos(2*x)
            x_symbol  : SymPy 符号变量
            lower, upper : 积分下限和上限
            numeric_func : NumPy 向量化函数，例如 lambda x: np.exp(-x)*np.cos(2*x)
        """
        self.func_expr = func_expr
        self.x = x_symbol
        self.lower = lower
        self.upper = upper
        self.numeric_func = numeric_func

    def symbolic_integral(self):
        """计算符号积分，返回精确值（SymPy 对象）"""
        integral = integrate(self.func_expr, (self.x, self.lower, self.upper))
        return integral.simplify()

    def numerical_integral(self, n_points=1000):
        """使用 SciPy 的 Simpson 法则计算数值积分"""
        x_vals = np.linspace(self.lower, self.upper, n_points)
        y_vals = self.numeric_func(x_vals)
        return simpson(y_vals, x_vals)

    def plot_with_positive_fill(self, n_points=500, show_error=True):
        """
        绘制函数曲线，只填充 y>0 的区域，并在图上标注数值积分值和误差
        """
        x_vals = np.linspace(self.lower, self.upper, n_points)
        y_vals = self.numeric_func(x_vals)

        # 计算数值积分和误差
        num_int = self.numerical_integral(n_points=n_points)
        sym_int = float(self.symbolic_integral())
        error = abs(num_int - sym_int)

        # 创建图形
        plt.figure(figsize=(10, 5))

        # 绘制完整曲线
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'$u(x) = {self.func_expr}$')

        # 只填充 y>0 的区域
        y_fill = np.where(y_vals > 0, y_vals, np.nan)
        plt.fill_between(x_vals, 0, y_fill, where=(y_vals > 0),
                         color='green', alpha=0.3, label='$u(x) > 0$ 区域')

        # 添加零线
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # 添加文本注释（积分值和误差）
        text_str = f"数值积分: {num_int:.6f}\n符号积分: {sym_int:.6f}\n绝对误差: {error:.2e}"
        if show_error:
            plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f"函数 $u(x)$ 在 $[{self.lower}, {self.upper}]$ 上的积分区域\n(仅填充正值部分)")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        return num_int, sym_int, error


# ========== 实例化并运行 ==========
if __name__ == "__main__":
    # 定义新函数：u(x) = exp(-x) * cos(2x)
    x_sym = symbols('x')
    func_sym = exp(-x_sym) * cos(2 * x_sym)          # SymPy 表达式
    func_numeric = lambda x: np.exp(-x) * np.cos(2 * x)   # NumPy 向量化函数

    # 积分区间 [0, 2π]
    a, b = 0, 2 * np.pi

    # 创建可视化器实例
    viz = IntegralVisualizer(func_sym, x_sym, a, b, func_numeric)

    # 输出符号积分值
    sym_int = viz.symbolic_integral()
    print("符号积分值 (精确):", sym_int)
    print("符号积分浮点数:", float(sym_int))

    # 数值积分并绘图
    num_int, sym_float, error = viz.plot_with_positive_fill(n_points=1000, show_error=True)
    print(f"数值积分值: {num_int:.8f}, 绝对误差: {error:.2e}")
