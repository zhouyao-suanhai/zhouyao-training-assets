import numpy as np
import scipy.integrate
from sympy import symbols, integrate, exp, cos, simplify
import matplotlib.pyplot as plt

# 封装成类
class IntegralDemo:
    def __init__(self):
        # 区间改为 [0, 2π]
        self.lower = 0
        self.upper = 2 * np.pi
        self.n = 200

        # 目标函数改为 u(x) = exp(-x) * cos(2x)
        self.x = symbols('x')
        self.u_expr = exp(-self.x) * cos(2 * self.x)
        self.func = lambda x: np.exp(-x) * np.cos(2 * x)

    # 函数1: SymPy - 符号积分
    def sympy_integral(self):
        integral = integrate(self.u_expr, (self.x, self.lower, self.upper))
        return simplify(integral)

    # 函数2: NumPy - 生成网格与函数值
    def numpy_grid(self):
        x_vals = np.linspace(self.lower, self.upper, self.n)
        u_vals = self.func(x_vals)
        return x_vals, u_vals

    # 函数3: SciPy - 数值积分
    def scipy_numerical_integral(self, x_vals, u_vals):
        return scipy.integrate.simpson(u_vals, x_vals)

    # 函数4: Matplotlib - 可视化（只填充 u>0 区域 + 标注）
    def plot_function_with_area(self, x_vals, u_vals, symbolic_val, numeric_val):
        plt.figure(figsize=(8, 5))

        # 画函数曲线
        plt.plot(x_vals, u_vals, label=r'$u=e^{-x}\cos(2x)$', color='blue')
        
        # 绘制x轴参考线
        plt.axhline(y=0, color='#7f8c8d', linestyle='--', linewidth=1.5)

        # 只填充 u(x) > 0 的区域
        mask = u_vals > 0
        plt.fill_between(x_vals, u_vals, where=mask,
                         color='green', alpha=0.4, label='正值区域')

        # 标注数值积分值与误差
        error = abs(float(symbolic_val) - numeric_val)
        text = f"数值积分: {numeric_val:.5f}\n误差: {error:.2e}"
        plt.text(0.05, 0.85, text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.title('函数 $u=e^{-x}\\cos(2x)$ 与正值积分区域')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.grid(True)
        plt.show()


# 实例化并运行
demo = IntegralDemo()

# SymPy 符号积分
symbolic_int = demo.sympy_integral()
print("符号积分值:", float(symbolic_int))

# NumPy 网格
x_vals, u_vals = demo.numpy_grid()

# SciPy 数值积分
numeric_int = demo.scipy_numerical_integral(x_vals, u_vals)
print("数值积分值:", numeric_int)

# Matplotlib 可视化
demo.plot_function_with_area(x_vals, u_vals, symbolic_int, numeric_int)
