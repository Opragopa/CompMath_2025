import numpy as np
import pandas as pd
import math
from scipy.interpolate import CubicSpline

x_data = np.array([1.0, 1.2, 1.5, 1.6, 1.8, 2.0])
y_data = np.array([5.000, 6.899, 11.180, 13.133, 18.119, 25.000])

f = CubicSpline(x_data, y_data)

# Создание таблицы
data = {
    'a': [],
    'b': [],
    'x': [],
    'f(x)': []
}
df = pd.DataFrame(data)

def equation(x):
    return f(x) - (6 * x + 3)

def bisection_method(a, b, tol, max_iter=100):
    if equation(a) * equation(b) >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала [a, b].")

    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        c = (a + b) / 2
        if equation(c) == 0:
            return c
        elif equation(a) * equation(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
        new_row = {'a': a, 'b': b, 'x': c, 'f(x)': equation(c)}
        df.loc[len(df)] = new_row

    return (a + b) / 2

tol = 1e-6
a = 1.0
b = 2.0

try:
    root = bisection_method(a, b, tol)
    print(df)
    digit = (-1) * math.log10(tol)
    print(f"Найденный корень: {round(root, int(digit))}")
except ValueError as e:
    print(e)
