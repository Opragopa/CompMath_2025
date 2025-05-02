import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def integral_eqn(x):
    integrand = lambda z: np.exp(-x * z) / (x + 1)
    integral, _ = quad(integrand, 0, 20)
    return integral - 0.59639442 * x


x_star = fsolve(integral_eqn, 1.0)[0]
l = 100 * x_star
E = 3e7
P_values = np.arange(500, 1100, 100)

print(f"Найденный корень x* = {x_star:.6f}")
print(f"Длина балки l = {l:.2f}")

def beam_equation(x, y, P):
    I = 5 * (1 + 4 * np.exp(-6 * x / l))
    dI = -120 / l * np.exp(-6 * x / l)
    d2I = 720 / (l ** 2) * np.exp(-6 * x / l)

    omega: int = 0  # d^2M/dx^2 = 0 для линейного момента

    dydx = np.zeros_like(y)
    dydx[0] = y[1]
    dydx[1] = y[2]
    dydx[2] = y[3]

    dydx[3] = - (2 / I) * dI * y[3] - (1 / I) * d2I * y[2] + omega / (E * I ** 2)

    return dydx


y_l_values = []

for P in P_values:
    I0 = 5 * (1 + 4)  # при x=0 (e^0 = 1)
    y0 = [0,
          0,
          (P * l)/75 * 1e-7,
          (P * 3.8)/75 * 1e-7]

    sol = solve_ivp(beam_equation, [0, l], y0, args=(P,),
                    method='RK45', rtol=1e-6, atol=1e-8)
    y_l_values.append(sol.y[0, -1])

cs = CubicSpline(P_values, y_l_values)
P_750 = 750
y_750 = cs(P_750)

h = P_values[1] - P_values[0]
error_estimate = h ** 4 * np.max(np.abs(cs(P_values, 4))) / 384

print("\nРезультаты расчета:")
print("P | y(l)")
for P, y in zip(P_values, y_l_values):
    print(f"{P:5d} | {y:.6f}")
print(f"\nИнтерполированное значение при P=750: {y_750:.6f} м")
print(f"Оценка погрешности: {error_estimate:.2e}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(P_values, y_l_values, 'o', label='Расчетные точки')
P_interp = np.linspace(500, 1000, 100)
plt.plot(P_interp, cs(P_interp), label='Кубический сплайн')
plt.axvline(x=750, color='r', linestyle='--', label='P=750 Н')
plt.xlabel('Нагрузка P')
plt.ylabel('Прогиб y(l)')
plt.title('Зависимость прогиба от нагрузки')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
sol_750 = solve_ivp(beam_equation, [0, l], y0, args=(750,),
                    method='RK45', dense_output=True)
x_plot = np.linspace(0, l, 100)
y_plot = sol_750.sol(x_plot)[0]
plt.plot(x_plot, y_plot)
plt.xlabel('Координата x')
plt.ylabel('Прогиб y(x)')
plt.title(f'Форма прогиба при P=750 Н (l={l:.2f} м)')
plt.grid(True)

plt.tight_layout()
plt.show()
