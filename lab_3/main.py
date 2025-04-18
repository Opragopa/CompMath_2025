import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def system_of_equations(time, state):
    x1, x2 = state
    dx1dt = -73 * x1 - 210 * x2 + np.log(1 + time ** 2)
    dx2dt = x1 + np.exp(-time) + time ** 2 + 1
    return [dx1dt, dx2dt]


def compute_jacobian():
    J = [
        [-73, -210],  # df1/dx1, df1/dx2
        [1, 0]  # df2/dx1, df2/dx2
    ]
    return J


def compute_eigenvalues(J):
    a, b = J[0][0], J[0][1]
    c, d = J[1][0], J[1][1]

    trace = a + d
    det = a * d - b * c

    discriminant = trace ** 2 - 4 * det

    if discriminant >= 0:
        λ1 = (trace + np.sqrt(discriminant)) / 2
        λ2 = (trace - np.sqrt(discriminant)) / 2
    else:
        real_part = trace / 2
        imag_part = np.sqrt(-discriminant) / 2
        λ1 = complex(real_part, imag_part)
        λ2 = complex(real_part, -imag_part)

    return λ1, λ2


def estimate_critical_step_size():
    J = compute_jacobian()
    λ1, λ2 = compute_eigenvalues(J)
    eigenvalues = [λ1, λ2]
    print(f"Собственные значения матрицы Якоби: {eigenvalues}")

    max_eig = max(abs(λ1), abs(λ2))
    h_crit = 1.0 / max_eig
    print(f"Грубая оценка критического шага: {h_crit:.6f}")

    h_values = np.linspace(0.8 * h_crit, 1.2 * h_crit, 100)
    stable = []

    for h in h_values:
        stable_h = True
        for λ in eigenvalues:
            stability = abs(1 + h * λ + 0.5 * (h * λ) ** 2)
            if stability > 1.0 + 1e-6:
                stable_h = False
                break
        stable.append(stable_h)

    if any(stable):
        h_crit_refined = max(h for h, s in zip(h_values, stable) if s)
        print(f"Уточненный критический шаг: {h_crit_refined:.6f}")
        return h_crit_refined
    else:
        print("Не удалось найти устойчивый шаг в заданном диапазоне")
        return h_crit


def adams_bashforth_second_order(system_func, time_span, initial_state,
                                 step_size, output_step):
    t_start, t_end = time_span
    time_points = [t_start]
    state_values = [np.array(initial_state)]

    k1 = np.array(system_func(t_start, initial_state))
    k2 = np.array(system_func(t_start + step_size / 2,
                              initial_state + step_size / 2 * k1))
    k3 = np.array(system_func(t_start + step_size / 2,
                              initial_state + step_size / 2 * k2))
    k4 = np.array(system_func(t_start + step_size,
                              initial_state + step_size * k3))

    next_state = initial_state + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    time_points.append(t_start + step_size)
    state_values.append(next_state)

    current_time = t_start + step_size
    while current_time < t_end - 1e-10:
        if current_time + step_size > t_end:
            step_size = t_end - current_time

        prev_derivative = np.array(system_func(time_points[-2], state_values[-2]))
        current_derivative = np.array(system_func(time_points[-1], state_values[-1]))

        next_state = state_values[-1] + step_size * (1.5 * current_derivative -
                                                     0.5 * prev_derivative)

        current_time += step_size
        time_points.append(current_time)
        state_values.append(next_state)

    output_times = np.arange(time_span[0], time_span[1] + 1e-10, output_step)
    x1_solution = np.interp(output_times, time_points, [s[0] for s in state_values])
    x2_solution = np.interp(output_times, time_points, [s[1] for s in state_values])

    return output_times, x1_solution, x2_solution


def plot_results(rkf45_solution, adams_solutions, h_crit):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(rkf45_solution.t, rkf45_solution.y[0], 'o-', label='RKF45')
    for sol, h in zip(adams_solutions, [0.025, 0.01, h_crit]):
        label = f'Адамс h={h:.4f}' if h == h_crit else f'Адамс h={h:.3f}'
        plt.plot(sol[0], sol[1], '--', label=label)
    plt.xlabel('Время')
    plt.ylabel('x1(t)')
    plt.title('Решение системы: x1(t)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(rkf45_solution.t, rkf45_solution.y[1], 'o-', label='RKF45')
    for sol, h in zip(adams_solutions, [0.025, 0.01, h_crit]):
        label = f'Адамс h={h:.4f}' if h == h_crit else f'Адамс h={h:.3f}'
        plt.plot(sol[0], sol[2], '--', label=label)
    plt.xlabel('Время')
    plt.ylabel('x2(t)')
    plt.title('Решение системы: x2(t)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def print_results_comparison(time_points, rkf45_solution, adams_solutions, h_crit):
    print("\nСравнение результатов в точках t = 0, 0.05, ..., 1.0")
    print("-" * 105)
    print("{:^6s} | {:^15s} | {:^15s} | {:^15s} | {:^15s} | {:^15s} | {:^15s} | {:^15s}"
          .format("t", "RKF45 x1", "Адамс(0.025) x1", "Адамс(0.01) x1", f"Адамс({h_crit:.4f}) x1",
                  "RKF45 x2", "Адамс(0.025) x2", "Адамс(0.01) x2", f"Адамс({h_crit:.4f}) x2"))
    print("-" * 105)

    for i, t in enumerate(time_points):
        idx1 = np.argmin(np.abs(adams_solutions[0][0] - t))
        idx2 = np.argmin(np.abs(adams_solutions[1][0] - t))
        idx3 = np.argmin(np.abs(adams_solutions[2][0] - t))
        print("{:^6.2f} | {:^15.6f} | {:^15.6f} | {:^15.6f} | {:^15.6f} | {:^15.6f} | {:^15.6f} | {:^15.6f}"
              .format(t, rkf45_solution.y[0][i],
                      adams_solutions[0][1][idx1], adams_solutions[1][1][idx2], adams_solutions[2][1][idx3],
                      rkf45_solution.y[1][i],
                      adams_solutions[0][2][idx1], adams_solutions[1][2][idx2], adams_solutions[2][2][idx3]))
    print("-" * 105)


def main():
    initial_state = [-3, 1]
    time_interval = [0, 1]
    output_times = np.arange(0, 1.0001, 0.05)

    h_crit = estimate_critical_step_size()
    print(f"\nИспользуемый критический шаг: {h_crit:.6f}")

    rkf45_solution = solve_ivp(
        system_of_equations,
        time_interval,
        initial_state,
        method='RK45',
        t_eval=output_times,
        rtol=1e-6,
        atol=1e-8
    )

    adams_solution_025 = adams_bashforth_second_order(
        system_of_equations,
        time_interval,
        initial_state,
        step_size=0.025,
        output_step=0.05
    )

    adams_solution_01 = adams_bashforth_second_order(
        system_of_equations,
        time_interval,
        initial_state,
        step_size=0.01,
        output_step=0.05
    )

    adams_solution_crit = adams_bashforth_second_order(
        system_of_equations,
        time_interval,
        initial_state,
        step_size=h_crit,
        output_step=0.05
    )

    plot_results(rkf45_solution, [adams_solution_025, adams_solution_01, adams_solution_crit], h_crit)
    print_results_comparison(
        output_times,
        rkf45_solution,
        [adams_solution_025, adams_solution_01, adams_solution_crit],
        h_crit
    )


if __name__ == "__main__":
    main()