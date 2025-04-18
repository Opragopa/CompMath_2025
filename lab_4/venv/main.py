import math


def taylor_exp(x, epsilon=1e-5):
    result = 0.0
    term = 1.0  # (n=0): x^0 / 0! = 1
    n = 1

    while abs(term) > epsilon:
        result += term
        term *= x / n
        n += 1

    return result

x_values = [0.5, -0.5, -10.5]

print("{:^10} | {:^15} | {:^15} | {:^15}".format("x", "Taylor", "Math.exp", "Разница"))
print("-" * 60)

for x in x_values:
    taylor_result = taylor_exp(x)
    math_result = math.exp(x)
    difference = abs(taylor_result - math_result)
    print("{:^10.2f} | {:^15.8f} | {:^15.8f} | {:^15.8f}".format(
        x, taylor_result, math_result, difference))