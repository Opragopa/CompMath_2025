import math
import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve, norm

A = np.array([
    [0, 6, -6, -4, -3, -8, -5, 5],
    [6, -13, -3, 5, 4, 3, 1, 7],
    [5, -5, -1, 7, 2, 0, 7, 1],
    [5, -5, 5, 6, 4, -7, 4, 0],
    [4, 4, 7, -4, 9, -8, -8, -4],
    [-4, 5, -4, 1, 0, 12, 0, 6],
    [-3, -2, -4, 2, -8, -3, 16, 4],
    [7, 5, 0, 2, 0, -6, 8, -12]
], dtype=float)

p_values = [1.0, 0.1, 0.01, 0.0001, 0.000001]
b = np.array([0, 133, 110, 112, 17, 32, 13, -18], dtype=float)

for p in p_values:
    A_modified = A.copy()
    A_modified[0, 0] = p - 29.0
    b[0] = 4.0 * p - 175.0
    x1 = solve(A_modified, b)
    ATA = np.dot(A_modified.T, A_modified)
    ATb = np.dot(A_modified.T, b)
    lu, piv = lu_factor(ATA)
    x2 = lu_solve((lu, piv), ATb)
    delta = norm(x1 - x2) / norm(x1)
    cond = np.linalg.cond(A_modified)

    print("p = ", p, ":")
    print("x1 = ", x1)
    print("x2 = ", x2)
    print("delta = ", np.round(delta,6))
    print("cond = ", np.round(cond,6))
    print()

    A_modified.fill(0)
