"""
WSI 26L - ćwiczenie 1 - Przeszukiwanie przestrzeni
Autor: Piotr Kozłowski
"""

from autograd import grad
import matplotlib.pyplot as plt
import numpy as np

def grad_descent(func: callable, x: np.array, lr: float):
    ITER_MAX = 1000
    EPSILON = 1e-6

    trajectory = [x.copy()]
    grad_f = grad(func)

    for _ in range(ITER_MAX):
        g = grad_f(x)
        if np.linalg.norm(g) < EPSILON:
            break
        x = x - lr * g
        trajectory.append(x.copy())
    
    return np.array(trajectory), x

def sum_of_sqrs(x: np.array):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x2**2

def matyas_func(x: np.array):
    x1 = x[0]
    x2 = x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def visualize_fun(obj_fun: callable, trajectory: np.ndarray):
    min_x, min_y = trajectory[-1]
    MIN_X = 10
    MAX_X = 10
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = obj_fun(np.array([X1, X2]))

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Objective Function Value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')

    plt.plot(trajectory[:, 0], trajectory[:, 1],
             marker='o', color='red',
             label='Gradient Descent Steps', alpha=0.5)
    
    plt.scatter(min_x, min_y, color='yellow',
                label='Minimum found by gradient descent alg.')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # funkcja_celu = sum_of_sqrs
    funkcja_celu = sum_of_sqrs

    # testy dla różnych kroków
    x0 = np.array([8.012, -6.430])
    for lr in [0.001, 0.01, 0.1, 0.25, 0.49, 0.5, 0.501, 0.75, 1, 1.1, 1.25, 1.5, 1.9, 1.91, 1.92, 1.93]:
        traj1, min = grad_descent(funkcja_celu, x0, lr)
        print(f"Learning rate: {lr}, ilość iteracji: {traj1.shape[0]-1}, znalezione minimum [x1: {min[0]}, x2: {min[1]}]")
        visualize_fun(funkcja_celu, traj1)
    
    # # testy dla różnych punktów początkowych
    # x01 = np.array([8.012, 6.430])
    # x02 = np.array([-10.0, 1.0])
    # x03 = np.array([2.034, 1.054532])
    # lr = 0.5

    # for x0 in [x01, x02, x03]:
    #     traj1, min = grad_descent(funkcja_celu, x0, lr)
    #     print(f"Punkt startowy: [{x0[0]} {x0[1]}], ilość iteracji: {traj1.shape[0]-1}, znalezione minimum [x1: {min[0]}, x2: {min[1]}]")
    #     visualize_fun(funkcja_celu, traj1)
