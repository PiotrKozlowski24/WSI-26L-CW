from autograd import grad
import matplotlib.pyplot as plt
import numpy as np

def grad_descent(func: callable, x: np.array, alpha: float):
    ITER_MAX = 100
    EPSILON = 1e-6

    trajectory = [x.copy()]
    grad_f = grad(func)

    for _ in range(ITER_MAX):
        g = grad_f(x)
        if np.linalg.norm(g) < EPSILON:
            break
        x = x - alpha * g
        trajectory.append(x.copy())
    
    return np.array(trajectory)

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
    # testy dla różnych kroków
    x0 = np.array([8.0, -6.0])
    for alpha in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]:
        traj1 = grad_descent(sum_of_sqrs, x0, alpha)
        visualize_fun(sum_of_sqrs, traj1)
    
    # testy dla różnych punktów początkowych
    x01 = np.array([8.0, -6.0])
    x02 = np.array([-10.0, 10.0])
    x03 = np.array([2.0, 1.0])
    alpha = 0.1

    for x0 in [x01, x02, x03]:
        traj1 = grad_descent(sum_of_sqrs, x0, alpha)
        visualize_fun(sum_of_sqrs, traj1)
