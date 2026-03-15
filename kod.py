"""
WSI 26L - ćwiczenie 1 - Przeszukiwanie przestrzeni
Autor: Piotr Kozłowski
"""

from autograd import grad
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(100)

def grad_descent(func: callable, x: np.array, lr: float, EPSILON=1e-6, ITER_MAX=1000):
    trajectory = [x.copy()]
    grad_f = grad(func)

    for _ in range(ITER_MAX):
        g = grad_f(x)
        if np.linalg.norm(g) < EPSILON:
            break
        x = x - lr * g
        trajectory.append(x.copy())
    
    return np.array(trajectory), x

def sgd(func: callable, x: np.array, lr: float, batch_size: int = 1, EPSILON=1e-6, ITER_MAX=1000):
    trajectory = [x.copy()]
    grad_f = grad(func)

    for i in range(1, ITER_MAX + 1):
        g = grad_f(x)
        if np.linalg.norm(g) < EPSILON:
            break
        
        g = g + np.random.normal(0, 1.0 / (np.sqrt(batch_size) * i), size=x.shape)
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

def visualize_fun(obj_fun: callable, trajectory: np.ndarray, ax):
    min_x, min_y = trajectory[-1]
    MIN_X = 10
    MAX_X = 10
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = obj_fun(np.array([X1, X2]))

    pcm = ax.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')

    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        marker='o', color='red',
        alpha=0.5, label="GD steps"
    )

    ax.scatter(min_x, min_y, color='yellow', label="Minimum")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

# BADANIE WPŁYWU HIPERPARAMETRÓW ---------------------------------------------

if __name__ == "__main__":
    TESTS = {
        "1": "Suma kwadratów - różne kroki",
        "2": "Matyas - różne kroki",
        "3": "Suma kwadratów - różne punkty startowe",
        "4": "Matyas - różne punkty startowe",
        "5": "GD vs SGD",
        "6": "SGD - różne rozmiary batcha",
    }

    print("Wybierz test:")
    for key, name in TESTS.items():
        print(f"  {key}. {name}")
    
    choice = input("\nWybór: ").strip()

    if choice == "1":
        # testy dla różnych kroków
        # suma kwadratów
        x0 = np.array([8.012, -6.430])
        lrs = [0.0001, 0.01, 0.1, 0.25, 0.49, 0.5, 0.501, 0.75, 1]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i, lr in enumerate(lrs):
            traj, minimum = grad_descent(sum_of_sqrs, x0, lr)
            mse = np.square(np.subtract(np.array([0,0]), minimum)).mean()

            print(f"Krok: {lr}, ilość iteracji: {traj.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum[0]}, x2: {minimum[1]}], mse: {mse}")

            visualize_fun(sum_of_sqrs, traj, axes[i])
            axes[i].set_title(f"Krok uczenia = {lr}")

        plt.tight_layout()
        plt.show()

    elif choice == "2":
        #funkcja matyasa
        x0 = np.array([8.012, 3.430])
        lrs = [0.01, 0.1, 1, 1.5, 1.89, 1.9, 1.91, 1.98, 1.99]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i, lr in enumerate(lrs):
            traj, minimum = grad_descent(matyas_func, x0, lr)
            mse = np.square(np.subtract(np.array([0,0]), minimum)).mean()

            print(f"Krok: {lr}, ilość iteracji: {traj.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum[0]}, x2: {minimum[1]}], mse: {mse}")

            visualize_fun(matyas_func, traj, axes[i])
            axes[i].set_title(f"Krok uczenia = {lr}")

        plt.tight_layout()
        plt.show()
    
    elif choice == "3":
        # testy dla różnych punktów początkowych
        # suma kwadratów
        x01 = np.array([8.012, 6.430])
        x02 = np.array([-9.0, 1.0])
        x03 = np.array([2.034, 1.054532])
        lr = 0.5

        starts = [x01, x02, x03]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, x0 in enumerate(starts):
            traj, minimum = grad_descent(sum_of_sqrs, x0, lr)
            mse = np.square(np.subtract(np.array([0,0]), minimum)).mean()

            print(f"Punkt startowy: [{x0[0]} {x0[1]}], ilość iteracji: {traj.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum[0]}, x2: {minimum[1]}], mse: {mse}")

            visualize_fun(sum_of_sqrs, traj, axes[i])
            axes[i].set_title(f"Punkt początkowy: ({x0[0]}, {x0[1]})")

        plt.tight_layout()
        plt.show()

    elif choice == "4":
        # funkcja matyasa
        x01 = np.array([8.012, 6.430])
        x02 = np.array([-9.0, 7.0])
        x03 = np.array([2.034, 1.054532])
        lr = 1.9

        starts = [x01, x02, x03]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, x0 in enumerate(starts):
            traj, minimum = grad_descent(matyas_func, x0, lr)
            mse = np.square(np.subtract(np.array([0,0]), minimum)).mean()

            print(f"Punkt startowy: [{x0[0]} {x0[1]}], ilość iteracji: {traj.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum[0]}, x2: {minimum[1]}], mse: {mse}")

            visualize_fun(matyas_func, traj, axes[i])
            axes[i].set_title(f"Punkt początkowy: ({x0[0]}, {x0[1]})")

        plt.tight_layout()
        plt.show()

    elif choice == "5":
        # gd vs sgd
        x0 = np.array([9.0, -2.0])
        funcs = [sum_of_sqrs, matyas_func]
        lr_list = [0.5, 1.9]

        for lr, func in zip(lr_list, funcs):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            traj1, minimum1 = grad_descent(func, x0, lr)
            mse1 = np.square(np.subtract(np.array([0,0]), minimum1)).mean()

            print(f"Punkt startowy: [{x0[0]} {x0[1]}], ilość iteracji: {traj1.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum1[0]}, x2: {minimum1[1]}], mse: {mse1}")

            visualize_fun(func, traj1, axes[0])
            axes[0].set_title(f"GD")

            traj2, minimum2 = sgd(func, x0, lr, 1)
            mse2 = np.square(np.subtract(np.array([0,0]), minimum2)).mean()

            print(f"Punkt startowy: [{x0[0]} {x0[1]}], ilość iteracji: {traj2.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum2[0]}, x2: {minimum2[1]}], mse: {mse2}")

            visualize_fun(func, traj2, axes[1])
            axes[1].set_title(f"SGD")
            plt.tight_layout()
            plt.show()

    elif choice == "6":
        # sgd dla różnych wielkości batchów

        x0 = np.array([9.0, -2.0])
        lr = 1.9
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, batch_size in enumerate([1, 10, 100]):
            traj2, minimum2 = sgd(matyas_func, x0, lr, batch_size)
            mse2 = np.square(np.subtract(np.array([0,0]), minimum2)).mean()

            print(f"Batch size: {batch_size}, ilość iteracji: {traj2.shape[0]-1}, "
                f"znalezione minimum [x1: {minimum2[0]}, x2: {minimum2[1]}], mse: {mse2}")

            visualize_fun(matyas_func, traj2, axes[i])
            axes[i].set_title(f"Wielkość batcha: {batch_size}")
        plt.tight_layout()
        plt.show()
    else: print("Nieznany test...")