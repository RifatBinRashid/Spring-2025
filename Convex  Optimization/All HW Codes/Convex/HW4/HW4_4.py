import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For smoothing

# Gradient Descent Function
def gradient_descent(grad_func, f, x0, step_flag, alpha, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns

    for k in range(max_iteration):
        grad = grad_func(x)  # Compute gradient

        # Check stopping criterion
        grad_norm = np.linalg.norm(grad, 2)
        if grad_norm <= tolerance:
            break
        
        # Step Size Selection
        if step_flag == 'fixed':
            t = alpha  # Fixed step size
        elif step_flag == 'backtracking':
            t = 1.0  # Initial step size for backtracking
            while True:
                if f(x - t * grad) <= f(x) - alpha_armijo * t * grad_norm**2:
                    break
                t *= beta_backtracking  # Reduce step size

        # Gradient Descent Update
        x = x - t * grad

        # Store iterate
        X = np.hstack((X, x.reshape(-1, 1)))
        
    return X

# Define Q matrices and functions
Q1 = np.array([[1, 0], [0, 1]])   # Case 1
Q2 = np.array([[10, 0], [0, 1]])  # Case 2

grad_func1 = lambda x: Q1 @ x  # Gradient for Q1
grad_func2 = lambda x: Q2 @ x  # Gradient for Q2

f1 = lambda x: 0.5 * x.T @ Q1 @ x  # Quadratic function for Q1
f2 = lambda x: 0.5 * x.T @ Q2 @ x  # Quadratic function for Q2

# Hyperparameters
x0 = np.array([1, 1])
max_iter = 2000
tolerance = 1e-7
alpha_armijo = 1e-4
beta_backtracking = 0.5

# Run Gradient Descent and Plot Results
Q_matrices = [Q1, Q2]
grad_funcs = [grad_func1, grad_func2]
funcs = [f1, f2]
alpha_sets = [[0.1, 0.5], [0.01, 0.05]]
Q_labels = ['Q1', 'Q2']

figure_number = 1

for i in range(len(Q_matrices)):
    Q = Q_matrices[i]
    grad_func = grad_funcs[i]
    f = funcs[i]
    alphas = alpha_sets[i]
    Q_name = Q_labels[i]

    for j in range(len(alphas)):
        alpha = alphas[j]
        
        # Run gradient descent
        X_fixed = gradient_descent(grad_func, f, x0, 'fixed', alpha, alpha_armijo, beta_backtracking, max_iter, tolerance)
        X_backtracking = gradient_descent(grad_func, f, x0, 'backtracking', alpha, alpha_armijo, beta_backtracking, max_iter, tolerance)
        
        # Compute function values
        f_vals_fixed = np.array([f(X_fixed[:, k]) for k in range(X_fixed.shape[1])])
        f_vals_backtracking = np.array([f(X_backtracking[:, k]) for k in range(X_backtracking.shape[1])])

        # Smooth the data (if enough points are available)
        if len(f_vals_fixed) > 5:  # Ensure enough points for smoothing
            f_vals_fixed_smooth = savgol_filter(f_vals_fixed, window_length=min(5, len(f_vals_fixed)), polyorder=2)
        else:
            f_vals_fixed_smooth = f_vals_fixed  # Skip smoothing if too few points

        if len(f_vals_backtracking) > 5:  # Ensure enough points for smoothing
            f_vals_backtracking_smooth = savgol_filter(f_vals_backtracking, window_length=min(5, len(f_vals_backtracking)), polyorder=2)
        else:
            f_vals_backtracking_smooth = f_vals_backtracking  # Skip smoothing if too few points

        # Semi-log Plot
        plt.figure(figure_number)
        figure_number += 1
        plt.semilogy(range(1, len(f_vals_fixed) + 1), f_vals_fixed_smooth, '-', linewidth=2, label='Fixed Step Size')
        plt.semilogy(range(1, len(f_vals_backtracking) + 1), f_vals_backtracking_smooth, '-', linewidth=2, label='Backtracking')
        plt.title(f'Function Value vs. Iterations ({Q_name}, α={alpha})')
        plt.xlabel('Iteration k')
        plt.ylabel('f(x^(k))')
        plt.legend()
        plt.grid(True)
        plt.show()