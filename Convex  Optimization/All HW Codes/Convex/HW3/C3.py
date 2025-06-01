import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(grad_func, x0, step_type, alpha, max_iter, tol):
    """
    Implements Gradient Descent

    Inputs:
        grad_func - Function handle computing the gradient ∇f(x)
        x0        - Initial point (column vector)
        step_type - 'fixed' or 'variable' step size (currently supports only 'fixed')
        alpha     - Step size value (for fixed step size)
        max_iter  - Maximum number of iterations
        tol       - Tolerance for stopping criterion ||∇f(x)||_2 ≤ tol

    Output:
        X         - Matrix storing all iterates {x^(k)}
    """
    
    # Check if step_type is supported
    if step_type != 'fixed':
        print('Variable step size is currently not supported.')
        return None

    # Initialize
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns

    for k in range(max_iter):
        grad = grad_func(x)  # Compute gradient

        # Check stopping criterion
        if np.linalg.norm(grad, 2) <= tol:
            break

        # Gradient Descent Update
        x = x - alpha * grad

        # Store iterate
        X = np.hstack((X, x.reshape(-1, 1)))

    return X

# Define two different Q matrices
Q1 = np.array([[1, 0], [0, 1]])   # Case 1
Q2 = np.array([[10, 0], [0, 1]])  # Case 2

# Define function handles for gradients
grad_func1 = lambda x: Q1 @ x
grad_func2 = lambda x: Q2 @ x

# Define function handles for quadratic function values
f1 = lambda x: 0.5 * x.T @ Q1 @ x
f2 = lambda x: 0.5 * x.T @ Q2 @ x

# Initial point
x0 = np.array([1, 1])

# Step sizes
alphas_Q1 = [0.1, 0.5]
alphas_Q2 = [0.01, 0.05]

# Maximum iterations and tolerance
max_iter = 1000
tol = 1e-6

# Store cases in lists
Q_matrices = [Q1, Q2]
grad_funcs = [grad_func1, grad_func2]
funcs = [f1, f2]
alpha_sets = [alphas_Q1, alphas_Q2]
Q_labels = ['Q1', 'Q2']

# Initialize figure counter
figure_number = 1

# Run Gradient Descent and Plot Results
for i in range(len(Q_matrices)):
    Q = Q_matrices[i]
    grad_func = grad_funcs[i]
    f = funcs[i]
    alphas = alpha_sets[i]
    Q_name = Q_labels[i]

    for j in range(len(alphas)):
        alpha = alphas[j]

        # Run gradient descent
        X = gradient_descent(grad_func, x0, 'fixed', alpha, max_iter, tol)

        # Compute function values and gradient norms
        num_iters = X.shape[1]
        f_vals = np.zeros(num_iters)
        grad_norms = np.zeros(num_iters)

        for k in range(num_iters):
            f_vals[k] = f(X[:, k])
            grad_norms[k] = np.linalg.norm(grad_func(X[:, k]), 2)

        # (a) Contour Plot with Iterates
        plt.figure(figure_number)
        figure_number += 1
        X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 30), np.linspace(-1.5, 1.5, 30))
        F_vals = np.array([[f(np.array([x1, x2])) for x1, x2 in zip(np.ravel(X1), np.ravel(X2))]]).reshape(X1.shape)
        plt.contour(X1, X2, F_vals, 20)
        plt.plot(X[0, :], X[1, :], '-o', linewidth=2, markersize=5)
        plt.title(f'Contour of f(x) with Iterates ({Q_name}, α={alpha})')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.grid(True)
        plt.show()

        # (b) Function Value vs. Iterations
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(range(1, num_iters + 1), f_vals, '-o', linewidth=2)
        plt.title(f'Function Value vs. Iterations ({Q_name}, α={alpha})')
        plt.xlabel('Iteration k')
        plt.ylabel('f(x^(k))')
        plt.grid(True)
        plt.show()

        # (c) Gradient Norm vs. Iterations
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(range(1, num_iters + 1), grad_norms, '-s', linewidth=2)
        plt.title(f'Gradient Norm vs. Iterations ({Q_name}, α={alpha})')
        plt.xlabel('Iteration k')
        plt.ylabel('||∇f(x^(k))||_2')
        plt.grid(True)
        plt.show()