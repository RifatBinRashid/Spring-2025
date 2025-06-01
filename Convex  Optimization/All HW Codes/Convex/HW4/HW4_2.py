import numpy as np
import matplotlib.pyplot as plt

# Global variable to store the number of iterations
iteration_count = 0

# Step_1: Define the gradient descent function with backtracking line search
def gradient_descent(grad_func, f, x0, step_flag, step_size, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    global iteration_count  # Declare the global variable first
    
    # Initialize
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns

    # Initialize step size for backtracking (only once)
    

    for k in range(max_iteration):
        grad = grad_func(x)  # Compute gradient using the provided grad_func

        # Check stopping criterion
        grad_norm = np.linalg.norm(grad, 2)
        if grad_norm <= tolerance:
            iteration_count = k + 1  # Update the global variable
            break

        # Step Size Selection
        if step_flag == 'fixed':
            t = step_size  # Use the provided fixed step size
        elif step_flag == 'backtracking':
            t = 1.0  # Initial step size for backtracking
            # Backtracking line search
            while True:
                if f(x - t * grad) <= f(x) - alpha_armijo * t * grad_norm**2:
                    break
                t *= beta_backtracking  # Reduce step size by beta

        # Gradient Descent Update
        x = x - t * grad

        # Store iterate
        X = np.hstack((X, x.reshape(-1, 1)))
        
    # If the loop completes without breaking, set iteration_count to max_iteration
    if grad_norm > tolerance:
        iteration_count = max_iteration

    return X

# Step_2: Define the quadratic function and gradient
# Define two different Q matrices
Q1 = np.array([[1, 0], [0, 1]])   # Case 1
Q2 = np.array([[10, 0], [0, 1]])  # Case 2

# Step sizes for fixed step size
alphas_Q1 = [0.1, 0.5]
alphas_Q2 = [0.01, 0.05]

# Functions for gradients
def grad_func1(x):
    return Q1 @ x  # Gradient for Q1

def grad_func2(x):
    return Q2 @ x  # Gradient for Q2

# Functions for quadratic function values
def f1(x):
    return 0.5 * x.T @ Q1 @ x  # Quadratic function for Q1

def f2(x):
    return 0.5 * x.T @ Q2 @ x  # Quadratic function for Q2

Q_matrices = [Q1, Q2]
grad_funcs = [grad_func1, grad_func2]
funcs = [f1, f2]
alpha_sets = [alphas_Q1, alphas_Q2]
Q_labels = ['Q1', 'Q2']

# Step_3: Hyperparameters
# Initial point
x0 = np.array([1, 1])

# Maximum iterations and tolerance
max_iter = 2000
tolerance = 1e-7

# Backtracking parameters
alpha_armijo = 1e-4  # Sufficient reduction parameter for Armijo condition
beta_backtracking = 0.5  # Backtracking parameter

# Step_4: Run Gradient Descent and Plot Results
# Initialize figure counter
figure_number = 1

for i in range(len(Q_matrices)):
    Q = Q_matrices[i]
    grad_func = grad_funcs[i]
    f = funcs[i]
    alphas = alpha_sets[i]
    Q_name = Q_labels[i]

    for j in range(len(alphas)):
        step_size = alphas[j]  # Fixed step size
        # Reset the global variable before each run
        iteration_count = 0
        
        # Run gradient descent with fixed step size
        X_fixed = gradient_descent(grad_func, f, x0, 'fixed', step_size, alpha_armijo, beta_backtracking, max_iter, tolerance)
        fixed_iterations = iteration_count
        
        # Reset the global variable before each run
        iteration_count = 0
        
        # Run gradient descent with backtracking line search
        X_backtracking = gradient_descent(grad_func, f, x0, 'backtracking', None, alpha_armijo, beta_backtracking, max_iter, tolerance)
        backtracking_iterations = iteration_count
        
        # Print the number of iterations for this case
        print(f'Case: {Q_name}, Step size (α): {step_size}, Iterations (Fixed): {fixed_iterations}')
        print(f'Case: {Q_name}, Backtracking, Iterations: {backtracking_iterations}')

        # Compute function values and gradient norms for fixed step size
        num_iters_fixed = X_fixed.shape[1]
        f_vals_fixed = np.zeros(num_iters_fixed)
        grad_norms_fixed = np.zeros(num_iters_fixed)

        for k in range(num_iters_fixed):
            f_vals_fixed[k] = f(X_fixed[:, k])
            grad_norms_fixed[k] = np.linalg.norm(grad_func(X_fixed[:, k]), 2)

        # Compute function values and gradient norms for backtracking
        num_iters_backtracking = X_backtracking.shape[1]
        f_vals_backtracking = np.zeros(num_iters_backtracking)
        grad_norms_backtracking = np.zeros(num_iters_backtracking)

        for k in range(num_iters_backtracking):
            f_vals_backtracking[k] = f(X_backtracking[:, k])
            grad_norms_backtracking[k] = np.linalg.norm(grad_func(X_backtracking[:, k]), 2)

        # (a) Contour Plot with Iterates
        plt.figure(figure_number)
        figure_number += 1

        # Create a grid for the contour plot
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
        X1, X2 = np.meshgrid(x1, x2)
        F = np.zeros_like(X1)

        # Evaluate the function over the grid
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                F[i, j] = f(np.array([X1[i, j], X2[i, j]]))

        # Plot the contour lines
        plt.contour(X1, X2, F, levels=20, cmap='viridis')
        plt.colorbar(label='f(x)')

        # Plot the iterates for fixed step size
        plt.plot(X_fixed[0, :], X_fixed[1, :], '-o', linewidth=2, markersize=5, label='Fixed Step Size')

        # Plot the iterates for backtracking line search
        plt.plot(X_backtracking[0, :], X_backtracking[1, :], '-s', linewidth=2, markersize=5, label='Backtracking')

        # Add labels, title, and legend
        plt.title(f'Contour Plot of f(x) with Iterates ({Q_name}, α={step_size})')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend()
        plt.grid(True)
        plt.show()

        # (b) Function Value vs. Iterations (Semi-log Scale)
        plt.figure(figure_number)
        figure_number += 1
        plt.semilogy(range(1, num_iters_fixed + 1), f_vals_fixed, '-o', linewidth=2, label='Fixed Step Size')
        plt.semilogy(range(1, num_iters_backtracking + 1), f_vals_backtracking, '-s', linewidth=2, label='Backtracking')
        plt.title(f'Function Value vs. Iterations ({Q_name}, α={step_size})')
        plt.xlabel('Iteration k')
        plt.ylabel('f(x^(k))')
        plt.legend()
        plt.grid(True)
        plt.show()

        # (c) Gradient Norm vs. Iterations
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(range(1, num_iters_fixed + 1), grad_norms_fixed, '-o', linewidth=2, label='Fixed Step Size')
        plt.plot(range(1, num_iters_backtracking + 1), grad_norms_backtracking, '-s', linewidth=2, label='Backtracking')
        plt.title(f'Gradient Norm vs. Iterations ({Q_name}, α={step_size})')
        plt.xlabel('Iteration k')
        plt.ylabel('||∇f(x^(k))||_2')
        plt.legend()
        plt.grid(True)
        plt.show()