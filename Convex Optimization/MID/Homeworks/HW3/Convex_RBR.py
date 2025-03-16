import numpy as np
import matplotlib.pyplot as plt
# Global variable to store the number of iterations
iteration_count = 0

#Step_1: Defining the gradient descent function
def gradient_descent(grad_func, x0, step_flag, alpha, max_iteration, tolerance):
    
    global iteration_count  # Declare the global variable first
    #Check if a non-zero step size is given. 
    if alpha <= 0:
        print('Step size alpha must be greater than zero.')
        return None

    # Check if step_type is supported
    if step_flag != 'fixed':
        print('Variable step size is currently not supported.')
        return None

    # Initialize
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns

    for k in range(max_iteration):
        grad = grad_func(x)  # Compute gradient using the provided grad_func

        # Check stopping criterion
        grad_norm = 0.0
        for g in grad:
            grad_norm += g**2
        grad_norm = grad_norm**0.5  # Take the square root

        # Check if the gradient norm is below the tolerance,break if not. 
        if grad_norm <= tolerance:
            iteration_count = k + 1  # Update the global variable
            break

        # Gradient Descent Update
        x = x - alpha * grad

        # Store iterate
        X = np.hstack((X, x.reshape(-1, 1)))
        
    # If the loop completes without breaking, set iteration_count to max_iteration
    if grad_norm > tolerance:
        iteration_count = max_iteration

    return X

#Step_2: Taking the inputs as mentioned in question
# Define two different Q matrices
Q1 = np.array([[1, 0], [0, 1]])   # Case 1
Q2 = np.array([[10, 0], [0, 1]])  # Case 2

# Step sizes
alphas_Q1 = [0.1, 0.5]
alphas_Q2 = [0.01, 0.05]

#Step_3: Function and gradient function 
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

#Step_4: Hyperparameters
# Initial point
x0 = np.array([1, 1])

# Maximum iterations and tolerance
max_iter = 2000
tolerance = 1e-7

#Step_5: Run Gradient Descent and Plot Results
# Initialize figure counter
figure_number = 1

for i in range(len(Q_matrices)):
    Q = Q_matrices[i]
    grad_func = grad_funcs[i]
    f = funcs[i]
    alphas = alpha_sets[i]
    Q_name = Q_labels[i]

    for j in range(len(alphas)):
        alpha = alphas[j]
        # Reset the global variable before each run
        iteration_count
        iteration_count = 0
        
        # Run gradient descent
        X = gradient_descent(grad_func, x0, 'fixed', alpha, max_iter, tolerance)
        
        # Print the number of iterations for this case
        print(f'Case: {Q_name}, Step size (α): {alpha}, Iterations: {iteration_count}')

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