import numpy as np
import matplotlib.pyplot as plt

# Step_1: Gradient descent function with backtracking line search
def gradient_descent(grad_func, f, x0, A, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    x = x0
    X = x.reshape(-1, 1)  # Store iterates
    f_vals = [f(x, A)]  # Store function values
    grad_norms = []  # Store gradient norms
    step_lengths = []  # Store step lengths

    for k in range(max_iteration):
        grad = grad_func(x, A)  # Compute gradient
        grad_norm = np.linalg.norm(grad, 2)
        grad_norms.append(grad_norm)

        if grad_norm <= tolerance:
            break  # Stop if gradient norm is below tolerance
        
        # Backtracking Line Search
        t = 1.0  
        while f(x - t * grad, A) > f(x, A) - alpha_armijo * t * grad_norm**2:
            t *= beta_backtracking  
        step_lengths.append(t)

        # Gradient Descent Update
        x = x - t * grad

        # Project x back into the feasible set
        x = np.clip(x, -0.99, 0.99)  
        if np.any(A @ x >= 1):
            x = x / (np.max(A @ x) * 1.01)  

        X = np.hstack((X, x.reshape(-1, 1)))
        f_vals.append(f(x, A))
        
    return X, f_vals, step_lengths, k+1

# Step_2: Newton's method function with backtracking line search
def newton_method(f, grad_func, hessian_func, x0, A, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    x = x0
    X = x.reshape(-1, 1)  
    f_vals = [f(x, A)]  
    step_lengths = []  

    for k in range(max_iteration):
        grad = grad_func(x, A)  
        H = hessian_func(x, A)  
        
        H_inv = np.linalg.inv(H)  
        delta_x = -H_inv @ grad  

        # Backtracking Line Search
        t = 1.0  
        while f(x + t * delta_x, A) > f(x, A) + alpha_armijo * t * (grad.T @ delta_x):
            t *= beta_backtracking  
        step_lengths.append(t)

        # Newton Update
        x = x + t * delta_x  

        # Project x back into the feasible set
        x = np.clip(x, -0.99, 0.99)  
        if np.any(A @ x >= 1):
            x = x / (np.max(A @ x) * 1.01)  

        X = np.hstack((X, x.reshape(-1, 1)))
        f_vals.append(f(x, A))
        
    return X, f_vals, step_lengths, k+1

# Step_3: Define the objective function
def f(x, A):
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.inf  
    return -np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - x**2))

# Step_4: Define the gradient
def grad_f(x, A):
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.zeros_like(x)  
    grad_term1 = np.sum(A.T / (1 - A @ x), axis=1)
    grad_term2 = 2 * x / (1 - x**2)
    return grad_term1 + grad_term2

# Step_5: Define the Hessian
def hessian_f(x, A):
    n = len(x)
    H = np.zeros((n, n))
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return H  
    for i in range(A.shape[0]):
        ai = A[i, :]
        H += np.outer(ai, ai) / (1 - ai @ x)**2
    H += 2 * np.diag((1 + x**2) / (1 - x**2)**2)
    return H

# Step_6: Generate problem instances
np.random.seed(42)  
n = 10  
m = 20  
A = np.random.randn(m, n)  
x0 = np.zeros(n)  

# Step_7: Hyperparameters
max_iter = 1000  
tolerance = 1e-7  

# Cases for backtracking parameters
cases = [
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.9},  
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.1},  
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.5},  
    {"alpha_armijo": 1e-1, "beta_backtracking": 0.5},  
    {"alpha_armijo": 1e-9, "beta_backtracking": 0.5},  
]

# Step_8: Run Gradient Descent and Newton's Method for each case
for i, case in enumerate(cases):
    alpha_armijo = case["alpha_armijo"]
    beta_backtracking = case["beta_backtracking"]
    
    print(f"Case {i+1}: alpha_armijo = {alpha_armijo}, beta_backtracking = {beta_backtracking}")
    
    X_gd, f_vals_gd, step_lengths_gd, gd_iterations = gradient_descent(
        grad_f, f, x0, A, alpha_armijo, beta_backtracking, max_iter, tolerance
    )
    print(f"Gradient Descent: Number of iterations = {gd_iterations}")
    
    X_newton, f_vals_newton, step_lengths_newton, newton_iterations = newton_method(
        f, grad_f, hessian_f, x0, A, alpha_armijo, beta_backtracking, max_iter, tolerance
    )
    print(f"Newton's Method: Number of iterations = {newton_iterations}")

    # Step_9: Plot Results
    plt.figure(figsize=(12, 5))

    # (a) Objective Function Value vs. Iteration Number
    plt.subplot(1, 2, 1)
    plt.plot(range(len(f_vals_gd)), f_vals_gd, '-o', linewidth=2, label='Gradient Descent')
    plt.plot(range(len(f_vals_newton)), f_vals_newton, '-s', linewidth=2, label="Newton's Method")
    plt.title(f'Objective Function Value vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    # (b) Step Length vs. Iteration Number
    plt.subplot(1, 2, 2)
    plt.plot(range(len(step_lengths_gd)), step_lengths_gd, '-o', linewidth=2, label='Gradient Descent')
    plt.plot(range(len(step_lengths_newton)), step_lengths_newton, '-s', linewidth=2, label='Newton\'s Method')
    plt.title(f'Step Length vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Step Length')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
