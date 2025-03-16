import numpy as np
import matplotlib.pyplot as plt

# Step_1: Gradient descent function with backtracking line search
def gradient_descent(f_grad, func, x0, step_type, alpha, beta, max_iter, tol, A):
    X = [x0]  # List to store iterates
    x = x0.copy()
    
    for k in range(max_iter):
        grad = f_grad(x, A)

        # Stopping criterion based on gradient norm
        if np.linalg.norm(grad, 2) <= tol:
            break

        # Step Size Selection
        if step_type == 'fixed':
            step_size = alpha
            x_new = x - step_size * grad
        elif step_type == 'variable':
            # Backtracking line search for step size
            t = 1  
            while func(x - t * grad, A) > func(x, A) - alpha * t * np.linalg.norm(grad, 2)**2:
                t *= beta
            x_new = x - t * grad
        else:
            raise ValueError('Invalid step type. Choose either "fixed" or "variable".')

        # Project onto the feasible region
        x_new = np.clip(x_new, -0.99, 0.99)  # Maintain |x_i| < 1
        while np.any(A @ x_new >= 1):
            t *= beta
            x_new = x - t * grad
            x_new = np.clip(x_new, -0.99, 0.99)

        # Save new iterate
        x = x_new
        X.append(x)
    
    return np.array(X).T  # Convert to column-wise storage


# Step_2: Newton's method function with backtracking line search
def newton_method(f_grad, f_hess, func, x0, tol, max_iter, A, step_type, alpha, beta):
    X = [x0]  # List to store iterates
    x = x0.copy()
    
    for k in range(max_iter):
        grad = f_grad(x, A)
        hess = f_hess(x, A)

        # Ensure Hessian is positive definite
        if np.min(np.linalg.eigvals(hess)) < 1e-6:
            break
        
        dx_nt = -np.linalg.solve(hess, grad)  # Compute Newton direction
        lambda2 = grad.T @ dx_nt  # Compute Newton decrement
        
        # Stopping condition based on Newton decrement
        if abs(lambda2) / 2 <= tol:
            break

        # Step size selection
        if step_type == 'fixed':
            t = 1  # Use a fixed step size
        elif step_type == 'variable':
            # Backtracking line search for Newton step
            t = 1
            while func(x + t * dx_nt, A) > func(x, A) + alpha * t * lambda2:
                t *= beta
        else:
            raise ValueError('Invalid step type. Choose either "fixed" or "variable".')

        # Apply the step update
        x_new = x + t * dx_nt

        # Project onto the feasible region
        x_new = np.clip(x_new, -0.99, 0.99)  # Maintain |x_i| < 1
        while np.any(A @ x_new >= 1):
            t *= beta
            x_new = x + t * dx_nt
            x_new = np.clip(x_new, -0.99, 0.99)

        x = x_new
        X.append(x)
    
    return np.array(X).T  # Convert to column-wise storage


# Define the objective function
def f(x, A):
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.inf
    return -np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - x**2))

# Define the gradient function
def f_grad(x, A):
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.zeros_like(x)
    grad_term1 = A.T @ (1 / (1 - A @ x))
    grad_term2 = 2 * x / (1 - x**2)
    return grad_term1 + grad_term2

# Define the Hessian function for Newton's Method
def f_hess(x, A):
    n = len(x)
    H = np.zeros((n, n))
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return H
    for i in range(A.shape[0]):
        ai = A[i, :]
        H += np.outer(ai, ai) / (1 - ai @ x) ** 2
    H += np.diag((2 * (1 + x**2)) / (1 - x**2) ** 2)
    return H

# Step_6: Generate problem instances

n = 10  # Number of variables
m = 20  # Number of constraints
np.random.seed(42)  # Fix random seed for reproducibility
A = np.random.randn(m, n) * 0.5  # Generate A matrix

# Step_7: Hyperparameters
# Set Initial Point
x0 = np.zeros(n)  # x(0) = 0 as given in the problem

step_type = 'variable'
alpha = 0.01
beta = 0.2
max_iter = 500
tol = 1e-6

# Step_8: Run Gradient Descent and Newton's Method 

X_gd = gradient_descent(f_grad, f, x0, step_type, alpha, beta, max_iter, tol, A)
num_iters_gd = X_gd.shape[1]

X_newton = newton_method(f_grad, f_hess, f, x0, tol, max_iter, A, step_type, alpha, beta)
num_iters_newton = X_newton.shape[1]

# Compute Optimal Value

p_star = min(f(X_gd[:, -1], A), f(X_newton[:, -1], A))


# Plot Results

plt.figure(figsize=(12, 10))

# (1) **Objective Function vs Iterations**
plt.subplot(2, 2, 1)
plt.plot(range(1, num_iters_gd+1), [f(X_gd[:, i], A) for i in range(num_iters_gd)], 'b-o', linewidth=1.5, label='Gradient Descent')
plt.plot(range(1, num_iters_newton+1), [f(X_newton[:, i], A) for i in range(num_iters_newton)], 'r-s', linewidth=1.5, label='Newton Method')
plt.title('Objective Function vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.legend()
plt.grid()

# (2) **Step Length vs Iterations**
step_lengths_gd = np.linalg.norm(np.diff(X_gd, axis=1), axis=0)
step_lengths_nt = np.linalg.norm(np.diff(X_newton, axis=1), axis=0)

plt.subplot(2, 2, 2)
plt.plot(range(1, len(step_lengths_gd)+1), step_lengths_gd, 'b-o', linewidth=1.5, label='Gradient Descent')
plt.plot(range(1, len(step_lengths_nt)+1), step_lengths_nt, 'r-s', linewidth=1.5, label='Newton Method')
plt.title('Step Length vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Step Length')
plt.legend()
plt.grid()

# (3) **f - p* vs Iterations**
plt.subplot(2, 2, 3)
plt.plot(range(1, num_iters_gd+1), [abs(f(X_gd[:, i], A) - p_star) for i in range(num_iters_gd)], 'b-o', linewidth=1.5, label='Gradient Descent')
plt.plot(range(1, num_iters_newton+1), [abs(f(X_newton[:, i], A) - p_star) for i in range(num_iters_newton)], 'r-s', linewidth=1.5, label='Newton Method')
plt.title('f - p* vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('f(x) - p*')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f'Gradient Descent Iterations: {num_iters_gd}')
print(f'Newton Method Iterations: {num_iters_newton}')
