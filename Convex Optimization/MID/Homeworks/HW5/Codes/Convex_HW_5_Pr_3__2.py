import numpy as np
import matplotlib.pyplot as plt

# Step_1: Gradient descent function with backtracking line search
def gradient_descent(grad_func, f, x0, A, step_flag, alpha, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    # Initialize
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns
    f_vals = [f(x, A)]  # Store function values
    grad_norms = []  # Store gradient norms
    step_lengths = []  # Store step lengths

    for k in range(max_iteration):
        grad = grad_func(x, A)  # Compute gradient using the provided grad_func

        # Check stopping criterion
        grad_norm = np.linalg.norm(grad, 2)
        grad_norms.append(grad_norm)

        # Check if the gradient norm is below the tolerance, break if not. 
        if grad_norm <= tolerance:
            iteration_count = k + 1  # Store the number of iterations
            break
        
        # Step Size Selection
        if step_flag == 'fixed':
            t = alpha  # Use the provided fixed step size
        elif step_flag == 'backtracking':
            t = 1.0  # Start with an initial step size
            while True:
                x_new = x - t * grad
                if f(x_new, A) <= f(x, A) - alpha_armijo * t * grad_norm**2:
                    break
                t *= beta_backtracking  # Reduce step size by beta

        # Store step length
        step_lengths.append(t)

        # Gradient Descent Update
        x = x - t * grad

        # Project x back into the feasible set
        x = np.clip(x, -1 + 1e-10, 1 - 1e-10)  # Ensure |x_i| < 1
        if np.any(A @ x >= 1):
            x = x / (np.max(A @ x) * 0.99)  # Ensure a_i^T x < 1

        # Store iterate and function value
        X = np.hstack((X, x.reshape(-1, 1)))
        f_vals.append(f(x, A))
        
    # If the loop completes without breaking, set iteration_count to max_iteration
    if grad_norm > tolerance:
        iteration_count = max_iteration

    return X, f_vals, grad_norms, step_lengths, iteration_count

# Step_2: Newton's method function with backtracking line search
def newton_method(f, grad_func, hessian_func, x0, A, alpha_armijo, beta_backtracking, max_iteration, tolerance):
    # Initialize
    x = x0
    X = x.reshape(-1, 1)  # Store iterates as columns
    f_vals = [f(x, A)]  # Store function values
    newton_decrements = []  # Store Newton decrements
    step_lengths = []  # Store step lengths

    for k in range(max_iteration):
        grad = grad_func(x, A)  # Compute gradient
        H = hessian_func(x, A)  # Compute Hessian

        # Compute Newton decrement directly
        H_inv = np.linalg.inv(H)  # Compute inverse of Hessian
        quadratic_form = grad.T @ H_inv @ grad  # ∇f(x)^T ∇²f(x)^{-1} ∇f(x)
        newton_decrement_sq = quadratic_form  # λ^2 = ∇f(x)^T ∇²f(x)^{-1} ∇f(x)
        newton_decrements.append(newton_decrement_sq)

        # Check stopping criterion
        if newton_decrement_sq <= tolerance:
            iteration_count = k + 1  # Store the number of iterations
            break
        
        # Newton direction
        delta_x = H_inv @ (-grad)  # Compute Δx = H^{-1} (-∇f(x))
        
        # Backtracking line search for Newton's method
        t = 1.0  # Start with an initial step size
        while True:
            x_new = x + t * delta_x  # Newton update: x_new = x + t * Δx
            if f(x_new, A) <= f(x, A) + alpha_armijo * t * (grad.T @ delta_x):
                break
            t *= beta_backtracking  # Reduce step size by beta

        # Store step length
        step_lengths.append(t)

        # Update x
        x = x + t * delta_x 

        # Project x back into the feasible set
        x = np.clip(x, -1 + 1e-10, 1 - 1e-10)  # Ensure |x_i| < 1
        if np.any(A @ x >= 1):
            x = x / (np.max(A @ x) * 0.99)  # Ensure a_i^T x < 1

        # Store iterate and function value
        X = np.hstack((X, x.reshape(-1, 1)))
        f_vals.append(f(x, A))
        
    # If the loop completes without breaking, set iteration_count to max_iteration
    if newton_decrement_sq > tolerance:
        iteration_count = max_iteration

    return X, f_vals, newton_decrements, step_lengths, iteration_count

# Step_3: Define the objective function
def f(x, A):
    # Ensure x is within the domain
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.inf  # Return infinity if x is outside the domain
    
    # First term: -sum(log(1 - a_i^T x))
    term1 = -np.sum(np.log(1 - A @ x))
    
    # Second term: -sum(log(1 - x_i^2))
    term2 = -np.sum(np.log(1 - x**2))
    
    return term1 + term2

# Step_4: Define the gradient of the objective function
def grad_f(x, A):
    # Ensure x is within the domain
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return np.zeros_like(x)  # Return zero gradient if x is outside the domain
    
    # Gradient of the first term: sum(a_i / (1 - a_i^T x))
    grad_term1 = np.sum(A.T / (1 - A @ x), axis=1)
    
    # Gradient of the second term: 2x / (1 - x^2)
    grad_term2 = 2 * x / (1 - x**2)
    
    return grad_term1 + grad_term2

# Step_5: Define the Hessian of the objective function
def hessian_f(x, A):
    n = len(x)
    H = np.zeros((n, n))
    
    # Ensure x is within the domain
    if np.any(np.abs(x) >= 1) or np.any(A @ x >= 1):
        return H  # Return zero Hessian if x is outside the domain
    
    # Hessian of the first term: sum((a_i a_i^T) / (1 - a_i^T x)^2)
    for i in range(A.shape[0]):
        ai = A[i, :]
        H += np.outer(ai, ai) / (1 - ai @ x)**2
    
    # Hessian of the second term: 2 * diag((1 + x_i^2) / (1 - x_i^2)^2)
    H += 2 * np.diag((1 + x**2) / (1 - x**2)**2)
    
    return H

# Step_6: Generate problem instances
n = 10  # Dimension of x
m = 20  # Number of constraints
A = np.random.randn(m, n)  # Randomly generate A matrix
x0 = np.zeros(n)  # Initial point

# Step_7: Hyperparameters
alpha = 0.1  # Fixed step size (not used in backtracking)
max_iter = 1000  # Maximum iterations
tolerance = 1e-7  # Stopping criterion

# Define cases for backtracking parameters
cases = [
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.9},  # Case 1
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.1},  # Case 2
    {"alpha_armijo": 1e-4, "beta_backtracking": 0.5},  # Case 3
    {"alpha_armijo": 1e-1, "beta_backtracking": 0.5},  # Case 4
    {"alpha_armijo": 1e-9, "beta_backtracking": 0.5},  # Case 5
]

# Step_8: Run Gradient Descent and Newton's Method for each case
for i, case in enumerate(cases):
    alpha_armijo = case["alpha_armijo"]
    beta_backtracking = case["beta_backtracking"]
    
    print(f"Case {i+1}: alpha_armijo = {alpha_armijo}, beta_backtracking = {beta_backtracking}")
    
    # Run Gradient Descent with Backtracking
    X_gd, f_vals_gd, grad_norms_gd, step_lengths_gd, gd_iterations = gradient_descent(
        grad_f, f, x0, A, 'backtracking', alpha, alpha_armijo, beta_backtracking, max_iter, tolerance
    )
    print(f"Gradient Descent: Number of iterations = {gd_iterations}")
    
    # Run Newton's Method with Backtracking
    X_newton, f_vals_newton, newton_decrements_newton, step_lengths_newton, newton_iterations = newton_method(
        f, grad_f, hessian_f, x0, A, alpha_armijo, beta_backtracking, max_iter, tolerance
    )
    print(f"Newton's Method: Number of iterations = {newton_iterations}")

    # Step_9: Plot Results
    plt.figure(figsize=(12, 5))

    # (a) Objective Function Value vs. Iteration Number
    plt.subplot(1, 2, 1)
    plt.plot(range(len(f_vals_gd)), f_vals_gd, '-o', linewidth=2, label='Gradient Descent')
    plt.plot(range(len(f_vals_newton)), f_vals_newton, '-s', linewidth=2, label='Newton\'s Method')
    plt.title(f'Objective Function Value vs. Iteration Number\n(alpha_armijo={alpha_armijo}, beta_backtracking={beta_backtracking})')
    plt.xlabel('Iteration k')
    plt.ylabel('f(x^(k))')
    plt.legend()
    plt.grid(True)

    # (b) Step Length vs. Iteration Number
    plt.subplot(1, 2, 2)
    plt.plot(range(len(step_lengths_gd)), step_lengths_gd, '-o', linewidth=2, label='Gradient Descent')
    plt.plot(range(len(step_lengths_newton)), step_lengths_newton, '-s', linewidth=2, label='Newton\'s Method')
    plt.title(f'Step Length vs. Iteration Number\n(alpha_armijo={alpha_armijo}, beta_backtracking={beta_backtracking})')
    plt.xlabel('Iteration k')
    plt.ylabel('Step Length ||Δx||')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()