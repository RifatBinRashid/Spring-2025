import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_constrained(grad_func, func, x0, step_type, alpha, beta, max_iter, tol, A_vectors):
    X = [x0]  # Store iterates
    x = x0
    
    for k in range(max_iter):
        grad = grad_func(x)
        
        # Stopping condition
        if np.linalg.norm(grad, 2) <= tol:
            break
        
        # Step Size Selection
        if step_type == 'fixed':
            step_size = alpha  # Use fixed step size
            x_new = x - step_size * grad
            
        elif step_type == 'variable':
            # Backtracking line search
            t = 1  # Initial step size
            while func(x - t * grad) > func(x) - alpha * t * np.linalg.norm(grad, 2)**2:
                t = beta * t
            x_new = x - t * grad  # Update with backtracking step
            
        else:
            print('Invalid step type. Use "fixed" or "variable".')
            return
        
        # Projection to Feasible Region (Apply in Both Cases)
        x_new = np.clip(x_new, -0.99, 0.99)  # Ensure |x_i| < 1
        
        # Fix matrix multiplication issue
        while np.any(A_vectors @ x_new >= 1):  # Ensure a_i^T x < 1
            t = beta * t  # Reduce step size
            x_new = x - t * grad  
            x_new = np.clip(x_new, -0.99, 0.99)
        
        # Store New Iterate
        x = x_new
        X.append(x)
    
    return np.array(X).T

def newton_method_constrained(grad_func, hess_func, func, x0, tol, max_iter, A, step_type):
    X = [x0]  # Store iterates
    x = x0
    
    for k in range(max_iter):
        grad = grad_func(x)
        hess = hess_func(x)
        
        # Ensure Hessian is positive definite
        if np.min(np.linalg.eigvals(hess)) < 1e-6:
            break
        
        dx_nt = -np.linalg.solve(hess, grad)  # Newton step
        lambda2 = grad.T @ dx_nt  # Newton decrement
        
        # Stopping condition
        if abs(lambda2) / 2 <= tol:
            break

        # Step size selection
        if step_type == 'fixed':
            t = 1  # Fixed step size
        elif step_type == 'variable':
            # Backtracking Line Search
            t = 1
            alpha_bt = 0.5
            beta_bt = 0.8
            while func(x + t * dx_nt) > func(x) + alpha_bt * t * lambda2:
                t = beta_bt * t
        else:
            raise ValueError('Invalid step type. Use "fixed" or "variable".')
        
        # Update x
        x_new = x + t * dx_nt

        # Projection to Feasible Region
        x_new = np.clip(x_new, -0.99, 0.99)  # Ensuring |x_i| < 1
        while np.any(A @ x_new >= 1):
            t = beta_bt * t
            x_new = x + t * dx_nt
            x_new = np.clip(x_new, -0.99, 0.99)

        x = x_new
        X.append(x)
    
    return np.array(X).T

def problem03():
    np.random.seed(42)  # Fix random seed for reproducibility
    
    # Step 1: Generate Problem Data
    n = 10  # Number of variables
    m = 20  # Number of constraints
    A = np.random.randn(m, n) * 0.5  # Generate ai from normal distribution

    # Define the objective function
    def f(x):
        return -np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - x**2))

    # Define the gradient
    def grad_f(x):
        return (A.T @ (1 / (1 - A @ x))) + (2 * x / (1 - x**2))

    # Define the Hessian for Newton's Method
    def hess_f(x):
        return (A.T @ np.diag(1 / (1 - A @ x)**2) @ A) + np.diag(2 * (1 + x**2) / (1 - x**2)**2)

    # Set Initial Point
    x0 = np.zeros(n)  # x(0) = 0 as given in the problem

    # Step 2: Solve using Gradient Descent with Constraints
    step_type = 'variable'  # Backtracking line search
    alpha = 0.0001  # Armijo parameter
    beta = 0.5  # Backtracking parameter
    max_iter = 500
    tol = 1e-6

    X_gd = gradient_descent_constrained(grad_f, f, x0, step_type, alpha, beta, max_iter, tol, A)
    num_iters_gd = X_gd.shape[1]

    # Step 3: Solve using Newton's Method with Constraints
    X_newton = newton_method_constrained(grad_f, hess_f, f, x0, tol, max_iter, A, step_type)
    num_iters_newton = X_newton.shape[1]

    # Step 4: Compute Optimal Value
    p_star = min(f(X_gd[:, -1]), f(X_newton[:, -1]))

    # Step 5: Plot Results
    plt.figure(figsize=(12, 8))

    # Plot 1: Objective function vs Iterations
    plt.subplot(2, 2, 1)
    plt.plot(range(num_iters_gd), [f(X_gd[:, i]) for i in range(num_iters_gd)], 'b-o', linewidth=1.5, label='Gradient Descent')
    plt.plot(range(num_iters_newton), [f(X_newton[:, i]) for i in range(num_iters_newton)], 'r-s', linewidth=1.5, label='Newton Method')
    plt.title('Objective Function vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()

    # Plot 2: Step length vs Iterations (Gradient Descent)
    step_lengths_gd = np.linalg.norm(np.diff(X_gd, axis=1), axis=0)
    step_lengths_nt = np.linalg.norm(np.diff(X_newton, axis=1), axis=0)
    plt.subplot(2, 2, 2)
    plt.plot(range(len(step_lengths_gd)), step_lengths_gd, 'b-o', linewidth=1.5, label='Gradient Descent')
    plt.plot(range(len(step_lengths_nt)), step_lengths_nt, 'r-s', linewidth=1.5, label='Newton Method')
    plt.title('Step Length vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Step Length')
    plt.grid()

    # Plot 3: f - p* vs Iterations
    plt.subplot(2, 2, 3)
    plt.plot(range(num_iters_gd), [abs(f(X_gd[:, i]) - p_star) for i in range(num_iters_gd)], 'b-o', linewidth=1.5, label='Gradient Descent')
    plt.plot(range(num_iters_newton), [abs(f(X_newton[:, i]) - p_star) for i in range(num_iters_newton)], 'r-s', linewidth=1.5, label='Newton Method')
    plt.title('f - p* vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('f(x) - p*')
    plt.legend()
    plt.grid()

    # Plot 4: Iterations Comparison
    plt.subplot(2, 2, 4)
    plt.bar(['Gradient Descent', 'Newton Method'], [num_iters_gd, num_iters_newton])
    plt.ylabel('Number of Iterations')
    plt.title('Total Iterations Required')
    plt.grid()

    plt.tight_layout()
    plt.show()

    print(f'Gradient Descent Iterations: {num_iters_gd}')
    print(f'Newton Method Iterations: {num_iters_newton}')

# Run the problem
problem03()