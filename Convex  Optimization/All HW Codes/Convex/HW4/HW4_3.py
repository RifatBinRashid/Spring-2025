import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(Q, x0, fixed_step=True, step_size=0.1, max_iters=100, tol=1e-6, alpha=1e-4, beta=0.5):
    """
    Gradient Descent with optional Backtracking Line Search.
    
    Parameters:
    - Q: Quadratic function matrix.
    - x0: Initial point.
    - fixed_step: If True, uses a fixed step size. If False, uses backtracking line search.
    - step_size: Step size for fixed step method.
    - max_iters: Maximum iterations.
    - tol: Convergence tolerance.
    - alpha: Sufficient reduction parameter for Armijo condition.
    - beta: Backtracking parameter.
    
    Returns:
    - iterates: List of iterates.
    - function_values: List of function values.
    - gradient_norms: List of gradient norms.
    - num_iterations: Number of iterations performed.
    """
    x = x0.copy()
    iterates = [x.copy()]
    function_values = [0.5 * x.T @ Q @ x]
    gradient_norms = [np.linalg.norm(Q @ x)]
    
    for i in range(max_iters):
        grad = Q @ x  # Gradient of f(x) = (1/2) x^T Q x is Qx
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            return np.array(iterates), np.array(function_values), np.array(gradient_norms), i + 1
        
        if fixed_step:
            t = step_size
        else:
            # Backtracking Line Search
            t = 1  # Initial step size
            while 0.5 * (x - t * grad).T @ Q @ (x - t * grad) > 0.5 * x.T @ Q @ x - alpha * t * grad_norm**2:
                t *= beta
        
        x = x - t * grad  # Gradient descent step
        iterates.append(x.copy())
        function_values.append(0.5 * x.T @ Q @ x)
        gradient_norms.append(np.linalg.norm(Q @ x))
    
    return np.array(iterates), np.array(function_values), np.array(gradient_norms), max_iters

def plot_results(iterates_fixed, function_values_fixed, iterates_var, function_values_var, title, fig_number, num_iters_fixed, num_iters_var):
    """Generate function value vs iterations (semi-log scale) for fixed and variable step sizes."""
    plt.figure(figsize=(8, 5))

    # Ensure function values are positive for log scale
    function_values_fixed = np.abs(function_values_fixed) + 1e-10  
    function_values_var = np.abs(function_values_var) + 1e-10  
    
    plt.semilogy(range(len(function_values_fixed)), function_values_fixed, 'b-o', label='Fixed Step Size')
    plt.semilogy(range(len(function_values_var)), function_values_var, 'r-s', label='Variable Step Size (Backtracking)')
    
    plt.yscale("log")  # Explicitly enforce log scaling
    plt.title(f"Figure {fig_number}: {title} (Iterations: Fixed={num_iters_fixed}, Variable={num_iters_var})")
    plt.xlabel("Iteration")
    plt.ylabel("f(x) (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add a grid for clarity
    plt.show()

# Test cases
Q1 = np.array([[1, 0], [0, 1]])
Q2 = np.array([[10, 0], [0, 1]])
fig_count = 1

for Q, step_size in [(Q1, 0.1), (Q1, 0.5), (Q2, 0.01), (Q2, 0.05)]:
    iterates_fixed, f_values_fixed, grad_norms_fixed, num_iters_fixed = gradient_descent(Q, np.array([1.5, 1.5]), fixed_step=True, step_size=step_size)
    iterates_var, f_values_var, grad_norms_var, num_iters_var = gradient_descent(Q, np.array([1.5, 1.5]), fixed_step=False, alpha=1e-4, beta=0.5)
    
    plot_results(iterates_fixed, f_values_fixed, iterates_var, f_values_var, f"Fixed vs Variable Step Size (α={step_size})", fig_count, num_iters_fixed, num_iters_var)
    fig_count += 1
