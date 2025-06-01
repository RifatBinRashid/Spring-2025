import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(Q, x0, alpha, max_iters=100, tol=1e-6):
    x = x0
    iterates = [x.copy()]
    grad_norms = []
    func_values = []
    
    for _ in range(max_iters):
        grad = Q @ x  # Gradient of f(x) = (1/2) x^T Q x is Qx
        grad_norm = np.linalg.norm(grad)
        
        func_values.append(0.5 * x.T @ Q @ x)
        grad_norms.append(grad_norm)
        
        if grad_norm < tol:
            break
        
        x = x - alpha * grad  # Gradient descent step
        iterates.append(x.copy())
    
    return np.array(iterates), np.array(func_values), np.array(grad_norms)

def plot_results(iterates, func_values, grad_norms, Q, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Contour plot
    x1_vals, x2_vals = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 * Q[0, 0] + X2**2 * Q[1, 1])
    
    axes[0].contour(X1, X2, Z, levels=20)
    iter_x1, iter_x2 = iterates[:, 0], iterates[:, 1]
    axes[0].plot(iter_x1, iter_x2, 'ro-', label='Iterates')
    axes[0].set_title("Contour Plot with Iterates")
    axes[0].legend()
    
    # Function values over iterations
    axes[1].plot(func_values, 'b-o')
    axes[1].set_title("Function Value vs Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("f(x)")
    
    # Gradient norm over iterations
    axes[2].plot(grad_norms, 'r-o')
    axes[2].set_title("Gradient Norm vs Iterations")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("||∇f(x)||")
    
    plt.suptitle(title)
    plt.show()

# Test cases
Q1 = np.array([[1, 0], [0, 1]])
Q2 = np.array([[10, 0], [0, 1]])

for Q, alpha in [(Q1, 0.1), (Q1, 0.5), (Q2, 0.01), (Q2, 0.05)]:
    iterates, func_values, grad_norms = gradient_descent(Q, np.array([1.5, 1.5]), alpha)
    plot_results(iterates, func_values, grad_norms, Q, f"Gradient Descent with alpha={alpha}")
