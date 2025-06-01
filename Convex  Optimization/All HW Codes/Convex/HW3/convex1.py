import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(grad_f, x0, alpha, tolerance, max_iter):
    """
    Performs gradient descent with a fixed step size.

    Args:
        grad_f: Function that computes the gradient of f.
        x0: Initial point (numpy array).
        alpha: Fixed step size (scalar).
        tolerance: Stopping tolerance (scalar).
        max_iter: Maximum number of iterations (integer).

    Returns:
        A numpy array containing the sequence of iterates [x0, x1, x2, ...].
        Returns None if variable step size is requested or max_iter exceeded.
    """

    if isinstance(alpha, list) or isinstance(alpha, tuple) or isinstance(alpha, np.ndarray):
        print("Variable step size is currently not supported.")
        return None

    x = x0.copy()  # Important: work with a copy!
    iterates = [x]
    k = 0

    while np.linalg.norm(grad_f(x)) > tolerance and k < max_iter:
        x = x - alpha * grad_f(x)
        iterates.append(x)
        k += 1
    
    if k == max_iter:
        print("Maximum number of iterations reached.")
        return None

    return np.array(iterates)


# Define gradient functions for each Q
def grad_f_q1(x):  # Q = [[1, 0], [0, 1]]
    return x

def grad_f_q2(x):  # Q = [[10, 0], [0, 1]]
    return np.array([10*x[0], x[1]])


# Test cases
tolerance = 1e-6
max_iter = 1000
x0 = np.array([1.0, 1.0])  # Example initial point

# Case 1: Q = [[1, 0], [0, 1]]
Q1 = np.array([[1, 0], [0, 1]])
alphas1 = [0.1, 0.5]

for alpha in alphas1:
    iterates = gradient_descent(grad_f_q1, x0, alpha, tolerance, max_iter)
    if iterates is not None:
        # Plotting
        x_range = np.linspace(-1.5, 1.5, 100)  # Adjust ranges as needed
        y_range = np.linspace(-1.5, 1.5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 0.5 * X**2 + 0.5 * Y**2  # Corresponding f(x) for Q1

        # (a) Contour plot
        plt.figure()
        plt.contour(X, Y, Z, levels=np.logspace(-1, 2, 10))  # Adjust levels
        plt.plot(iterates[:, 0], iterates[:, 1], 'r-o', markersize=4)  # Overlay iterates
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot with Iterates (Q1, alpha={alpha})')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        # (b) Function value vs. iterations
        f_values = [0.5 * iterate.T @ Q1 @ iterate for iterate in iterates]
        plt.figure()
        plt.plot(f_values)
        plt.xlabel('Iteration (k)')
        plt.ylabel('f(x^(k))')
        plt.title(f'Function Value vs. Iterations (Q1, alpha={alpha})')
        plt.grid(True)
        plt.show()

        # (c) Gradient norm vs. iterations
        grad_norms = [np.linalg.norm(Q1 @ iterate) for iterate in iterates]
        plt.figure()
        plt.plot(grad_norms)
        plt.xlabel('Iteration (k)')
        plt.ylabel('||∇f(x^(k))||')
        plt.title(f'Gradient Norm vs. Iterations (Q1, alpha={alpha})')
        plt.grid(True)
        plt.show()



# Case 2: Q = [[10, 0], [0, 1]]
Q2 = np.array([[10, 0], [0, 1]])
alphas2 = [0.01, 0.05]

for alpha in alphas2:
    iterates = gradient_descent(grad_f_q2, x0, alpha, tolerance, max_iter)
    if iterates is not None:
        # Plotting (similar structure as above, but adjust for Q2)
        x_range = np.linspace(-1.5, 1.5, 100)  # Adjust ranges as needed
        y_range = np.linspace(-1.5, 1.5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 0.5 * 10*X**2 + 0.5 * Y**2  # Corresponding f(x) for Q2

        # (a) Contour plot
        plt.figure()
        plt.contour(X, Y, Z, levels=np.logspace(-1, 2, 10))  # Adjust levels
        plt.plot(iterates[:, 0], iterates[:, 1], 'r-o', markersize=4)  # Overlay iterates
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot with Iterates (Q2, alpha={alpha})')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        # (b) Function value vs. iterations
        f_values = [0.5 * iterate.T @ Q2 @ iterate for iterate in iterates]
        plt.figure()
        plt.plot(f_values)
        plt.xlabel('Iteration (k)')
        plt.ylabel('f(x^(k))')
        plt.title(f'Function Value vs. Iterations (Q2, alpha={alpha})')
        plt.grid(True)
        plt.show()

        # (c) Gradient norm vs. iterations
        grad_norms = [np.linalg.norm(Q2 @ iterate) for iterate in iterates]
        plt.figure()
        plt.plot(grad_norms)
        plt.xlabel('Iteration (k)')
        plt.ylabel('||∇f(x^(k))||')
        plt.title(f'Gradient Norm vs. Iterations (Q2, alpha={alpha})')
        plt.grid(True)
        plt.show()