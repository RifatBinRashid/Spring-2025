import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its derivatives
def f(x):
    x1, x2 = x[0], x[1]
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)

def f_grad(x):
    x1, x2 = x[0], x[1]
    df_dx1 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1)
    df_dx2 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)
    return np.array([df_dx1, df_dx2])

def f_hess(x):
    x1, x2 = x[0], x[1]
    d2f_dx1dx1 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    d2f_dx1dx2 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)
    d2f_dx2dx2 = 9 * np.exp(x1 + 3*x2 - 0.1) + 9 * np.exp(x1 - 3*x2 - 0.1)
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx1dx2, d2f_dx2dx2]])

# Backtracking line search
def backtracking_line_search(x, delta_x, alpha=0.1, beta=0.7):
    t = 1.0
    while f(x + t * delta_x) > f(x) + alpha * t * np.dot(f_grad(x), delta_x):
        t *= beta
    return t

# Steepest descent in the ℓ1 norm
def steepest_descent_l1(x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    X = [x]
    for k in range(max_iter):
        grad = f_grad(x)
        i = np.argmax(np.abs(grad))
        delta_x = -grad[i] * np.eye(2)[i]
        t = backtracking_line_search(x, delta_x)
        x_new = x + t * delta_x
        X.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(X).T, k+1

# Steepest descent in the ℓ∞ norm
def steepest_descent_linf(x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    X = [x]
    for k in range(max_iter):
        grad = f_grad(x)
        delta_x = -np.linalg.norm(grad, 1) * np.sign(grad)
        t = backtracking_line_search(x, delta_x)
        x_new = x + t * delta_x
        X.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(X).T, k+1

# Steepest descent in a quadratic P-norm
def steepest_descent_pnorm(x0, P, max_iter=100, tol=1e-6):
    x = x0.copy()
    X = [x]
    for k in range(max_iter):
        grad = f_grad(x)
        delta_x = -np.linalg.solve(P, grad)
        t = backtracking_line_search(x, delta_x)
        x_new = x + t * delta_x
        X.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(X).T, k+1

# Gradient descent (Euclidean norm)
def gradient_descent(x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    X = [x]
    for k in range(max_iter):
        grad = f_grad(x)
        delta_x = -grad
        t = backtracking_line_search(x, delta_x)
        x_new = x + t * delta_x
        X.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(X).T, k+1

# Newton's method
def newton_method(x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    X = [x]
    for k in range(max_iter):
        grad = f_grad(x)
        hess = f_hess(x)
        delta_x = -np.linalg.solve(hess, grad)
        t = backtracking_line_search(x, delta_x)
        x_new = x + t * delta_x
        X.append(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(X).T, k+1

# Define P matrices for quadratic norm
P1 = np.array([[2, 0], [0, 8]])
P2 = np.array([[8, 0], [0, 2]])

# Initial point
x0 = np.array([0.1, 0.1])

# Run all methods and store results
methods = [
    ('ℓ1 norm', *steepest_descent_l1(x0), 'r-o'),
    ('ℓ∞ norm', *steepest_descent_linf(x0), 'g-s'),
    ('P1 norm', *steepest_descent_pnorm(x0, P1), 'b-^'),
    ('P2 norm', *steepest_descent_pnorm(x0, P2), 'm-D'),
    ('Gradient descent', *gradient_descent(x0), 'c-*'),
    ('Newton method', *newton_method(x0), 'k-+')
]

# Print iteration counts
print("Iteration counts for each method:")
for name, _, iterations, _ in methods:
    print(f"{name}: {iterations} iterations")

# Compute optimal value (approximated by the best solution found)
p_star = min(f(X[:, -1]) for (_, X, _, _) in methods)

# Create contour plot of f(x1, x2)
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([f(np.array([x1_, x2_])) for x1_, x2_ in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

# Create individual contour plots for each method
plt.figure(figsize=(15, 10))
for i, (name, X, iterations, style) in enumerate(methods):
    plt.subplot(2, 3, i+1)
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.plot(X[0], X[1], style, markersize=4, linewidth=1.5)
    plt.title(f"{name}\nIterations: {iterations}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()

plt.tight_layout()
plt.show()

# Plot function value gap vs iterations (semi-log)
plt.figure(figsize=(10, 6))
for name, X, iterations, style in methods:
    plt.semilogy(range(iterations+1), [f(x) - p_star for x in X.T], 
                style, markersize=4, linewidth=1.5, label=f"{name} ({iterations} iters)")
plt.title('Function value gap vs iterations (semi-log)')
plt.xlabel('Iteration')
plt.ylabel('f(x) - p*')
plt.legend()
plt.grid()
plt.show()