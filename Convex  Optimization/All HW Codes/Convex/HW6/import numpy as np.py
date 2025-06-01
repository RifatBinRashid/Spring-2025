import numpy as np
import matplotlib.pyplot as plt

# Step 1: Defining the objective function f(x)
def f(x):
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

# Step 2: Defining the gradient ∇f(x)
def grad_f(x):
    df_dx1 = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1)
    df_dx2 = 3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    return np.array([df_dx1, df_dx2])

# Step 3: Defining the Hessian ∇²f(x)
def hess_f(x):
    e1 = np.exp(x[0] + 3*x[1] - 0.1)
    e2 = np.exp(x[0] - 3*x[1] - 0.1)
    e3 = np.exp(-x[0] - 0.1)
    h11 = e1 + e2 + e3
    h12 = 3*e1 - 3*e2
    h22 = 9*e1 + 9*e2
    return np.array([[h11, h12], [h12, h22]])

# Step 4: Backtracking line search (Armijo rule)
def backtracking(x, d, alpha=0.1, beta=0.7):
    t = 1
    fx = f(x)
    grad_fx = grad_f(x)
    while f(x + t*d) > fx + alpha * t * grad_fx.dot(d):
        t *= beta
    return t

# Step 5: ℓ1-norm steepest descent
def sd_l1(x0, max_iter=100):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        i = np.argmax(np.abs(g))
        d = -np.sign(g[i]) * np.eye(2)[i]
        t = backtracking(x, d)
        x = x + t * d
        history.append(x.copy())
        if np.linalg.norm(g) < 1e-6:
            break
    return np.array(history)

# Step 6: ℓ∞-norm steepest descent
def sd_linf(x0, max_iter=100):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        d = -np.linalg.norm(g, 1) * np.sign(g)
        t = backtracking(x, d)
        x = x + t * d
        history.append(x.copy())
        if np.linalg.norm(g) < 1e-6:
            break
    return np.array(history)

# Step 7: Quadratic norm (P-norm) steepest descent
def sd_quad(x0, P, max_iter=100):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        d = -np.linalg.solve(P, g)
        t = backtracking(x, d)
        x = x + t * d
        history.append(x.copy())
        if np.linalg.norm(g) < 1e-6:
            break
    return np.array(history)

# Step 8: Euclidean gradient descent
def gd_euclidean(x0, max_iter=100):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        d = -g
        t = backtracking(x, d)
        x = x + t * d
        history.append(x.copy())
        if np.linalg.norm(g) < 1e-6:
            break
    return np.array(history)

# Step 9: Newton’s method using Hessian
def newton_method(x0, max_iter=100):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        d = -np.linalg.solve(H, g)
        t = backtracking(x, d)
        x = x + t * d
        history.append(x.copy())
        if np.linalg.norm(g) < 1e-6:
            break
    return np.array(history)

# Step 10: Execute all methods and store trajectories
x0 = np.array([0.1, 0.1])
paths = {
    'L1': sd_l1(x0),
    'Linf': sd_linf(x0),
    'P1': sd_quad(x0, np.array([[2, 0], [0, 8]])),
    'P2': sd_quad(x0, np.array([[8, 0], [0, 2]])),
    'Euclidean': gd_euclidean(x0),
    'Newton': newton_method(x0)
}

# Step 11: Determine optimal function value p_star
p_star = min(f(path[-1]) for path in paths.values())

# Step 12: Define grid for contour plotting
x1 = np.linspace(-0.6, 0.2, 400)
x2 = np.linspace(-0.2, 0.2, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(np.array([X1, X2]))
colors = ['r', 'b', 'g', 'm', 'orange', 'cyan']

# Step 13: Plot contour with iterates for each method
for i, (label, path) in enumerate(paths.items()):
    plt.figure(figsize=(6, 5))
    CS = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.plot(path[:, 0], path[:, 1], marker='o', markersize=4, linestyle='-', linewidth=1.5, color=colors[i], label=label)
    plt.plot(path[0, 0], path[0, 1], marker='x', color='black', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], marker='o', color='black', markersize=6, label='End')
    plt.title(f'Iterates on Contour Plot - {label}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 14: Plot function value gap for each method (semi-log)
for i, (label, path) in enumerate(paths.items()):
    gaps = [f(xk) - p_star for xk in path]
    plt.figure(figsize=(6, 4))
    plt.semilogy(range(len(gaps)), gaps, marker='o', markersize=4, color=colors[i])
    plt.title(f'Function Value Gap $f(x^{{(k)}}) - p^*$ - {label}')
    plt.xlabel('Iteration')
    plt.ylabel('$f(x^{(k)}) - p^*$ (log scale)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

# Step 15: Plot function value gap for all methods combined (semi-log)
plt.figure(figsize=(10, 6))
for i, (label, path) in enumerate(paths.items()):
    gaps = [f(xk) - p_star for xk in path]
    plt.semilogy(range(len(gaps)), gaps, marker='o', markersize=3, linewidth=1.5, label=label, color=colors[i])
plt.title('Function Value Gap $f(x^{(k)}) - p^*$ vs Iteration (All Methods)')
plt.xlabel('Iteration Number')
plt.ylabel('$f(x^{(k)}) - p^*$ (log scale)')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# Step 16: Print summary of iterations and final values
print("Method Summary:")
print(f"{'Method':<10} | {'Iterations':<10} | {'Final f(x)':<12}")
print("-" * 36)
for label, path in paths.items():
    iterations = len(path) - 1
    final_fx = f(path[-1])
    print(f"{label:<10} | {iterations:<10} | {final_fx:<12.6f}")
