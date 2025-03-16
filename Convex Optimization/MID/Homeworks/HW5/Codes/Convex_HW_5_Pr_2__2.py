import numpy as np
import matplotlib.pyplot as plt

# Safe exponential function to prevent overflow
def safe_exp(x):
    return np.exp(np.clip(x, -50, 50))  # Clip values to avoid large exponentials

# Safe log function to prevent log(0) or log(negative)
def safe_log(x):
    return np.log(np.maximum(x, 1e-10))  # Avoids NaN issues

# Define function f1 and its derivatives
def f1(x):
    return safe_log(safe_exp(x) + safe_exp(-x))

def f1_prime(x):
    return (safe_exp(x) - safe_exp(-x)) / (safe_exp(x) + safe_exp(-x))

def f1_double_prime(x):
    return (4 * safe_exp(x) * safe_exp(-x)) / (safe_exp(x) + safe_exp(-x))**2

# Define function f2 and its derivatives
def f2(x):
    return -safe_log(x) + x  # Ensure x > 0

def f2_prime(x):
    return -1 / np.maximum(x, 1e-10) + 1  # Avoid division by zero

def f2_double_prime(x):
    return 1 / np.maximum(x, 1e-10) ** 2  # Avoid division by zero

# Newton's Method with Newton Decrement stopping criterion
def newton_method_with_decrement(f, f_prime, f_double_prime, x0, max_iters=10, tol=1e-6):
    iterates = [x0]

    for _ in range(max_iters):
        grad = f_prime(x0)
        hess = f_double_prime(x0)

        # Ensure Hessian is not too small
        if abs(hess) < tol:
            print(f"Stopped: Hessian too small at x = {x0:.6f}")
            break

        try:
            delta_x = -grad / hess  # Newton step
        except np.linalg.LinAlgError:
            print(f"Stopped: Singular Hessian at x = {x0:.6f}")
            break

        # Compute Newton decrement λ² = ∇f(x)^T H^-1 ∇f(x)
        newton_decrement = grad * delta_x  # Since it's scalar 1D, this is valid
        
        # Stop if Newton decrement is below tolerance
        if newton_decrement / 2 <= tol:
            print(f"Converged: Newton decrement λ² = {newton_decrement:.6e} at x = {x0:.6f}")
            break

        x0 += delta_x  # Update x
        iterates.append(x0)

    print(f'Final converged point x* = {x0:.6f}, Number of iterations = {len(iterates) - 1}')
    return np.array(iterates)

# Initial conditions
x0_1, x0_1_1, x0_2 = 1, 1.1, 3
iterates_f1_x0_1 = newton_method_with_decrement(f1, f1_prime, f1_double_prime, x0_1)
iterates_f1_x0_1_1 = newton_method_with_decrement(f1, f1_prime, f1_double_prime, x0_1_1)
iterates_f2_x0_2 = newton_method_with_decrement(f2, f2_prime, f2_double_prime, x0_2)

# Generate x values for plotting
x_f1 = np.linspace(-2, 2, 400)
x_f2 = np.linspace(0.1, 5, 400)

# Function and derivative plots with iterates
cases = [
    ("f1(x) with x0=1", x_f1, f1, iterates_f1_x0_1, 'r'),
    ("f1(x) with x0=1.1", x_f1, f1, iterates_f1_x0_1_1, 'g'),
    ("f2(x) with x0=3", x_f2, f2, iterates_f2_x0_2, 'b'),
]

# Plot function f(x)
for title, x_vals, f, iterates, color in cases:
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, f(x_vals), 'b', label=f'{title}')
    plt.scatter(iterates, f(iterates), c=color, marker='o', label='Iterates')
    
    # Add arrows to indicate iteration steps
    plt.quiver(
        iterates[:-1], f(iterates[:-1]),
        np.diff(iterates), f(iterates[1:]) - f(iterates[:-1]),
        angles='xy', scale_units='xy', scale=1, color=color
    )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function and derivative plots
cases_prime = [
    ("f1'(x) with x0=1", x_f1, f1_prime),
    ("f1'(x) with x0=1.1", x_f1, f1_prime),
    ("f2'(x) with x0=3", x_f2, f2_prime),
]

# Plot f'(x)
for title, x_vals, f_prime in cases_prime:
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, f_prime(x_vals), 'r', label=f"{title}")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
