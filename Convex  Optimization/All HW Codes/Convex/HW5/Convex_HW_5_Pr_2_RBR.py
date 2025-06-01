import numpy as np
import matplotlib.pyplot as plt

# Step_1: Newton's method with fixed step size 
def newton_method_fixed_step(f, f_grad, f_hess, x0, tol, max_iters):
    iterates = [x0]  # Store iterates
    grad_values = [f_grad(x0)]  # Store gradient values
    l = x0  # Store initial point for printing
    
    for _ in range(max_iters):
        grad = f_grad(x0)
        hess = f_hess(x0)
        
        # Avoid division by very small Hessian values
        if abs(hess) < 1e-6:
            print("Hessian is too small. Stopping early.")
            break
        
        # Compute Newton decrement λ²
        lambda2 = grad**2 / hess
        
        # Stopping criterion: λ²/2 ≤ tolerance
        if lambda2 / 2 <= tol:
            print(f"Stopping criterion met: λ²/2 = {lambda2 / 2:.6f} ≤ {tol}")
            break
        
        dx_nt = -grad / hess  # Newton direction
        x0 += dx_nt  # Fixed step size t = 1
        
        # Store iterate and gradient value
        iterates.append(x0)
        grad_values.append(f_grad(x0))
    
    print(f'Initial point x0 = {l}; Final converged point x* = {x0:.6f}, Number of iterations = {len(iterates) - 1}')
    return np.array(iterates), np.array(grad_values)

# Step_2: Define the functions and their derivatives
def f1(x):
    return np.log(np.exp(x) + np.exp(-x))

def f1_grad(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def f1_hess(x):
    return (4 * np.exp(x) * np.exp(-x)) / (np.exp(x) + np.exp(-x))**2

def f2(x):
    return -np.log(x) + x

def f2_grad(x):
    return -1/x + 1

def f2_hess(x):
    return 1/x**2

# Step_3: Initial points
x0_1, x0_1_1, x0_2 = 1, 1.1, 3

# Step_4: Hyperparameters
tol = 1e-6
max_iters = 100

# Step_5: Run and plot
# Run Newton's method for f1(x) with x0 = 1
print("Running Newton's method for f1(x) with x0 = 1")
iterates_f1_x0_1, grad_values_f1_x0_1 = newton_method_fixed_step(f1, f1_grad, f1_hess, x0_1, tol, max_iters)

# Run Newton's method for f1(x) with x0 = 1.1
print("\nRunning Newton's method for f1(x) with x0 = 1.1")
iterates_f1_x0_1_1, grad_values_f1_x0_1_1 = newton_method_fixed_step(f1, f1_grad, f1_hess, x0_1_1, tol, max_iters)

# Run Newton's method for f2(x) with x0 = 3
print("\nRunning Newton's method for f2(x) with x0 = 3")
iterates_f2_x0_2, grad_values_f2_x0_2 = newton_method_fixed_step(f2, f2_grad, f2_hess, x0_2, tol, max_iters)

# Generate x values for plotting
x_f1 = np.linspace(-2, 2, 400)
x_f2 = np.linspace(0.1, 5, 400)

# Function to plot all three required graphs
def plot_case(title, x_vals, f, f_grad, iterates, grad_values, color):
    plt.figure(figsize=(18, 5))

    # (1) Plot f(x) with iterates
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, f(x_vals), 'b', label='f(x)')
    plt.scatter(iterates, f(iterates), c=color, label='Iterates')
    plt.quiver(iterates[:-1], f(iterates[:-1]), 
               np.diff(iterates), f(iterates[1:]) - f(iterates[:-1]), 
               angles='xy', scale_units='xy', scale=1, color=color)
    plt.title(f'Function {title}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()

    # (2) Plot gradient vs iterations
    plt.subplot(1, 3, 2)
    plt.plot(range(len(grad_values)), grad_values, f'{color}-o', label="Gradient values")
    plt.title(f'Gradient {title} vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid()

    # (3) Plot gradient vs x
    plt.subplot(1, 3, 3)
    plt.plot(x_vals, f_grad(x_vals), 'r', label="Gradient vs x")
    plt.scatter(iterates, f_grad(iterates), c=color, label="Iterate Gradients")
    plt.title(f'Gradient {title} vs x')
    plt.xlabel('x')
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Plot for f1 with x0=1
plot_case("f1(x) with x0=1", x_f1, f1, f1_grad, iterates_f1_x0_1, grad_values_f1_x0_1, 'r')

# Plot for f1 with x0=1.1
plot_case("f1(x) with x0=1.1", x_f1, f1, f1_grad, iterates_f1_x0_1_1, grad_values_f1_x0_1_1, 'g')

# Plot for f2 with x0=3
plot_case("f2(x) with x0=3", x_f2, f2, f2_grad, iterates_f2_x0_2, grad_values_f2_x0_2, 'b')
