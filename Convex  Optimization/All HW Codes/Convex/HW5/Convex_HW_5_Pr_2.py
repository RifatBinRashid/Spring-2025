import numpy as np
import matplotlib.pyplot as plt

# Newton's method with full step (t = 1) and safeguards
def newton_method(f, f_grad, f_hess, x0, tol=1e-6, max_iters=10):
    iterates = [x0]
    k = 0  # k: Number of iterations.
    
    while k < max_iters:
        grad = f_grad(x0)
        hess = f_hess(x0)
        
        # Safeguard: Avoid division by very small Hessian values
        if abs(hess) < 1e-6:
            print("Hessian is too small. Stopping early.")
            break
        
        dx_nt = -grad / hess  # Newton direction
        
        # Update x with full Newton step (t = 1)
        x0 = x0 + dx_nt
        
        # Safeguard: Clip x to avoid invalid values
        if f == f2:
            x0 = max(x0, 1e-6)  # Ensure x > 0 for f2(x)
        
        iterates.append(x0)
        k += 1
        
        # Check stopping condition
        if abs(grad) <= tol:
            break
    
    print(f'Final optimizer x* = {x0:.6f}, Iterations needed: {k}')
    return np.array(iterates), k

# Define the functions and their derivatives
def f1(x):
    # Log-sum-exp trick to avoid overflow/underflow
    if x >= 0:
        return x + np.log(1 + np.exp(-2 * x))
    else:
        return -x + np.log(1 + np.exp(2 * x))

def f1_grad(x):
    # Gradient of f1(x)
    return np.tanh(x)

def f1_hess(x):
    # Hessian of f1(x) with safeguard to avoid overflow
    x_clipped = np.clip(x, -100, 100)  # Clip x to avoid overflow in cosh(x)
    return 1 / np.cosh(x_clipped)**2

def f2(x):
    # Safeguard: Ensure x > 0
    x = max(x, 1e-6)
    return -np.log(x) + x

def f2_grad(x):
    # Gradient of f2(x)
    x = max(x, 1e-6)
    return -1 / x + 1

def f2_hess(x):
    # Hessian of f2(x)
    x = max(x, 1e-6)
    return 1 / x**2

# Set parameters
tol = 1e-6
max_iters = 10
x0_1 = 1.0      # Initial point for f1
x0_1_1 = 1.1    # Slightly different start for f1
x0_2 = 3.0      # Initial point for f2

# Run Newton's method for f1(x) with x0 = 1.0
print("Running Newton's method for f1(x) with x0 = 1.0")
iterates_f1_x0_1, iterations_f1_x0_1 = newton_method(f1, f1_grad, f1_hess, x0_1, tol, max_iters)

# Run Newton's method for f1(x) with x0 = 1.1
print("\nRunning Newton's method for f1(x) with x0 = 1.1")
iterates_f1_x0_1_1, iterations_f1_x0_1_1 = newton_method(f1, f1_grad, f1_hess, x0_1_1, tol, max_iters)

# Run Newton's method for f2(x) with x0 = 3.0
print("\nRunning Newton's method for f2(x) with x0 = 3.0")
iterates_f2_x0_2, iterations_f2_x0_2 = newton_method(f2, f2_grad, f2_hess, x0_2, tol, max_iters)

# Generate x values for plotting
x_f1 = np.linspace(-2, 2, 400)
x_f2 = np.linspace(0.1, 5, 400)  # Avoiding log singularity at x=0

# Plot 1: f1(x) with x0 = 1.0
plt.figure(figsize=(12, 6))
plt.plot(x_f1, [f1(x) for x in x_f1], 'b', linewidth=1.5, label='f1(x)')
for i in range(1, len(iterates_f1_x0_1)):
    plt.arrow(iterates_f1_x0_1[i-1], f1(iterates_f1_x0_1[i-1]), 
              iterates_f1_x0_1[i] - iterates_f1_x0_1[i-1], f1(iterates_f1_x0_1[i]) - f1(iterates_f1_x0_1[i-1]), 
              head_width=0.05, head_length=0.1, fc='r', ec='r')
plt.scatter(iterates_f1_x0_1, [f1(x) for x in iterates_f1_x0_1], c='r', marker='o', label=f'Iterates (x0=1.0), Iterations: {iterations_f1_x0_1}')
plt.title('Function f1(x) with Newton Iterations (x0=1.0)')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.grid()
plt.legend()
plt.show()

# Plot 2: f1_grad(x) with x0 = 1.0
plt.figure(figsize=(12, 6))
plt.plot(x_f1, [f1_grad(x) for x in x_f1], 'r', linewidth=1.5, label="f1_grad(x)")
plt.scatter(iterates_f1_x0_1, [f1_grad(x) for x in iterates_f1_x0_1], c='r', marker='o', label=f'Iterates (x0=1.0), Iterations: {iterations_f1_x0_1}')
plt.title("Gradient f1_grad(x) with Newton Iterations (x0=1.0)")
plt.xlabel('x')
plt.ylabel("f1_grad(x)")
plt.grid()
plt.legend()
plt.show()

# Plot 3: f1(x) with x0 = 1.1
plt.figure(figsize=(12, 6))
plt.plot(x_f1, [f1(x) for x in x_f1], 'b', linewidth=1.5, label='f1(x)')
for i in range(1, len(iterates_f1_x0_1_1)):
    plt.arrow(iterates_f1_x0_1_1[i-1], f1(iterates_f1_x0_1_1[i-1]), 
              iterates_f1_x0_1_1[i] - iterates_f1_x0_1_1[i-1], f1(iterates_f1_x0_1_1[i]) - f1(iterates_f1_x0_1_1[i-1]), 
              head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.scatter(iterates_f1_x0_1_1, [f1(x) for x in iterates_f1_x0_1_1], c='g', marker='o', label=f'Iterates (x0=1.1), Iterations: {iterations_f1_x0_1_1}')
plt.title('Function f1(x) with Newton Iterations (x0=1.1)')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.grid()
plt.legend()
plt.show()

# Plot 4: f1_grad(x) with x0 = 1.1
plt.figure(figsize=(12, 6))
plt.plot(x_f1, [f1_grad(x) for x in x_f1], 'r', linewidth=1.5, label="f1_grad(x)")
plt.scatter(iterates_f1_x0_1_1, [f1_grad(x) for x in iterates_f1_x0_1_1], c='g', marker='o', label=f'Iterates (x0=1.1), Iterations: {iterations_f1_x0_1_1}')
plt.title("Gradient f1_grad(x) with Newton Iterations (x0=1.1)")
plt.xlabel('x')
plt.ylabel("f1_grad(x)")
plt.grid()
plt.legend()
plt.show()

# Plot 5: f2(x) with x0 = 3.0
plt.figure(figsize=(12, 6))
plt.plot(x_f2, [f2(x) for x in x_f2], 'b', linewidth=1.5, label='f2(x)')
for i in range(1, len(iterates_f2_x0_2)):
    plt.arrow(iterates_f2_x0_2[i-1], f2(iterates_f2_x0_2[i-1]), 
              iterates_f2_x0_2[i] - iterates_f2_x0_2[i-1], f2(iterates_f2_x0_2[i]) - f2(iterates_f2_x0_2[i-1]), 
              head_width=0.05, head_length=0.1, fc='r', ec='r')
plt.scatter(iterates_f2_x0_2, [f2(x) for x in iterates_f2_x0_2], c='r', marker='o', label=f'Iterates (x0=3.0), Iterations: {iterations_f2_x0_2}')
plt.title('Function f2(x) with Newton Iterations (x0=3.0)')
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.grid()
plt.legend()
plt.show()

# Plot 6: f2_grad(x) with x0 = 3.0
plt.figure(figsize=(12, 6))
plt.plot(x_f2, [f2_grad(x) for x in x_f2], 'r', linewidth=1.5, label="f2_grad(x)")
plt.scatter(iterates_f2_x0_2, [f2_grad(x) for x in iterates_f2_x0_2], c='r', marker='o', label=f'Iterates (x0=3.0), Iterations: {iterations_f2_x0_2}')
plt.title("Gradient f2_grad(x) with Newton Iterations (x0=3.0)")
plt.xlabel('x')
plt.ylabel("f2_grad(x)")
plt.grid()
plt.legend()
plt.show()