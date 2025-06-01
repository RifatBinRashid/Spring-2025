import numpy as np
import matplotlib.pyplot as plt

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

# Newton's method with fixed step size (t = 1) and safeguards
def newton_method_fixed_step(f, f_grad, f_hess, x0, tol=1e-6, max_iters=10):
    iterates = [x0]  # Store iterates
    grad_values = [f_grad(x0)]  # Store gradient values
    l = x0  # Store initial point for printing
    
    for _ in range(max_iters):
        grad = f_grad(x0)
        hess = f_hess(x0)
        
        # Safeguard: Avoid division by very small Hessian values
        if abs(hess) < 1e-6:
            print("Hessian is too small. Stopping early.")
            break
        
        dx_nt = -grad / hess  # Newton step
        x0 += dx_nt  # Fixed step size t = 1
        
        # Safeguard: Clip x to avoid invalid values
        if f == f1:
            x0 = np.clip(x0, -100, 100)  # Clip x to avoid overflow in f1(x)
        elif f == f2:
            x0 = max(x0, 1e-6)  # Ensure x > 0 for f2(x)
        
        # Store iterate and gradient value
        iterates.append(x0)
        grad_values.append(f_grad(x0))
        
        # Check stopping condition
        if abs(grad) <= tol:
            break
    
    print(f'Initial point x0 = {l}; Final converged point x* = {x0:.6f}, Number of iterations = {len(iterates) - 1}')
    return np.array(iterates), np.array(grad_values)

# Initial points
x0_1, x0_1_1, x0_2 = 1, 1.1, 3

# Run Newton's method for f1(x) with x0 = 1
print("Running Newton's method for f1(x) with x0 = 1")
iterates_f1_x0_1, grad_values_f1_x0_1 = newton_method_fixed_step(f1, f1_grad, f1_hess, x0_1)

# Run Newton's method for f1(x) with x0 = 1.1
print("\nRunning Newton's method for f1(x) with x0 = 1.1")
iterates_f1_x0_1_1, grad_values_f1_x0_1_1 = newton_method_fixed_step(f1, f1_grad, f1_hess, x0_1_1)

# Run Newton's method for f2(x) with x0 = 3
print("\nRunning Newton's method for f2(x) with x0 = 3")
iterates_f2_x0_2, grad_values_f2_x0_2 = newton_method_fixed_step(f2, f2_grad, f2_hess, x0_2)

# Generate x values for plotting
x_f1 = np.linspace(-2, 2, 400)
x_f2 = np.linspace(0.1, 5, 400)

# Plot for f1 with x0=1
plt.figure(figsize=(12, 6))

# Plot f1(x) and iterates
plt.subplot(1, 2, 1)
plt.plot(x_f1, [f1(x) for x in x_f1], 'b', label='f1(x)')
plt.scatter(iterates_f1_x0_1, [f1(x) for x in iterates_f1_x0_1], c='r', label='Iterates')
plt.quiver(iterates_f1_x0_1[:-1], [f1(x) for x in iterates_f1_x0_1[:-1]], 
           np.diff(iterates_f1_x0_1), np.array([f1(x) for x in iterates_f1_x0_1[1:]]) - np.array([f1(x) for x in iterates_f1_x0_1[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='r')
plt.title('Function f1(x) with x0=1')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.grid()

# Plot f1_grad(x) and iterates
plt.subplot(1, 2, 2)
plt.plot(x_f1, [f1_grad(x) for x in x_f1], 'r', label="f1_grad(x)")
plt.scatter(iterates_f1_x0_1, [f1_grad(x) for x in iterates_f1_x0_1], c='r', label='Iterates')
plt.quiver(iterates_f1_x0_1[:-1], [f1_grad(x) for x in iterates_f1_x0_1[:-1]], 
           np.diff(iterates_f1_x0_1), np.array([f1_grad(x) for x in iterates_f1_x0_1[1:]]) - np.array([f1_grad(x) for x in iterates_f1_x0_1[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='r')
plt.title("Gradient f1_grad(x) with x0=1")
plt.xlabel('x')
plt.ylabel("f1_grad(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot for f1 with x0=1.1
plt.figure(figsize=(12, 6))

# Plot f1(x) and iterates
plt.subplot(1, 2, 1)
plt.plot(x_f1, [f1(x) for x in x_f1], 'b', label='f1(x)')
plt.scatter(iterates_f1_x0_1_1, [f1(x) for x in iterates_f1_x0_1_1], c='g', label='Iterates')
plt.quiver(iterates_f1_x0_1_1[:-1], [f1(x) for x in iterates_f1_x0_1_1[:-1]], 
           np.diff(iterates_f1_x0_1_1), np.array([f1(x) for x in iterates_f1_x0_1_1[1:]]) - np.array([f1(x) for x in iterates_f1_x0_1_1[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='g')
plt.title('Function f1(x) with x0=1.1')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.grid()

# Plot f1_grad(x) and iterates
plt.subplot(1, 2, 2)
plt.plot(x_f1, [f1_grad(x) for x in x_f1], 'r', label="f1_grad(x)")
plt.scatter(iterates_f1_x0_1_1, [f1_grad(x) for x in iterates_f1_x0_1_1], c='g', label='Iterates')
plt.quiver(iterates_f1_x0_1_1[:-1], [f1_grad(x) for x in iterates_f1_x0_1_1[:-1]], 
           np.diff(iterates_f1_x0_1_1), np.array([f1_grad(x) for x in iterates_f1_x0_1_1[1:]]) - np.array([f1_grad(x) for x in iterates_f1_x0_1_1[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='g')
plt.title("Gradient f1_grad(x) with x0=1.1")
plt.xlabel('x')
plt.ylabel("f1_grad(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot for f2 with x0=3
plt.figure(figsize=(12, 6))

# Plot f2(x) and iterates
plt.subplot(1, 2, 1)
plt.plot(x_f2, [f2(x) for x in x_f2], 'b', label='f2(x)')
plt.scatter(iterates_f2_x0_2, [f2(x) for x in iterates_f2_x0_2], c='r', label='Iterates')
plt.quiver(iterates_f2_x0_2[:-1], [f2(x) for x in iterates_f2_x0_2[:-1]], 
           np.diff(iterates_f2_x0_2), np.array([f2(x) for x in iterates_f2_x0_2[1:]]) - np.array([f2(x) for x in iterates_f2_x0_2[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='r')
plt.title('Function f2(x) with x0=3')
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()
plt.grid()

# Plot f2_grad(x) and iterates
plt.subplot(1, 2, 2)
plt.plot(x_f2, [f2_grad(x) for x in x_f2], 'r', label="f2_grad(x)")
plt.scatter(iterates_f2_x0_2, [f2_grad(x) for x in iterates_f2_x0_2], c='r', label='Iterates')
plt.quiver(iterates_f2_x0_2[:-1], [f2_grad(x) for x in iterates_f2_x0_2[:-1]], 
           np.diff(iterates_f2_x0_2), np.array([f2_grad(x) for x in iterates_f2_x0_2[1:]]) - np.array([f2_grad(x) for x in iterates_f2_x0_2[:-1]]), 
           angles='xy', scale_units='xy', scale=1, color='r')
plt.title("Gradient f2_grad(x) with x0=3")
plt.xlabel('x')
plt.ylabel("f2_grad(x)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()