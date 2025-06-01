import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

# Create a meshgrid over the domain
x1 = np.linspace(-0.6, 0.2, 400)
x2 = np.linspace(-0.2, 0.2, 400)
X1, X2 = np.meshgrid(x1, x2)

# Compute Z = f(x) over the grid
Z = f(np.array([X1, X2]))

# Plot contour lines
plt.figure(figsize=(6, 5))
CS = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
plt.clabel(CS, inline=True, fontsize=8)
plt.title('Level Sets of $f(x)$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.tight_layout()
plt.show()
