import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the transformation matrix
A = np.array([[1, 0.5], [0.5, 1]])  # Example matrix

# Define the original plane (grid)
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
original_plane = np.vstack([X.flatten(), Y.flatten()])

# Apply the matrix transformation to the plane
transformed_plane = A @ original_plane

# Reshape the transformed plane back to grid format
X_transformed = transformed_plane[0].reshape(X.shape)
Y_transformed = transformed_plane[1].reshape(Y.shape)

# Define some vectors to show the effect of the matrix
vectors = np.array([[1, 0], [0, 1], [2, 2], [-1, 1]]).T  # Example vectors (2x4 array)
transformed_vectors = A @ vectors

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid(True)

# Plot the original plane (background)
original_plane_plot = ax.scatter(original_plane[0], original_plane[1], c='blue', alpha=0.3, label='Original Plane')

# Plot the transformed plane (foreground)
transformed_plane_plot = ax.scatter([], [], c='red', alpha=0.5, label='Transformed Plane')

# Plot the original vectors
original_quivers = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5) for v in vectors.T]

# Plot the transformed vectors
transformed_quivers = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5) for v in transformed_vectors.T]

# Animation function
def animate(i):
    # Interpolate between original and transformed plane
    t = i / 100  # t goes from 0 to 1
    interpolated_X = (1 - t) * original_plane[0] + t * transformed_plane[0]
    interpolated_Y = (1 - t) * original_plane[1] + t * transformed_plane[1]
    transformed_plane_plot.set_offsets(np.c_[interpolated_X, interpolated_Y])

    # Interpolate vectors
    for idx, (v_orig, v_trans) in enumerate(zip(vectors.T, transformed_vectors.T)):
        interpolated_vector = (1 - t) * v_orig + t * v_trans
        transformed_quivers[idx].set_UVC(interpolated_vector[0], interpolated_vector[1])

    return transformed_plane_plot, *transformed_quivers

# Create the animation
ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

# Add legend and labels
ax.legend()
ax.set_title("Matrix Transformation Animation")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Show the animation
plt.show()