# Re-import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the transformation matrix
transformation_matrix = np.array([
    [1, 1],
    [0, -1]
])

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(transformation_matrix)

# Define the initial vectors
vectors = [
    np.array([3, 4]),
    np.array([1, -5]),
]

# Define the grid
grid_size = 8
x = np.arange(-grid_size, grid_size + 1)
y = np.arange(-grid_size, grid_size + 1)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack([X.ravel(), Y.ravel()])

# Create horizontal and vertical grid lines
horizontal_lines = [[(x[0], y_val), (x[-1], y_val)] for y_val in y]
vertical_lines = [[(x_val, y[0]), (x_val, y[-1])] for x_val in x]
grid_lines_all = horizontal_lines + vertical_lines

# Speed parameter
speed = 50

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-grid_size - 1, grid_size + 1)
ax.set_ylim(-grid_size - 1, grid_size + 1)
ax.set_title('Transformation with Eigenvectors', pad=20)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Draw faded static grid
for (start, end) in grid_lines_all:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', linestyle='--', alpha=0.3)

# Plot static eigenvectors if they are real
eigenvector_lines = []
eigenvector_traces = [[] for _ in eigvecs.T]  # Store eigenvector transformation points
eigenvector_trace_lines = [ax.plot([], [], 'o', color='orange', markersize=4, alpha=0.6)[0] for _ in eigvecs.T]

if np.isreal(eigvecs).all():
    for i in range(len(eigvals)):
        ev = np.real(eigvecs[:, i]) * grid_size
        line, = ax.plot([0, ev[0]], [0, ev[1]], linestyle='-', color='orange', linewidth=2, label=f'Eigenvector {i+1}')
        eigenvector_lines.append(line)

# Initialize grid elements
grid_dots, = ax.plot([], [], 'ko', markersize=3, label='Transformed Grid Points')
animated_grid_lines = [ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)[0] for _ in grid_lines_all]

# Initialize vector arrows and lines
colors = ['r', 'g', 'b', 'm', 'c', 'y']
quivers = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                     color=colors[i % len(colors)], label=f'Vector {i+1}') for i, v in enumerate(vectors)]
initial_lines = [ax.plot([0, v[0]], [0, v[1]], color=colors[i % len(colors)], linestyle='-', linewidth=2, alpha=0.5)[0]
                 for i, v in enumerate(vectors)]
previous_positions = [[] for _ in vectors]
previous_lines = [ax.plot([], [], '--', color=colors[i % len(colors)], alpha=0.5, linewidth=2)[0]
                  for i in range(len(vectors))]

# Text annotations inside the graph but aligned properly
initial_coords = [ax.text(v[0] * 1.1, v[1] * 1.1, f'({v[0]}, {v[1]})', fontsize=12, color=colors[i % len(colors)],
                          ha='right', va='bottom', fontweight='bold')
                  for i, v in enumerate(vectors)]
final_coords = [ax.text(0, 0, '', fontsize=12, color=colors[i % len(colors)],
                        ha='left', va='top', fontweight='bold') for i in range(len(vectors))]

# Eigenvector transformation arrows
eigen_quivers = []
if np.isreal(eigvecs).all():
    for i in range(len(eigvals)):
        eq = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.5, linestyle='dashed')
        eigen_quivers.append(eq)

# Pause frames at the beginning
def initial_pause():
    for _ in range(10):
        yield None

# Update function for animation
def update(frame):
    if frame is None:
        return *quivers, grid_dots, *animated_grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines, *eigen_quivers, *eigenvector_trace_lines

    alpha = frame / 90
    interpolated_matrix = np.eye(2) * (1 - alpha) + transformation_matrix * alpha

    # Transform grid points and update dots
    transformed_grid = interpolated_matrix @ grid_points
    grid_dots.set_data(transformed_grid[0], transformed_grid[1])

    for i, ((start, end), line) in enumerate(zip(grid_lines_all, animated_grid_lines)):
        p1 = interpolated_matrix @ np.array(start)
        p2 = interpolated_matrix @ np.array(end)
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # Transform vectors
    transformed_vectors = [interpolated_matrix @ v for v in vectors]
    for i, (quiver, vec) in enumerate(zip(quivers, transformed_vectors)):
        quiver.set_UVC(vec[0], vec[1])
        previous_positions[i].append((vec[0], vec[1]))
        x_vals, y_vals = zip(*previous_positions[i])
        previous_lines[i].set_data(x_vals, y_vals)

    if frame == 90:
        for coord, vec in zip(final_coords, transformed_vectors):
            coord.set_position((vec[0] * 1.1, vec[1] * 1.1))
            coord.set_text(f'({vec[0]:.2f}, {vec[1]:.2f})')

    # Animate eigenvector transformations and track their motion
    if np.isreal(eigvecs).all():
        for i, eq in enumerate(eigen_quivers):
            ev = np.real(eigvecs[:, i]) * grid_size
            tev = interpolated_matrix @ np.real(eigvecs[:, i]) * grid_size
            eq.set_UVC(tev[0], tev[1])
            eigenvector_traces[i].append((tev[0], tev[1]))

            # Update the eigenvector trace dots
            trace_x, trace_y = zip(*eigenvector_traces[i])
            eigenvector_trace_lines[i].set_data(trace_x, trace_y)

    return *quivers, grid_dots, *animated_grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines, *eigen_quivers, *eigenvector_trace_lines

# Run animation
def run_animation():
    for pos in previous_positions:
        pos.clear()
    for txt in final_coords:
        txt.set_text('')
    ani = FuncAnimation(fig, update, frames=[*initial_pause(), *np.arange(0, 91, 1)],
                        interval=speed, blit=True, repeat=False)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))  # Move legend outside graph
    plt.show()

run_animation()
