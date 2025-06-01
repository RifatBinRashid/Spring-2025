import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a general transformation matrix (modify this for different transformations)
transformation_matrix = np.array([
    [1, 1],  # Example: Scaling and Shearing
    [0, -1]
])

# Define the initial vectors
vectors = [
    np.array([2, 3]),
    np.array([1, -5]),
    np.array([-1, 2]),
]

# Define the grid
grid_size = 6
x = np.arange(-grid_size, grid_size + 1)
y = np.arange(-grid_size, grid_size + 1)
X, Y = np.meshgrid(x, y)  # Create grid points
grid_points = np.vstack([X.ravel(), Y.ravel()])

# Prepare horizontal and vertical grid lines
horizontal_lines = [[(x[0], y_val), (x[-1], y_val)] for y_val in y]
vertical_lines = [[(x_val, y[0]), (x_val, y[-1])] for x_val in x]
grid_lines_all = horizontal_lines + vertical_lines

# Speed parameter
speed = 50

# Define colors
colors = ['r', 'g', 'b', 'm', 'c', 'y']

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-grid_size - 1, grid_size + 1)
ax.set_ylim(-grid_size - 1, grid_size + 1)
ax.set_title('General Matrix Transformation with Before and After Planes')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Draw faded static grid lines (before transformation)
for (start, end) in grid_lines_all:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', linestyle='--', alpha=0.3)

# Initialize animated grid points (dots) and connected lines
grid_dots, = ax.plot([], [], 'ko', markersize=3, label='Grid Points')
animated_grid_lines = [ax.plot([], [], color='gray', linestyle='--', linewidth=1, alpha=0.5)[0] for _ in grid_lines_all]

# Initialize vector arrows and lines
quivers = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                     color=colors[i % len(colors)], label=f'Vector {i+1}')
           for i, v in enumerate(vectors)]
initial_lines = [ax.plot([0, v[0]], [0, v[1]], color=colors[i % len(colors)], linestyle='-', linewidth=2, alpha=0.5)[0]
                 for i, v in enumerate(vectors)]
previous_positions = [[] for _ in vectors]
previous_lines = [ax.plot([], [], '--', color=colors[i % len(colors)], alpha=0.5, linewidth=2)[0]
                  for i in range(len(vectors))]

# Text annotations
initial_coords = [ax.text(v[0], v[1], f'({v[0]}, {v[1]})', fontsize=16, color=colors[i % len(colors)],
                          ha='right', va='bottom', fontfamily='Times New Roman', fontweight='bold')
                  for i, v in enumerate(vectors)]
final_coords = [ax.text(0, 0, '', fontsize=16, color=colors[i % len(colors)],
                        ha='left', va='top', fontfamily='Times New Roman', fontweight='bold')
                for i in range(len(vectors))]

# Add plane for before transformation (static)
before_plane = ax.fill_between([-grid_size, grid_size], -grid_size, grid_size, color='blue', alpha=0.1, label='Before Plane')

# Initialize after plane (will be represented by the transformed grid points)
# after_plane = ax.fill_between([], [], color='red', alpha=0.1, label='After Plane')  # Commented out

# Pause frames at the beginning
def initial_pause():
    for _ in range(10):  # 10 frames for a short pause
        yield None

# Update function for animation
def update(frame):
    if frame is None:
        return *quivers, grid_dots, *animated_grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines

    alpha = frame / 90
    interpolated_matrix = np.eye(2) * (1 - alpha) + transformation_matrix * alpha

    # Transform grid points and update dots
    transformed_grid = interpolated_matrix @ grid_points
    grid_dots.set_data(transformed_grid[0], transformed_grid[1])

    # Update connected grid lines
    for i, ((start, end), line) in enumerate(zip(grid_lines_all, animated_grid_lines)):
        p1 = interpolated_matrix @ np.array(start)
        p2 = interpolated_matrix @ np.array(end)
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # Transform and update vectors
    transformed_vectors = [interpolated_matrix @ v for v in vectors]
    for i, (quiver, vec) in enumerate(zip(quivers, transformed_vectors)):
        quiver.set_UVC(vec[0], vec[1])
        previous_positions[i].append((vec[0], vec[1]))
        x_vals, y_vals = zip(*previous_positions[i])
        previous_lines[i].set_data(x_vals, y_vals)

    # Update after plane (represented by the transformed grid points)
    # No need to explicitly create an after_plane; the transformed grid points already represent it.

    if frame == 90:
        for coord, vec in zip(final_coords, transformed_vectors):
            coord.set_position((vec[0], vec[1]))
            coord.set_text(f'({vec[0]:.2f}, {vec[1]:.2f})')

    return *quivers, grid_dots, *animated_grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines

# Run animation
def run_animation():
    for pos in previous_positions:
        pos.clear()
    for txt in final_coords:
        txt.set_text('')
    ani = FuncAnimation(fig, update, frames=[*initial_pause(), *np.arange(0, 91, 1)],
                        interval=speed, blit=True, repeat=False)
    plt.legend()
    plt.show()

run_animation()