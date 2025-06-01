import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Define the rotation angle (90 degrees in radians)
theta = np.pi / 2 # 90 degrees

# Define the initial vectors (you can modify this list to add more vectors)
vectors = [
    np.array([3, 4]),  # Vector 1: 3i + 4j
    np.array([2, -5]),  # Vector 2: 2i - 5j
]

# Define the grid
grid_size = 6
x = np.arange(-grid_size, grid_size + 1)
y = np.arange(-grid_size, grid_size + 1)
X, Y = np.meshgrid(x, y)  # Create the grid
grid_points = np.vstack([X.ravel(), Y.ravel()])  # Flatten the grid

# Speed parameter (lower value = faster animation)
speed = 50  # Change this to adjust animation speed (milliseconds per frame)

# Define colors for vectors dynamically
colors = ['r', 'g', 'b', 'm', 'c', 'y']  # Extend if needed

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-grid_size - 1, grid_size + 1)
ax.set_ylim(-grid_size - 1, grid_size + 1)
ax.grid(True)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('Rotation of Multiple Vectors with Initial and Previous Position Lines')

# Initialize the quiver plots (arrows) for the vectors
quivers = [ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colors[i % len(colors)], label=f'Vector {i+1}')
           for i, v in enumerate(vectors)]

# Draw the initial vector lines
initial_lines = [ax.plot([0, v[0]], [0, v[1]], color=colors[i % len(colors)], linestyle='-', linewidth=2, alpha=0.5)[0]
                 for i, v in enumerate(vectors)]

# Initialize the grid plot
grid_lines, = ax.plot(grid_points[0], grid_points[1], 'ko', markersize=3, label='Grid Points')

# Store the previous position lines
previous_positions = [[] for _ in vectors]  # Store previous positions for each vector
previous_lines = [ax.plot([], [], '--', color=colors[i % len(colors)], alpha=0.5, linewidth=2)[0] for i in range(len(vectors))]

# Initialize coordinate labels for initial and final positions
initial_coords = [ax.text(v[0], v[1], f'({v[0]}, {v[1]})', fontsize=16, color=colors[i % len(colors)],
                          ha='right', va='bottom', fontfamily='Times New Roman', fontweight='bold')
                  for i, v in enumerate(vectors)]
final_coords = [ax.text(0, 0, '', fontsize=16, color=colors[i % len(colors)],
                        ha='left', va='top', fontfamily='Times New Roman', fontweight='bold') for i in range(len(vectors))]

# Function to create an initial pause
def initial_pause():
    for _ in range(50):  # 20 frames for 2 seconds (speed adjustable)
        yield None

# Update function for the animation
def update(frame):
    if frame is None:  # Pause frames
        return *quivers, grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines
    
    # Calculate the rotation matrix for the current frame
    angle = theta * frame / 90  # Rotate from 0 to 90 degrees
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Apply the rotation matrix to the vectors
    rotated_vectors = [rotation_matrix @ v for v in vectors]

    # Apply the rotation matrix to the entire grid
    rotated_grid = rotation_matrix @ grid_points

    # Update the quiver plots
    for i, (quiver, rotated_vector) in enumerate(zip(quivers, rotated_vectors)):
        quiver.set_UVC(rotated_vector[0], rotated_vector[1])
        
        # Store previous positions
        previous_positions[i].append((rotated_vector[0], rotated_vector[1]))

        # Update previous position lines
        x_vals, y_vals = zip(*previous_positions[i])  # Unpack stored positions
        previous_lines[i].set_data(x_vals, y_vals)

    # Update the final coordinates at the end of the rotation
    if frame == 90:  # Final frame
        for coord, rotated_vector in zip(final_coords, rotated_vectors):
            coord.set_position((rotated_vector[0], rotated_vector[1]))
            coord.set_text(f'({rotated_vector[0]:.2f}, {rotated_vector[1]:.2f})')

    # Update the grid plot
    grid_lines.set_data(rotated_grid[0], rotated_grid[1])
    
    return *quivers, grid_lines, *previous_lines, *initial_coords, *final_coords, *initial_lines  # Return the updated Artist objects

# Function to run the animation
def run_animation():
    # Reset the previous positions and final coordinates
    for i in range(len(vectors)):
        previous_positions[i].clear()
        final_coords[i].set_text('')
    # Create the animation
    ani = FuncAnimation(fig, update, frames=[*initial_pause(), *np.arange(0, 91, 1)], interval=speed, blit=True, repeat=False)
    plt.show()

# Run the animation initially
run_animation()

# Wait for 5 seconds and run the animation again
#time.sleep(5)
#run_animation()