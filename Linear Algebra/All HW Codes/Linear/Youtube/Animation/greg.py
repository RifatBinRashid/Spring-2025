import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle
import matplotlib.cm as cm
import pandas

# Matrix setup
A = np.array([[4, .5, 0],
              [0, 0, 1],
              [2, 0, -3]])

# Gershgorin disc centers and radii
centers = np.diag(A)
radii = np.sum(np.abs(A), axis=1) - np.abs(centers)

# Eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Format cleaner numbers
def format_num(x):
    return f"{int(x)}" if x == int(x) else f"{x:.2f}"

# Setup figure
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)
ax.set_title("Gershgorin Discs and Eigenvalues", fontsize=16)
ax.set_xlabel("Re(z)", fontsize=12)
ax.set_ylabel("Im(z)", fontsize=12)
ax.grid(True)

# Axis limits
all_real = np.concatenate([np.real(centers), np.real(eigenvalues)])
all_imag = np.concatenate([np.imag(centers), np.imag(eigenvalues)])
buffer = max(radii) + 1
ax.set_xlim(min(all_real) - buffer, max(all_real) + buffer)
ax.set_ylim(min(all_imag) - buffer, max(all_imag) + buffer)

# Buttons
ax_disc = plt.axes([0.25, 0.05, 0.2, 0.075])
ax_eig = plt.axes([0.55, 0.05, 0.2, 0.075])
btn_disc = Button(ax_disc, 'Toggle Discs', color='lightblue')
btn_eig = Button(ax_eig, 'Toggle Eigenvalues', color='lightgreen')

# Text box for eigenvalues
eigenvalue_box = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         visible=False)

# Plot element storage
discs = []
disc_labels = []
center_dots = []
eig_plot = None
eig_labels = []
eigenvalue_box_visible = False

# Colors for disc fill
color_map = cm.get_cmap('Pastel1', len(centers))

def show_discs(event):
    global discs, disc_labels, center_dots
    if not discs:
        for i, (center, radius) in enumerate(zip(centers, radii)):
            disc = Circle((center.real, center.imag), radius,
                          facecolor=color_map(i), edgecolor='blue',
                          alpha=0.25, lw=2, ls='--')
            ax.add_patch(disc)
            discs.append(disc)

            # Mark center point
            dot = ax.plot(center.real, center.imag, 'bo', markersize=5)
            center_dots.append(dot[0])

            # Labels
            cx, cy = format_num(center.real), format_num(center.imag)
            label1 = ax.text(center.real, center.imag + 0.25,
                             f'D{i+1}', fontsize=10, ha='center', color='blue')
            label2 = ax.text(center.real, center.imag - 0.25,
                             f'({cx}, {cy})', fontsize=8, ha='center', color='blue')
            disc_labels.extend([label1, label2])
    else:
        visible = discs[0].get_alpha() > 0
        for disc in discs:
            disc.set_alpha(0 if visible else 0.25)
        for label in disc_labels:
            label.set_alpha(0 if visible else 1)
        for dot in center_dots:
            dot.set_alpha(0 if visible else 1)
    plt.draw()

def show_eigenvalues(event):
    global eig_plot, eig_labels, eigenvalue_box_visible

    if eig_plot is None:
        eig_plot = ax.scatter(eigenvalues.real, eigenvalues.imag,
                              color='red', s=100, label='Eigenvalues', marker='x')
        for i, eig in enumerate(eigenvalues):
            label = ax.text(eig.real + 0.2, eig.imag + 0.2,
                            f'λ{i+1}', fontsize=10, color='darkred')
            eig_labels.append(label)

        # Summary box
        text = "Eigenvalues:\n"
        for i, eig in enumerate(eigenvalues):
            re, im = format_num(eig.real), format_num(eig.imag)
            text += f"λ{i+1} = ({re}, {im})\n"
        eigenvalue_box.set_text(text)
        eigenvalue_box.set_visible(True)
        ax.legend()
        eigenvalue_box_visible = True
    else:
        eig_plot.remove()
        eig_plot = None
        for label in eig_labels:
            label.remove()
        eig_labels.clear()
        eigenvalue_box.set_visible(False)
        eigenvalue_box_visible = False
        if ax.get_legend():
            ax.get_legend().remove()
    plt.draw()

# Connect buttons
btn_disc.on_clicked(show_discs)
btn_eig.on_clicked(show_eigenvalues)

plt.show()
