import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the .mat file. Replace the file path with your actual file.
mat_contents = loadmat('/Users/rifatbinrashid/Documents/VS_Code/Python/Linear/HW5/columbia256.mat')

# If you know the variable name in the .mat file (e.g., 'img_array'), you can do:
# img_array = mat_contents['img_array']
# Otherwise, extract the first variable that does not start with '__'
img_array = None
for key in mat_contents:
    if not key.startswith('__'):
        img_array = mat_contents[key]
        break

if img_array is None:
    raise ValueError("No valid variable found in the .mat file.")

plt.imshow(img_array, cmap='gray')
plt.show()
