import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
from scipy.io import loadmat

def load_matrix(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.npy':
        A = np.load(file)
    elif ext == '.mat':
        mat_data = loadmat(file)
        A = None
        for key in mat_data:
            if not key.startswith('__'):
                A = mat_data[key]
                break
        if A is None:
            raise ValueError("No valid variable found in the .mat file.")
    elif ext in ['.png', '.jpg', '.jpeg']:
        A = plt.imread(file)
        # If the image is in color (i.e., 3D array), convert to grayscale.
        if A.ndim == 3:
            # If there is an alpha channel (4 channels), ignore it.
            if A.shape[2] == 4:
                A = A[:, :, :3]
            # Convert to grayscale by taking the average of the channels.
            A = np.mean(A, axis=2)
    else:
        raise ValueError("Unsupported file type: " + ext)
    return A

def svd_via_ata(A, tol=1e-10):
    # Form A^T A
    ATA = A.T @ A
    # Compute spectral decomposition of ATA (ATA is symmetric)
    eigenvalues, U = np.linalg.eigh(ATA)
    # Sort eigenvalues (and corresponding eigenvectors) in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    # Form Σ^TΣ (diagonal matrix with eigenvalues)
    SigmaT_Sigma = np.diag(eigenvalues)
    
    # Determine rank r (number of positive eigenvalues above tolerance)
    r = np.sum(eigenvalues > tol)
    U1 = U[:, :r]
    singular_values = np.sqrt(eigenvalues[:r])
    Sigma1 = np.diag(singular_values)
    
    # Compute V1 using: A U1 = V1 Σ1  =>  V1 = A U1 Σ1^{-1}
    V1 = A @ U1 @ np.linalg.inv(Sigma1)
    
    return U, SigmaT_Sigma, U1, Sigma1, V1

def rank_k_approximation(U1, Sigma1, V1, k):
    U1_k = U1[:, :k]           # (d x k)
    Sigma1_k = Sigma1[:k, :k]    # (k x k)
    V1_k = V1[:, :k]           # (m x k)
    A_k = V1_k @ Sigma1_k @ U1_k.T
    return A_k

def plot_approximations(image_matrices, image_names, ks_list):
    n_images = len(image_matrices)
    n_cols = len(ks_list) + 1  # +1 for the original image
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4*n_cols, 4*n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, (A, name) in enumerate(zip(image_matrices, image_names)):
        # Compute SVD factors via A^T A method for the image A
        U, SigmaT_Sigma, U1, Sigma1, V1 = svd_via_ata(A)
        r = Sigma1.shape[0]  # effective rank
        approximations = {}
        approximations["original"] = A
        
        for k in ks_list:
            if isinstance(k, str) and k.lower() == "full":
                k_used = r
            else:
                k_used = k if k <= r else r
            approximations[k] = rank_k_approximation(U1, Sigma1, V1, k_used)
        
        col_keys = ["original"] + ks_list
        for j, key in enumerate(col_keys):
            ax = axes[i, j]
            if approximations[key].ndim == 2:
                ax.imshow(approximations[key], cmap='gray', aspect='equal')
            else:
                ax.imshow(approximations[key], aspect='equal')
            ax.axis('off')
            if i == 0:
                if key == "original":
                    title = "Original"
                elif isinstance(key, str) and key.lower() == "full":
                    title = f"full (r={r})"
                else:
                    title = f"k = {key}"
                ax.set_title(title, fontsize=14)
            ax.set_ylabel(name, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set up Tkinter file dialog to select image files (.npy, .mat, .png, .jpg, .jpeg)
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select image files (.npy, .mat, .png, .jpg)",
        filetypes=[("NumPy files", "*.npy"),
                   ("MAT files", "*.mat"),
                   ("PNG files", "*.png"),
                   ("JPEG files", "*.jpg *.jpeg")]
    )
    
    if not file_paths:
        print("No files selected.")
        return
    
    image_matrices = []
    image_names = []
    for file in file_paths:
        print("\n----------------------------------")
        print(f"Processing file: {file}")
        try:
            A = load_matrix(file)
        except Exception as e:
            print(f"Error loading file {file}:", e)
            continue
        print(f"Matrix shape: {A.shape}")
        image_matrices.append(A)
        image_names.append(os.path.basename(file))
    
    # Define desired k values for approximations (e.g., 4, 20, and "full")
    ks_list = [4, 50, 100, "full"]
    
    plot_approximations(image_matrices, image_names, ks_list)
    
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
