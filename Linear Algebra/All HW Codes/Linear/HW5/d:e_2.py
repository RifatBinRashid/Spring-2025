import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def compute_svd_via_ata(A, tol=1e-10):
    """
    Given an m x d matrix A, compute the spectral decomposition of A^T A.
    Returns:
      U: d x d orthonormal matrix of eigenvectors for A^T A.
      SigmaT_Sigma: d x d diagonal matrix with eigenvalues.
      U1: d x r matrix (eigenvectors corresponding to positive eigenvalues).
      Sigma1: r x r diagonal matrix (with positive singular values in descending order).
      V1: m x r matrix computed as V1 = A U1 Sigma1^{-1}.
    """
    ATA = A.T @ A
    # Compute eigen-decomposition (ATA is symmetric)
    eigenvalues, U = np.linalg.eigh(ATA)
    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    # Form Sigma^T Sigma (diagonal matrix)
    SigmaT_Sigma = np.diag(eigenvalues)
    
    # Identify the positive eigenvalues (above tolerance)
    pos_idx = eigenvalues > tol
    r = np.sum(pos_idx)
    U1 = U[:, :r]
    singular_values = np.sqrt(eigenvalues[:r])
    Sigma1 = np.diag(singular_values)
    
    # Compute V1 using: A U1 = V1 Sigma1  =>  V1 = A U1 Sigma1^{-1}
    V1 = A @ U1 @ np.linalg.inv(Sigma1)
    
    return U, SigmaT_Sigma, U1, Sigma1, V1

def rank_k_approximation(A, U1, Sigma1, V1, k):
    """
    Given SVD factors for A (with U1, Sigma1, V1 corresponding to the r positive singular values),
    compute the rank-k approximation: A_k = sum_{i=1}^k sigma_i v_i u_i^T.
    """
    U1_k = U1[:, :k]
    Sigma1_k = Sigma1[:k, :k]
    V1_k = V1[:, :k]
    A_k = V1_k @ Sigma1_k @ U1_k.T
    return A_k

def main():
    # Use tkinter's file dialog to ask for .npy files
    root = Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select image .npy files", 
                                             filetypes=[("NumPy Files", "*.npy")])
    if not file_paths:
        print("No files selected.")
        return
    
    # Define the k values for approximation
    ks = [4, 20]  # And we'll also use full rank (r)
    
    # Prepare a figure for plotting approximations for each image.
    n_files = len(file_paths)
    fig, axes = plt.subplots(nrows=n_files, ncols=3, figsize=(12, 4 * n_files))
    
    # If there's only one file, axes will be 1D, so ensure we have 2D array for consistency:
    if n_files == 1:
        axes = np.array([axes])
    
    for i, path in enumerate(file_paths):
        # Load the image matrix from .npy file
        A = np.load(path)
        m, d = A.shape
        
        # Compute SVD factors via ATA
        U, SigmaT_Sigma, U1, Sigma1, V1 = compute_svd_via_ata(A)
        r = Sigma1.shape[0]  # effective rank
        
        # Reconstruct the full image from the truncated SVD (should equal A)
        A_full = V1 @ Sigma1 @ U1.T
        
        # Compute rank-k approximations for each chosen k and for full rank (r)
        approximations = {}
        for k in ks:
            approximations[k] = rank_k_approximation(A, U1, Sigma1, V1, k)
        approximations[r] = A_full  # full reconstruction
        
        # Set column titles
        col_titles = {4: "k = 4", 20: "k = 20", r: "full rank (r)"}
        # Plot: Column 1 for k=4, Column 2 for k=20, Column 3 for full rank.
        for j, k in enumerate([4, 20, r]):
            ax = axes[i, j]
            # Display the image approximation in grayscale.
            ax.imshow(approximations[k], cmap='gray', aspect='equal')
            ax.axis('off')
            if i == 0:
                ax.set_title(col_titles[k], fontsize=14)
            ax.set_ylabel(f"Image {i+1}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
