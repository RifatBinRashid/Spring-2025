import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, messagebox
import os

def select_directory():
    """Open dialog to select output directory"""
    root = Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    root.destroy()
    return output_dir if output_dir else os.getcwd()

def load_npy_image():
    """Open dialog to select .npy image file"""
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select .npy Image File",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    root.destroy()
    if not filepath:
        return None, None
    img = np.load(filepath)
    if img.max() > 1:
        img = img / 255.0
    return img, os.path.basename(filepath)

def compute_svd(A):
    """Compute SVD components"""
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(s)
    return U, Sigma, Vh

def rank_k_approximation(U, Sigma, Vh, k):
    """Compute rank-k approximation"""
    return U[:, :k] @ Sigma[:k, :k] @ Vh[:k, :]

def save_results(original, approximations, titles, singular_values, base_name, output_dir):
    """Save all results to the specified directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save approximations
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(approximations)+1, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    for i, (img, title) in enumerate(zip(approximations, titles)):
        plt.subplot(1, len(approximations)+1, i+2)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
    
    approx_path = os.path.join(output_dir, f"{base_name}_approximations.png")
    plt.tight_layout()
    plt.savefig(approx_path)
    plt.close()
    
    # Plot and save singular values
    plt.figure(figsize=(10, 5))
    plt.plot(singular_values, 'bo-')
    plt.yscale('log')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('σ_i (log scale)')
    plt.grid(True)
    svals_path = os.path.join(output_dir, f"{base_name}_singular_values.png")
    plt.savefig(svals_path)
    plt.close()
    
    return approx_path, svals_path

def main():
    print("SVD Image Analysis Tool")
    print("----------------------")
    
    # Select input image
    img_data, filename = load_npy_image()
    if img_data is None:
        print("No file selected. Exiting.")
        return
    
    # Select output directory
    print("\nPlease select output directory for results...")
    output_dir = select_directory()
    if not output_dir:
        output_dir = os.getcwd()
    
    # Compute SVD
    U, Sigma, Vh = compute_svd(img_data)
    r = len(Sigma)
    singular_values = np.diag(Sigma)
    
    # Compute approximations
    k_values = [4, 20, r]
    approximations = [rank_k_approximation(U, Sigma, Vh, k) for k in k_values]
    titles = [f'Rank-{k} approx' for k in k_values]
    
    # Save results
    base_name = os.path.splitext(filename)[0]
    approx_path, svals_path = save_results(
        img_data, approximations, titles, singular_values, base_name, output_dir
    )
    
    # Show completion message
    root = Tk()
    root.withdraw()
    messagebox.showinfo(
        "Analysis Complete",
        f"Results saved to:\n{output_dir}\n\n"
        f"- Approximations: {os.path.basename(approx_path)}\n"
        f"- Singular values: {os.path.basename(svals_path)}"
    )
    root.destroy()
    
    print("\nAnalysis complete!")
    print(f"Original shape: {img_data.shape}")
    print(f"Matrix rank: {r}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()