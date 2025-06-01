import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk, messagebox
import os
import traceback

class NPYViewer:
    def __init__(self):
        self.root = Tk()
        self.root.withdraw()

    def load_npy_file(self):
        """Open file dialog and load .npy file with multiple fallback methods"""
        filepath = filedialog.askopenfilename(
            title="Select .npy file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if not filepath:
            return None, "No file selected"

        attempts = [
            {'allow_pickle': False, 'mmap_mode': None},
            {'allow_pickle': True, 'mmap_mode': None},
            {'allow_pickle': False, 'mmap_mode': 'r'},
            {'allow_pickle': True, 'mmap_mode': 'r'}
        ]

        last_error = None
        for attempt in attempts:
            try:
                data = np.load(
                    filepath,
                    allow_pickle=attempt['allow_pickle'],
                    mmap_mode=attempt['mmap_mode']
                )
                return data, None
            except Exception as e:
                last_error = e
                continue

        # Try raw binary loading as last resort
        try:
            with open(filepath, 'rb') as f:
                header = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, header)
                if dtype.fields is not None:
                    raise ValueError("Structured arrays not supported in fallback mode")
                data = np.fromfile(f, dtype=dtype)
                data = data.reshape(shape)
                return data, "Loaded via binary fallback"
        except Exception as e:
            last_error = e

        return None, f"All loading methods failed. Last error: {str(last_error)}"

    def visualize_data(self, data):
        """Create appropriate visualization based on data shape and type"""
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Array Visualization (shape: {data.shape}, dtype: {data.dtype})")

        if data.ndim == 1:
            plt.subplot(1, 2, 1)
            plt.plot(data)
            plt.title("1D Array")
            plt.xlabel("Index")
            plt.ylabel("Value")

            plt.subplot(1, 2, 2)
            plt.hist(data, bins=50)
            plt.title("Value Distribution")

        elif data.ndim == 2:
            plt.subplot(1, 2, 1)
            plt.imshow(data, cmap='viridis')
            plt.colorbar()
            plt.title("2D Array")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.hist(data.flatten(), bins=50)
            plt.title("Value Distribution")

        elif data.ndim == 3:
            if data.shape[2] == 3:  # RGB image
                plt.subplot(1, 2, 1)
                plt.imshow(data)
                plt.title("RGB Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                for i, color in enumerate(['red', 'green', 'blue']):
                    plt.hist(data[..., i].flatten(), bins=50, 
                            color=color, alpha=0.5, label=color)
                plt.legend()
                plt.title("Channel Distributions")
            else:  # 3D volume
                slices = [s//2 for s in data.shape]
                titles = ["XY Slice", "XZ Slice", "YZ Slice"]
                for i, (axis, sl) in enumerate(zip([2, 1, 0], slices)):
                    plt.subplot(1, 3, i+1)
                    if axis == 0:
                        plt.imshow(data[sl, :, :], cmap='gray')
                    elif axis == 1:
                        plt.imshow(data[:, sl, :], cmap='gray')
                    else:
                        plt.imshow(data[:, :, sl], cmap='gray')
                    plt.title(titles[i])
                    plt.axis('off')

        else:  # Higher dimensions
            plt.subplot(1, 1, 1)
            plt.hist(data.flatten(), bins=50)
            plt.title("Flattened Value Distribution")

        plt.tight_layout()
        plt.show()

    def show_data_info(self, data, filename):
        """Display array information in console"""
        print("\n" + "="*50)
        print(f"File: {filename}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {np.nanmin(data):.4f}, Max: {np.nanmax(data):.4f}")
        print(f"Mean: {np.nanmean(data):.4f}, Std: {np.nanstd(data):.4f}")
        print("="*50)

        if data.size < 100:  # Show full small arrays
            print("\nArray contents:")
            print(data)
        elif data.ndim == 1:
            print("\nFirst 10 elements:")
            print(data[:10])
        elif data.ndim == 2:
            print("\nTop-left 5x5 corner:")
            print(data[:5, :5])
        elif data.ndim == 3:
            print("\nFirst slice, top-left 3x3 corner:")
            print(data[0, :3, :3])

    def run(self):
        """Main application loop"""
        try:
            data, message = self.load_npy_file()
            if data is None:
                messagebox.showerror("Error", message)
                return

            filename = os.path.basename(self.root.tk.splitlist(filedialog.askopenfilenames())[0])
            self.show_data_info(data, filename)
            self.visualize_data(data)

            if message:
                messagebox.showinfo("Note", message)

        except Exception as e:
            messagebox.showerror("Critical Error", 
                               f"An unexpected error occurred:\n{str(e)}\n\n"
                               f"{traceback.format_exc()}")
        finally:
            self.root.destroy()

if __name__ == "__main__":
    viewer = NPYViewer()
    viewer.run()