import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from math import ceil


def gaussian_kernel_1d(sigma):
    size = 2 * ceil(3 * sigma) + 1
    center = size // 2
    x = np.arange(-center, center + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def apply_gaussian_filter(image, sigma_s):
    kernel_1d = gaussian_kernel_1d(sigma_s)
    blurred_rows = cv2.filter2D(image, -1, np.expand_dims(kernel_1d, axis=0))
    blurred_image = cv2.filter2D(blurred_rows, -1, np.expand_dims(kernel_1d, axis=1))
    return blurred_image



if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, "images", "taj.jpg")

    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get user input for sigma_s
    sigma_s_str = input("Enter the value for sigma_s (start with 3, and increment by step 2): ")
    try:
        sigma_s = float(sigma_s_str)
    except ValueError:
        print("Invalid input for sigma_s. Using default value of 3.")
        sigma_s = 3.0



    # Apply Gaussian filter
    blurred_image_gaussian = apply_gaussian_filter(grayscale_image, sigma_s)


    # Display the results
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image_gaussian, cmap='gray')
    plt.title(f'Gaussian Filter ($\sigma$={sigma_s:.2f})')
    plt.axis('off')

    plt.tight_layout()

    output_dir = os.path.join(script_directory, "output")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, f"gaussian_filter_s{sigma_s:.2f}_compared.jpg"))

    plt.show()

    # Save the results

    cv2.imwrite(os.path.join(output_dir, f"gaussian_filter_s{sigma_s:.2f}.jpg"), blurred_image_gaussian)