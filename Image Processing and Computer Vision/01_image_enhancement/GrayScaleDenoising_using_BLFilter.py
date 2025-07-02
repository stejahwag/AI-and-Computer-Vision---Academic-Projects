import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from math import ceil

def apply_bilateral_filter_cv2(image, sigma_s, sigma_t):

    # Calculate the filter size based on sigma_s
    filter_size = int(2 * ceil(3 * sigma_s) + 1)
    # Apply cv2.bilateralFilter
    denoised_image = cv2.bilateralFilter(image, filter_size, sigma_t, sigma_s)

    return denoised_image

def estimate_noise_std(image, patch_size=5):
    rows, cols = image.shape
    variance_values = []
    # Iterate over a few random patches to find a low-variance one
    num_patches = 100
    for _ in range(num_patches):
        y = np.random.randint(0, rows - patch_size)
        x = np.random.randint(0, cols - patch_size)
        patch = image[y:y + patch_size, x:x + patch_size]
        variance = np.var(patch)
        variance_values.append(variance)

    # Choose the minimum variance as an estimate of noise variance
    noise_variance_estimate = np.min(variance_values) if variance_values else 0
    noise_std_estimate = np.sqrt(noise_variance_estimate)
    return noise_std_estimate


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, "images", "taj.jpg")

    # Load the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get user input for sigma_s
    sigma_s_str = input("Enter the value for sigma_s (start with 3, and increment by step 2): ")
    try:
        sigma_s = float(sigma_s_str)
    except ValueError:
        print("Invalid input for sigma_s. Using default value of 3.")
        sigma_s = 3.0

    # Estimate noise standard deviation
    estimated_noise_std = estimate_noise_std(grayscale_image)
    print(f"Estimated noise standard deviation: {estimated_noise_std:.2f}")

    # Calculate a suggested starting value for sigma_t
    suggested_sigma_t = 2 * estimated_noise_std
    sigma_t_str = input(f"Enter the value for sigma_t (suggested starting value: {suggested_sigma_t:.2f}): ")
    try:
        sigma_t = float(sigma_t_str)
    except ValueError:
        print(f"Invalid input for sigma_t. Using suggested value of {suggested_sigma_t:.2f}.")
        sigma_t = suggested_sigma_t

    # Apply OpenCV's bilateral filter
    denoised_image_cv2 = apply_bilateral_filter_cv2(grayscale_image, sigma_s, sigma_t)

    # Apply Gaussian filter for comparison
    blurred_image_gaussian = apply_bilateral_filter_cv2(grayscale_image, sigma_s, 99999)


    # Display the results
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    plt.subplot(1, 3, 1)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(denoised_image_cv2, cmap='gray')
    plt.title(f'Bilateral Filter ($\\sigma_s$={sigma_s:.2f}, $\\sigma_t$={sigma_t:.2f})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blurred_image_gaussian, cmap='gray')
    plt.title(f'Gaussian Filter ($\\sigma$={sigma_s:.2f})')
    plt.axis('off')

    plt.tight_layout()

    output_dir = os.path.join(script_directory, "output")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, f"denoising_comparison_s{sigma_s:.2f}_t{sigma_t:.2f}.jpg"))

    plt.show()

    # Save the results

    cv2.imwrite(os.path.join(output_dir, f"taj_gray_scaled.jpg"), grayscale_image)
    cv2.imwrite(os.path.join(output_dir, f"taj_denoised_cv2_s{sigma_s:.2f}_r{sigma_t:.2f}.jpg"), denoised_image_cv2)
    cv2.imwrite(os.path.join(output_dir, f"taj_gaussian_s{sigma_s:.2f}.jpg"), blurred_image_gaussian)
