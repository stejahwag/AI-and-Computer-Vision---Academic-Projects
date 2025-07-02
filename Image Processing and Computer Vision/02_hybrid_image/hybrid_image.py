import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from pathlib import Path

def low_pass_cv2(image, cutoff_frequency):
    kernel_size = int(cutoff_frequency * 6 + 1)  # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), cutoff_frequency)
    return blurred_image

def high_pass(image, cutoff_frequency):

    low_freq = low_pass_cv2(image, cutoff_frequency)
    high_freq = image.astype(np.float32) - low_freq.astype(np.float32)
    return high_freq


def compute_fft(image):

    fft_magnitude = np.zeros(image.shape[:2], dtype=np.float32)
    for i in range(3):  # Iterate over color channels
        fft_result = fft2(image[:, :, i])
        fft_shifted = fftshift(fft_result)
        fft_magnitude += np.log(np.abs(fft_shifted) + 1e-6)
    fft_magnitude /= 3.0  # Average the magnitudes of the channels
    return fft_magnitude

def vis_hybrid_image(input_image):

    original_height, original_width,channels = input_image.shape
    
    scale_factor = 0.5      # Factor to scale the image down
    
    combined_image = input_image.copy()

    for i in range(1, 5):
        # Create a new image background with the same height as the original
        # and a width scaled by the scale factor
        new_image_background = np.ones((original_height, int(original_width*scale_factor), channels))
        
        resized_image= cv2.resize(input_image, (int(original_width * scale_factor), int(original_height * scale_factor)))
        
        resized_h, resized_w = resized_image.shape[:2]
        
        start_row = original_height - resized_h
        
        new_image_background[start_row:, :resized_w] = resized_image

        separator = np.ones((original_height, 5, channels))
        
        combined_image = np.hstack((combined_image, separator, new_image_background))

        scale_factor *= 0.5
    
    return combined_image

if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image1_name = input("Enter the name of the first image for low pass filter: ")
    image2_name = input("Enter the name of the second image for high pass filter: ")
    # Load your two input images (can be color now)
    image1 = cv2.imread(os.path.join(script_directory, 'data', image1_name), cv2.IMREAD_COLOR_RGB)
    image2 = cv2.imread(os.path.join(script_directory, 'data', image2_name), cv2.IMREAD_COLOR_RGB)

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        min_height = min(image1.shape[0], image2.shape[0])
        min_width = min(image1.shape[1], image2.shape[1])
        image1_resized = cv2.resize(image1, (min_width, min_height))
        image2_resized = cv2.resize(image2, (min_width, min_height))
        image1 = image1_resized
        image2 = image2_resized

    # Convert to float32 and normalize to [0, 1]
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0

    # Choose your cutoff frequencies
    cutoff_freq_low = float(input("Enter the cutoff frequency for low-pass filter: "))
    cutoff_freq_high = float(input("Enter the cutoff frequency for high-pass filter: "))

    # Get the filtered images
    low_freq_image = low_pass_cv2(image1, cutoff_freq_low)
    high_freq_image = high_pass(image2, cutoff_freq_high)

    # Create the hybrid image
    hybrid_image = low_freq_image + high_freq_image

    # Clip pixel values
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

    # vis_hybrid_image
    scaled_images = vis_hybrid_image(hybrid_image)

    # Compute and display the log magnitude of the Fourier transforms
    fft_original1 = compute_fft(image1)
    fft_filtered_low = compute_fft(low_freq_image)
    fft_original2 = compute_fft(image2)
    fft_filtered_high = compute_fft(high_freq_image)
    fft_hybrid = compute_fft(hybrid_image)

    output_path = os.path.join(script_directory, "output")
    Path(output_path).mkdir(parents= True,exist_ok=True)

    ###########################
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(hybrid_image)
    plt.title('Hybrid Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fft_hybrid)
    plt.title('FFT (Hybrid Image)')
    plt.axis('off')
    file_name = image1_name.split('.')[0] + "_"+str(cutoff_freq_low) +"_"+ image2_name.split('.')[0]+ "_"+str(cutoff_freq_high) + "_hybrid.jpg"
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name))

    # Scaled images
    plt.figure(figsize=(15, 10))
    plt.imshow(scaled_images)
    plt.title('Scaled Images')
    plt.axis('off')
    file_name = image1_name.split('.')[0] + "_"+str(cutoff_freq_low) +"_"+ image2_name.split('.')[0]+ "_"+str(cutoff_freq_high) + "_scaled.jpg"
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name))

    ###########################
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title('Original Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image2)
    plt.title('Original Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hybrid_image)
    plt.title('Hybrid Image')
    plt.axis('off')
    plt.tight_layout()
    file_name = image1_name.split('.')[0] + "_"+str(cutoff_freq_low) +"_"+ image2_name.split('.')[0]+ "_"+str(cutoff_freq_high) + "_original_hybrid.jpg"
    plt.savefig(os.path.join(output_path, file_name))

    ###########################
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image1)
    plt.title('Original Image 1')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(low_freq_image)
    plt.title(f'Low-Pass Filtered (cutoff={cutoff_freq_low})')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(fft_original1)
    plt.title('FFT of Original Image 1')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(fft_filtered_low)
    plt.title('FFT of Low-Pass Image 1')
    plt.axis('off')

    plt.tight_layout()
    file_name = image1_name.split('.')[0] + "_"+str(cutoff_freq_low)+ "_original1_fft.jpg"
    plt.savefig(os.path.join(output_path, file_name))

    ###########################
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image2)
    plt.title('Original Image 2')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow((high_freq_image + 0.5))
    plt.title(f'High-Pass Filtered (cutoff={cutoff_freq_high})')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(fft_original2)
    plt.title('FFT of Original Image 1')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(fft_filtered_high)
    plt.title('FFT of Low-Pass Image 1')
    plt.axis('off')

    plt.tight_layout()
    file_name = image2_name.split('.')[0] + "_"+str(cutoff_freq_high)+ "_original2_fft.jpg"
    plt.savefig(os.path.join(output_path, file_name))

   
    plt.show()