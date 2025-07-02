import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def histogram_equalization2(image_path, image_name):
    input_image = cv2.imread(image_path)

    if len(input_image.shape)==2:
        # If the image is grayscale, convert it to RGB
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # Histogram equalization
        equ = cv2.equalizeHist(img_rgb)
    elif len(input_image.shape)==3:
        # If the image is already RGB, just convert it to RGB
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Split the RGB image into its individual channels
        r, g, b = cv2.split(img_rgb)

        # Apply histogram equalization to each channel
        equalized_r = cv2.equalizeHist(r)
        equalized_g = cv2.equalizeHist(g)
        equalized_b = cv2.equalizeHist(b)

        # Merge the equalized channels back into an RGB image
        equ = cv2.merge((equalized_r, equalized_g, equalized_b))


    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(equ)
    plt.title('Histogram Equalized Image')
    plt.axis('off')
    plt.tight_layout()

      # Save the comparison figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    comparison_output_path = os.path.join(script_directory, "output", f"Histogram_Equalized_Image_part2_{image_name}_Comparison.jpg")
    plt.savefig(comparison_output_path)
    plt.show()

    equalized_output_path = os.path.join(script_directory, "output", f"Histogram_Equalized_Image_part2_{image_name}_Equalized.jpg")
    output_image_bgr = cv2.cvtColor(equ, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(equalized_output_path, output_image_bgr)
    
    plt.show()

    return equ

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_name = "Low_Contrast.jpg"
    image_path = os.path.join(script_directory, "images", image_file_name) # Assuming 'images' subdirectory
    
    # Make sure the 'images' directory exists and contains the image
    output_image = histogram_equalization2(image_path, image_name="Low_Contrast")