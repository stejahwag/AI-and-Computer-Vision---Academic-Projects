
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def gamma_correction(original_image_path, gamma, image_name):
    
    # Read the image
    input_img = cv2.imread(original_image_path)
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Normalize to the range [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Apply gamma correction
    corrected_img_normalized = np.power(img_normalized, gamma)

    # Denormalize to the range [0, 255] and convert to uint8
    corrected_img_uint8 = np.clip(corrected_img_normalized * 255.0, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(corrected_img_uint8)
    plt.title('Gamma Corrected Image with Gamma = {}'.format(gamma))
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory,"output", f"GammaCorrectedImage_{image_name}gamma{gamma}Comparison.jpg")
    plt.savefig(output_path)
    
    # Save the brightened image
    corrected_img_uint8=cv2.cvtColor(corrected_img_uint8, cv2.COLOR_RGB2BGR)
    gcorrected_output_path = os.path.join(script_directory, "output", f"GammaCorrectedImage_{image_name}_gamma{gamma}.jpg")
    cv2.imwrite(gcorrected_output_path, corrected_img_uint8)

    plt.show()

    return corrected_img_uint8

if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the 'images' directory exists and contains the image
    image_path = os.path.join(script_directory, "images", "Low_Contrast.jpg") # Assuming 'images' subdirectory
    image_name = "Low_Contrast"
    gamma = input("Enter the alpha value for brightness adjustment (e.g., 1.5): ")

    try:
        gamma = float(gamma)
    except ValueError:
        print("Invalid input. Using default alpha value of 1.5.")
        gamma = 1.5
    
    # Brighten the image
    gamma_correction(image_path, gamma, image_name)