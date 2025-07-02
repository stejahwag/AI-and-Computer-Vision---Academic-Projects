import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def contrast_image_rgb(image_path, beta, image_name):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32 for safe addition
    img_float = img.astype(np.float32)

    # Apply the contrast and brightness adjustment
    adjusted_img_float = img_float + beta

    # Clip the values to the valid range [0, 255]
    adjusted_img_clipped = np.clip(adjusted_img_float, 0, 255)

    # Convert back to uint8 for the final image
    contrast_img_uint8 = adjusted_img_clipped.astype(np.uint8)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(contrast_img_uint8)
    plt.title('Contrast Image with Beta = {}'.format(beta))
    plt.axis('off')
    plt.tight_layout()

    # Save the comparison figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    comparison_output_path = os.path.join(script_directory,"output", f"ContrastImage_{image_name}_beta{beta}_Comparison.jpg")
    plt.savefig(comparison_output_path)

    # Save the contrasted image
    contrast_img_uint8=cv2.cvtColor(contrast_img_uint8, cv2.COLOR_RGB2BGR)
    contrast_output_path = os.path.join(script_directory, "output", f"ContrastImage_{image_name}_beta{beta}.jpg")
    cv2.imwrite(contrast_output_path, contrast_img_uint8)

    plt.show()


    return img, contrast_img_uint8

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the 'images' directory exists and contains the image
    image_path = os.path.join(script_directory, "images", "Low_Contrast.jpg") # Assuming 'images' subdirectory
    image_name="Low_Contrast"
    
    beta = input("Enter the beta value for contrast adjustment (e.g., 15): ")
    try:
        beta = float(beta)
    except ValueError:
        print("Invalid input. Using default beta value of 15.")
        beta = 15
    # Brighten the image
    contrast_image_rgb(image_path, beta, image_name)