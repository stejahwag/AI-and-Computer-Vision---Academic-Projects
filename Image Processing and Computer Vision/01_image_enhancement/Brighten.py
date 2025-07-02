import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def brighten_image_rgb(image_path, alpha, image_name):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32 for safe addition
    img_float = img.astype(np.float32)

    # Apply the contrast and brightness adjustment
    adjusted_img_float = alpha * img_float

    # Clip the values to the valid range [0, 255]
    adjusted_img_clipped = np.clip(adjusted_img_float, 0, 255)

    # Convert back to uint8 for the final image
    adjusted_img_uint8 = adjusted_img_clipped.astype(np.uint8)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_img_uint8)
    plt.title('Brightened Image with Alpha = {}'.format(alpha))
    plt.axis('off')
    plt.tight_layout()
    

    # Save the figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory,"output", f"BrightenedImage_{image_name}_alpha{alpha}Comparison.jpg")
    plt.savefig(output_path)
    
    # Save the brightened image
    adjusted_img_uint8=cv2.cvtColor(adjusted_img_uint8, cv2.COLOR_RGB2BGR)
    brightened_output_path = os.path.join(script_directory, "output", f"BrightenedImage_{image_name}_alpha{alpha}.jpg")
    cv2.imwrite(brightened_output_path, adjusted_img_uint8)
    
    plt.show()


    return img, adjusted_img_uint8

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the 'images' directory exists and contains the image
    image_path = os.path.join(script_directory, "images", "Low_Contrast.jpg") # Assuming 'images' subdirectory
    image_name = "Low_Contrast"
    alpha = input("Enter the alpha value for brightness adjustment (e.g., 1.5): ")

    try:
        alpha = float(alpha)
    except ValueError:
        print("Invalid input. Using default alpha value of 1.5.")
        alpha = 1.5
    
    # Brighten the image
    brighten_image_rgb(image_path, alpha, image_name)