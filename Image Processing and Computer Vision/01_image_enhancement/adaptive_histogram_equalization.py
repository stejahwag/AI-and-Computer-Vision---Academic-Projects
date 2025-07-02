import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def adaptive_histogram_equalization(image_path, clip_limit=2.0, grid_size=(7, 7), image_name="Low_Contrast"):
    # Read the image
    input_image = cv2.imread(image_path)
    

    if len(input_image.shape)==2:
        # If the image is grayscale, convert it to RGB
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        equalized_img = clahe.apply(img_rgb)

    elif len(input_image.shape)==3:
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        r, g, b = cv2.split(img_rgb)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        equalized_r = clahe.apply(r)
        equalized_g = clahe.apply(g)
        equalized_b = clahe.apply(b)

        equalized_img = cv2.merge((equalized_r, equalized_g, equalized_b))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img)
    plt.title('Adaptive Histogram Equalized Image')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory,"output", f"AdaptiveHistogramEqualization_{image_name}_Comparison.jpg")
    plt.savefig(output_path)
    
    # Save the brightened image
    equalized_img=cv2.cvtColor(equalized_img, cv2.COLOR_RGB2BGR)
    equalized_img_output_path = os.path.join(script_directory, "output", f"AdaptiveHistogramEqualization_{image_name}.jpg")
    cv2.imwrite(equalized_img_output_path, equalized_img)

    plt.show()

    return equalized_img

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))

    image_file_name = "Low_Contrast.jpg"
    image_name = "Low_Contrast"
    image_path = os.path.join(script_directory, "images", image_file_name) # Assuming 'images' subdirectory
    
    # Make sure the 'images' directory exists and contains the image
    adaptive_histogram_equalization(image_path ,clip_limit=3, grid_size=(7,7))

    