
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from contrast import contrast_image_rgb
from Brighten import brighten_image_rgb


def plot_histogram(original_image_file, enhanced_image_file, description):

    original_image = cv2.imread(original_image_file)
    enhanced_image = cv2.imread(enhanced_image_file)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)     

    original_image_channels = cv2.split(original_image)
    enhanced_image_channels = cv2.split(enhanced_image)

    if len(original_image_channels) != len(enhanced_image_channels):
        raise ValueError("The number of channels in the original and enhanced images must be the same.")

    if len(original_image_channels) == 3:
        colors = ('r', 'g', 'b')
        subplot_rows = 3
    else:
        colors = ('gray',)
        subplot_rows = 1


    plt.figure(figsize=(20, 15))
    for i, color in enumerate(colors):

        plt.subplot(subplot_rows, 2, 2*i + 1)
        plt.hist(original_image_channels[i].ravel(), bins=256, range=[0,256], color=color, alpha=0.5)
        plt.title(f'Original Image Histogram - {color.upper()} Channel')

        plt.subplot(subplot_rows, 2, 2*i + 2)
        plt.hist(enhanced_image_channels[i].ravel(), bins=256, range=[0,256], color=color, alpha=0.5)
        plt.title(f'Enhanced Image Histogram - {color.upper()} Channel')        

    plt.tight_layout()

    # Save the figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "output", f"{description} Histogram_Comparison.jpg")
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    enhanced_image_file_name = input("Enter the enhanced image file name within output directory (e.g., Histogram_Equalized_Image_Part1.jpg): ")

    original_image_file = os.path.join(script_directory, "images", "Low_Contrast.jpg") # Assuming 'images' subdirectory
    enhanced_image_file = os.path.join(script_directory, "output", enhanced_image_file_name) # Assuming 'output' subdirectory

    description = input("Enter the enhanced_image_file description")  # Extracting the description from the file name
    
    plot_histogram(original_image_file, enhanced_image_file, description)