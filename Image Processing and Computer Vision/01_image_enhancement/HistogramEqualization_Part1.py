import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def HistogramEqualization(input_image_path):
    input_image = cv2.imread(input_image_path)
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # img = input_image.copy()
    
    # Histogram equalization
    equ = cv2.equalizeHist(img)

    # Display the original and equalized images
    plt.figure(figsize=(15, 15))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 256, range=[0, 256])
    plt.title("Histogram of Original Image")

    plt.subplot(2, 2, 3)
    plt.imshow(equ, cmap='gray')
    plt.title("Histogram Equalized Image")
    plt.axis('off')


    plt.subplot(2, 2, 4)   
    plt.hist(equ.ravel(), 256, range=[0, 256])
    plt.title("Histogram of Equalized Image")
    plt.tight_layout()
    

   # Save the figure
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory,"output", f"Histogram_Equalized_Image_Part1_{input_image_path.split("/")[-1][:-4]}.jpg")
    plt.savefig(output_path)

    print(f"Histogram equalized image saved at: {output_path}")
    plt.show()



if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    file_name = "Checkerboard.png"
    # Make sure the 'images' directory exists and contains the image
    image_path = os.path.join(script_directory, "images", file_name) # Assuming 'images' subdirectory
    
    # Make sure the 'images' directory exists and contains the image
    HistogramEqualization(image_path)