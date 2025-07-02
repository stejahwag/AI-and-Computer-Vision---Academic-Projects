from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label
import matplotlib.pyplot as plt
import os
from pathlib import Path
from skimage import data

def histogram_segmentation(image):
    thresh = threshold_otsu(image)
    binary_hist = image > thresh

    # Optional: clean small artifacts
    cleaned = remove_small_objects(binary_hist, min_size=100)
    return cleaned

if __name__ =="__main__":
    
    image = data.coins()
    
    
    segmented_image = histogram_segmentation(image)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "output")
    Path(output_path).mkdir(parents= True,exist_ok=True)    

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image, cmap='gray')
    # plt.title("Histogram based segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    plt.contour(segmented_image, colors='r')
    # plt.title('Histogram segment overlayed on original image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'histogram_segmentation_result.jpg'), bbox_inches="tight", pad_inches=0)



    plt.figure()
    plt.imshow(segmented_image, cmap='gray')
    # plt.title("Histogram-based Segmentation")
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'histogram_segmented_result.jpg'), bbox_inches="tight", pad_inches=0)

    plt.show()
