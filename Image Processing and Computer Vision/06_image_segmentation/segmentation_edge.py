from skimage.filters import sobel
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects
from skimage import data
import matplotlib.pyplot as plt
import os
from pathlib import Path


def edge_segmentation(image):
    # Edge detection
    edges = sobel(image)

    # Threshold edges
    edges_bin = edges > 0.1

    # Fill holes inside coins
    filled = binary_fill_holes(edges_bin)

    # closing to smooth shapes
    closed = closing(filled, square(3))

    # Remove small noisy objects
    cleaned_edges = remove_small_objects(closed, min_size=100)

    return cleaned_edges


if __name__=="__main__":

    image = data.coins()

    cleaned_edges = edge_segmentation(image)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "output")
    Path(output_path).mkdir(parents= True,exist_ok=True)



    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cleaned_edges, cmap='gray')
    # plt.title('Edge detected with Sobel algorithm')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    plt.contour(cleaned_edges, colors='r')
    # plt.title('Sobel algirith edges algorithm overlayed on original image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sobel_edge_segmentation_result_comparison.jpg'), bbox_inches="tight", pad_inches=0)

    plt.figure()
    plt.imshow(cleaned_edges, cmap='gray')
    # plt.title('Edge based segmented image using Sobel algorithm')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'sobel_edge_segmented_image.jpg'), bbox_inches="tight", pad_inches=0)

    plt.show()
