import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy.ndimage import laplace
import os
from pathlib import Path
from math import ceil


def gaussian_smoothing(image, sigma=1):
    """
    Apply Gaussian smoothing to an input image.

    Parameters:
        image (np.ndarray): Grayscale input image.
        sigma (float): Standard deviation for the Gaussian kernel.
        kernel_size (int): Size of the Gaussian kernel
    Returns:
        smoothed_image (np.ndarray): The blurred image.
    """
    
    kernel_size = 2 * ceil(3 * sigma) + 1
    
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return smoothed_image


def sobel_X(image):
    """
    Compute the horizontal gradient using the Sobel operator.

    Parameters:
        image (np.ndarray): Grayscale image (2D).

    Returns:
        Gx (np.ndarray): Gradient in the x-direction.
    """
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    Gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    return Gx

def sobel_Y(image):
    """
    Compute the vertical gradient using the Sobel operator.

    Parameters:
        image (np.ndarray): Grayscale image (2D).

    Returns:
        Gy (np.ndarray): Gradient in the y-direction.
    """
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    Gy = convolve2d(image, kernel_y, mode='same', boundary='symm')
    return Gy


def gradient_magnitude(Gx, Gy):
    """
    Compute the gradient magnitude from horizontal and vertical components.

    Parameters:
        Gx (np.ndarray): Gradient in x-direction.
        Gy (np.ndarray): Gradient in y-direction.

    Returns:
        magnitude (np.ndarray): Gradient magnitude image.
    """
    return np.sqrt(Gx**2 + Gy**2)

def suppressNormgx(Gx, Gy, magnitude):
    """
    Perform non-maximum suppression.

    Parameters:
        Gx (np.ndarray): Gradient in x-direction.
        Gy (np.ndarray): Gradient in y-direction.
        magnitude (np.ndarray): Gradient magnitude.

    Returns:
        is_local_max (np.ndarray): Boolean array where True indicates local maxima.
    """
    H, W = magnitude.shape
    is_local_max = np.ones((H, W), dtype=bool)

    # Compute gradient direction in degrees
    angle = np.arctan2(Gy, Gx) * (180.0 / np.pi)
    angle = angle % 180  # Limit angle to [0, 180)

    # Quantize angle to 4 directions
    angle_quant = np.zeros_like(angle)
    angle_quant[(angle >= 0) & (angle < 22.5) | (angle >= 157.5) & (angle < 180)] = 0  # 0°
    angle_quant[(angle >= 22.5) & (angle < 67.5)] = 45  # 45°
    angle_quant[(angle >= 67.5) & (angle < 112.5)] = 90  # 90°
    angle_quant[(angle >= 112.5) & (angle < 157.5)] = 135  # 135°

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255

            if angle_quant[i, j] == 0:
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif angle_quant[i, j] == 45:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]
            elif angle_quant[i, j] == 90:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            elif angle_quant[i, j] == 135:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] < q or magnitude[i, j] < r:
                is_local_max[i, j] = False

    return is_local_max

def double_thresholding(nms_result, low_thresh_ratio=0.05, high_thresh_ratio=0.15):
    """
    Apply double thresholding to the NMS result.

    Parameters:
        nms_result (np.ndarray): Suppressed gradient magnitudes.
        low_thresh_ratio (float): Ratio of max value for low threshold.
        high_thresh_ratio (float): Ratio of max value for high threshold.

    Returns:
        thresholded (np.ndarray): Image with 0 (non-edge), 75 (weak), 255 (strong).
    """
    high_thresh = nms_result.max() * high_thresh_ratio
    low_thresh = high_thresh * low_thresh_ratio

    strong = 255
    weak = 75

    thresholded = np.zeros_like(nms_result, dtype=np.uint8)

    strong_i, strong_j = np.where(nms_result >= high_thresh)
    weak_i, weak_j = np.where((nms_result <= high_thresh) & (nms_result >= low_thresh))

    thresholded[strong_i, strong_j] = strong
    thresholded[weak_i, weak_j] = weak

    return thresholded


def hysteresis(thresholded_img, weak_val=75, strong_val=255):
    """
    Perform edge tracking by hysteresis.

    Parameters:
        thresholded_img (np.ndarray): Output from double thresholding.
        weak_val (int): Pixel value for weak edges.
        strong_val (int): Pixel value for strong edges.

    Returns:
        final_edges (np.ndarray): Image with only strong edges preserved.
    """
    H, W = thresholded_img.shape
    final_edges = thresholded_img.copy()

    # Define 8-connected neighborhood kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    while True:
        # Find all weak pixels
        weak_mask = (final_edges == weak_val)

        # Count strong neighbors using convolution
        strong_neighbors = convolve((final_edges == strong_val).astype(np.uint8), kernel, mode='constant')

        # Promote weak pixels that are connected to strong edges
        to_strong = weak_mask & (strong_neighbors > 0)

        if not np.any(to_strong):
            break

        final_edges[to_strong] = strong_val

    # Suppress remaining weak edges
    final_edges[final_edges == weak_val] = 0

    return final_edges


if __name__== '__main__':

    script_directory = os.path.dirname(os.path.abspath(__file__))

    im = cv2.imread(os.path.join(script_directory, 'data', 'images', 'gates.png'), cv2.COLOR_BGR2GRAY)

    img_float = im.astype(float)/255

    gaussian_sigma=2

    low_thresh_ratio=0.1   #0.1
    high_thresh_ratio=0.15  #0.15

    output_path = os.path.join(script_directory, "output", "gaussian_kernel_size={}  gaussian_sigma={} low_thresh_ratio={} high_thresh_ratio={}".format(2 * ceil(3 * gaussian_sigma) + 1, gaussian_sigma, low_thresh_ratio, high_thresh_ratio))
    Path(output_path).mkdir(parents= True,exist_ok=True)


    img_blur = gaussian_smoothing(img_float, gaussian_sigma)

    # # Step 1: Compute gradients
    Gx = sobel_X(img_blur)
    Gy = sobel_Y(img_blur)

    # # Step 2: Compute gradient magnitude
    magnitude = gradient_magnitude(Gx, Gy)

    theta_mask = suppressNormgx(Gx, Gy, magnitude)
    nms_result = np.zeros_like(magnitude)
    nms_result[theta_mask] = magnitude[theta_mask]

    dt_result = double_thresholding(nms_result, low_thresh_ratio, high_thresh_ratio)
    
    final_edges = hysteresis(dt_result)

    # plot images
    # original image
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.tight_layout()    
    file_name = '00_original_file.jpg'
    plt.savefig(os.path.join(output_path,file_name))

    # Gaussian blurred image
    plt.figure()
    plt.imshow(img_blur, cmap='gray')
    plt.title("Gaussian Blurred image sigma={} and kernel={}".format(gaussian_sigma, 2 * ceil(3 * gaussian_sigma) + 1))
    plt.axis('off')
    plt.tight_layout()    
    file_name = '01_gaussian_blurred.jpg'
    plt.savefig(os.path.join(output_path,file_name))

    # First order edge detector
    plt.figure()
    plt.imshow(magnitude, cmap='gray')
    plt.title("First order edge detector on Gaussian blurred image")
    plt.axis('off')
    plt.tight_layout()    
    file_name = '02_first_order_edge_detector.jpg'
    plt.savefig(os.path.join(output_path,file_name))

    # non-maximum suppressed image
    plt.figure()
    plt.imshow(nms_result, cmap='gray')
    plt.title("Non maximum suppressed image")
    plt.axis('off')
    plt.tight_layout()    
    file_name = '03_non_maximum_suppressed_image.jpg'
    plt.savefig(os.path.join(output_path,file_name))    

    # double thresholding
    plt.figure()
    plt.imshow(dt_result, cmap='gray')
    plt.title("Double Thresholding low_thresh_ratio={} and high_thresh_ratio={}".format(low_thresh_ratio, high_thresh_ratio))
    plt.axis('off')
    plt.tight_layout()

    file_name = '04_double_thresholding.jpg'
    plt.savefig(os.path.join(output_path,file_name))    

    # hysteresis
    plt.figure()
    plt.imshow(final_edges, cmap='gray')
    plt.title("Edge detection using hysteresis")
    plt.axis('off')    
    plt.tight_layout()
    file_name = '05_edge_Detection_hysteresis.jpg'
    plt.savefig(os.path.join(output_path,file_name))    

    # plt.show()