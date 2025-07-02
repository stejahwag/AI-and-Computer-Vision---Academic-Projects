import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import os
from pathlib import Path    

def pyramidsGL(im, N):
    """
    Constructs Gaussian and Laplacian pyramids of level N from image im.
    Returns:
        G: Gaussian pyramid (list of images)
        L: Laplacian pyramid (list of images)
    """
    G = [im.astype(np.float32)]  # Start with original image
    for i in range(1, N):
        im = cv2.pyrDown(im)  # Downsample image by factor of 2
        G.append(im.astype(np.float32))

    L = []
    for i in range(N - 1):
        # Upsample the next level and subtract from current level to get Laplacian
        size = (G[i].shape[1], G[i].shape[0])  # width, height
        GE = cv2.pyrUp(G[i + 1], dstsize=size)
        L.append(G[i] - GE)
    L.append(G[-1])  # The last Gaussian level is also the last Laplacian level

    return G, L

def displayPyramids(G, L, title_prefix=""):
    """
    Displays Gaussian and Laplacian pyramids as grayscale images.
    """
    N = len(G)
    plt.figure(figsize=(15, 6))
    for i in range(N):
        # Gaussian
        plt.subplot(2, N, i + 1)
        plt.imshow(np.clip(G[i] / 255, 0, 1))
        plt.title(f'{title_prefix} G[{i}]')
        plt.axis('off')

        # Laplacian
        plt.subplot(2, N, N + i + 1)
        norm_L = (L[i] - L[i].min()) / (L[i].max() - L[i].min() + 1e-6)  # Normalize for display
        plt.imshow(np.clip(norm_L, 0, 1))
        plt.title(f'{title_prefix} L[{i}]')
        plt.axis('off')
    plt.tight_layout()

def displayFFT(im, minv=0, maxv=8):
    """
    Displays the FFT amplitude spectrum of an image.
    minv, maxv: used to clip the spectrum for better visibility.
    """
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    fft_result = fft2(im)
    fft_shifted = fftshift(fft_result)  # Center low frequencies
    magnitude = np.log(np.abs(fft_shifted) + 1e-6)
    magnitude = np.clip(magnitude, minv, maxv)
    plt.imshow(magnitude)
    plt.title('FFT Amplitude')
    plt.axis('off')

def reconstruct_from_Laplacian_pyramid(L):
    """
    Reconstructs an image from its Laplacian pyramid.
    """
    current = L[-1]  # Start from the smallest level
    for i in range(len(L) - 2, -1, -1):
        size = (L[i].shape[1], L[i].shape[0])
        current = cv2.pyrUp(current, dstsize=size)
        current = cv2.add(current, L[i])
    return current


if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "output")
    Path(output_path).mkdir(parents= True,exist_ok=True)


    # Load images and mask
    img1 = cv2.imread(os.path.join(script_directory,'Data','jet.jpeg'), cv2.IMREAD_COLOR_RGB)
    img2 = cv2.imread(os.path.join(script_directory,'Data','cloud.jpg'), cv2.IMREAD_COLOR_RGB)
    mask = cv2.imread(os.path.join(script_directory,'Data','mask.jpg'), cv2.IMREAD_GRAYSCALE)   # Load mask as grayscale and normalize

    # Resize all to same size (e.g., 512x512)
    img1 = cv2.resize(img1, (512, 512)).astype(np.float32)
    img2 = cv2.resize(img2, (512, 512)).astype(np.float32)
    mask = cv2.resize(mask, (512, 512)).astype(np.float32)
    mask = mask / 255.0  # Normalize mask to [0, 1]

    # Convert mask to 3 channels for RGB blending
    mask = np.stack([mask]*3, axis=-1)

    # Number of pyramid levels
    N = 5

    # Generate Gaussian and Laplacian pyramids for both images and mask
    G1, L1 = pyramidsGL(img1, N)
    G2, L2 = pyramidsGL(img2, N)
    Gm, _ = pyramidsGL(mask, N)

    # Display image pyramids
    displayPyramids(G1, L1, "Image 1")
    plt.savefig(os.path.join(output_path, 'pyramids_image1.png'))
    displayPyramids(G2, L2, "Image 2")
    plt.savefig(os.path.join(output_path, 'pyramids_image2.png'))

    # Show FFTs of Laplacian pyramid levels for Image 1
    fig1 = plt.figure(figsize=(15, 5))
    fig1.suptitle("FFT Analysis of Laplacian Pyramid Image 1", fontsize=16)
    for i in range(N):
        plt.subplot(1, N, i + 1)
        displayFFT(L1[i], minv=0, maxv=15)
        plt.title(f'Level {i}')
    plt.tight_layout()
    fig1.savefig(os.path.join(output_path, 'fft_laplacian_pyramid_image_1.png'))
    # Show FFTs of Gaussian pyramid levels for Image 1
    
    fig2 = plt.figure(figsize=(15, 5))
    fig2.suptitle("FFT Analysis of Gaussian Pyramid Image 1", fontsize=16)
    for i in range(N):
        plt.subplot(1, N, i + 1)
        displayFFT(G1[i], minv=0, maxv=15)
        plt.title(f'Level {i}')
    plt.tight_layout()
    fig2.savefig(os.path.join(output_path,'fft_gaussian_pyramid_image_1.png'))

    # Show FFTs of Laplacian pyramid levels for Image 2
    fig3 = plt.figure(figsize=(15, 5))
    fig3.suptitle("FFT Analysis of Laplacian Pyramid Image 2", fontsize=16)
    for i in range(N):
        plt.subplot(1, N, i + 1)
        displayFFT(L2[i], minv=0, maxv=15)
        plt.title(f'Level {i}')
    plt.tight_layout()
    fig3.savefig(os.path.join(output_path, 'fft_laplacian_pyramid_image_2.png'))
    # Show FFTs of Gaussian pyramid levels for Image 2
    fig4 = plt.figure(figsize=(15, 5))
    fig4.suptitle("FFT Analysis of Gaussian Pyramid Image 2", fontsize=16)
    for i in range(N):
        plt.subplot(1, N, i + 1)
        displayFFT(G2[i], minv=0, maxv=15)
        plt.title(f'Level {i}')
    plt.tight_layout()
    fig4.savefig(os.path.join(output_path,'fft_gaussian_pyramid_image_2.png'))

    # Blend the Laplacian pyramids using the Gaussian mask
    blended_L = []
    for i in range(N):
        blended = Gm[i] * L1[i] + (1 - Gm[i]) * L2[i]
        blended_L.append(blended)

    # Reconstruct the final blended image
    blended_img = reconstruct_from_Laplacian_pyramid(blended_L)

    # Display original and blended images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img1.astype(np.uint8))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2.astype(np.uint8))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(blended_img / 255, 0, 1))
    plt.title('Blended Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path,'Original_blended_iamges.png'))
    plt.show()
