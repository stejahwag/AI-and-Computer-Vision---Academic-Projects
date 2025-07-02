import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import footprint_rectangle
from skimage.morphology import rectangle


from skimage.filters import threshold_otsu
from pathlib import Path
import os

if __name__ == "__main__":

    orig_phantom = img_as_ubyte(data.shepp_logan_phantom())

    # Define structuring element
    rectangle = footprint_rectangle([5,5])

    # Apply morphological operations
    eroded = erosion(orig_phantom, rectangle)
    dilated = dilation(orig_phantom, rectangle)
    opened = opening(orig_phantom, rectangle)
    closed = closing(orig_phantom, rectangle)
    white_top = white_tophat(orig_phantom, rectangle)
    black_top = black_tophat(orig_phantom, rectangle)


    # Binarize image for skeleton and convex hull
    thresh = threshold_otsu(orig_phantom)
    binary = orig_phantom > thresh
    skeleton = skeletonize(binary)
    convex_hull = convex_hull_image(binary)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "output")
    Path(output_path).mkdir(parents= True,exist_ok=True)



    # Image dictionary
    results = {
        "original_image": orig_phantom,
        "eroded": eroded,
        "dilated": dilated,
        "opened": opened,
        "closed": closed,
        "white_tophat": white_top,
        "black_tophat": black_top,
        "skeleton": skeleton,
        "convex_hull": convex_hull
    }

    # Save each image
    for name, image in results.items():
        plt.figure()
        plt.imshow(image, cmap="gray")
        # plt.title(name.replace("_", " ").title())
        plt.axis("off")
        file_name = f"{name}.jpg"
        plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight", pad_inches=0)
        plt.close()