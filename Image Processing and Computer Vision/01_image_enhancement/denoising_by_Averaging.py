import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def denoise_by_averaging(input_image_file, noisy_image_output_path, n_iterations, output_file_name):
    
    img = cv2.imread(input_image_file)

     # Convert to float32 for adding noise safely
    image_float = img.astype(np.float32)    

    noisy_images = []
    mean = 0
    std_dev = 16

    
    for i in range(n_iterations):
        
        filename = os.path.join(noisy_image_output_path, f"galaxy_noisy_image_{i}.tif")
        
        if not os.path.exists(filename):
            
            # Generate Gaussian noise
            noise = np.random.normal(mean, std_dev, image_float.shape)
            
            # Add noise to the image
            noisy_image = image_float + noise

            # Clip the values to be in the valid range [0, 255]
            noisy_image_clipped = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            # Save the noisy image
            cv2.imwrite(os.path.join(noisy_image_output_path, f"galaxy_noisy_image_{i}.tif"), noisy_image_clipped)

        else:
            # If the file exists, load it instead of re-generating
            noisy_image_clipped = cv2.imread(filename)
            noisy_images.append(noisy_image_clipped)
        
        # Append the noisy image to the list
        noisy_images.append(noisy_image_clipped)

    stacked_images = np.stack(noisy_images, axis=0)
    average_image = np.mean(stacked_images, axis=0)
    average_image_uint8 = np.clip(average_image, 0, 255).astype(np.uint8)

    


    mse_n = np.sum((image_float - average_image_uint8.astype("float")) ** 2)
    mse = mse_n / float(image_float.shape[0] * image_float.shape[1] * image_float.shape[2])

    cv2.imwrite(output_file_name, average_image_uint8)



    print("------------------------------------------------------------------------------------")
    print("Size of original image:", img.shape)
    print("Largest intensity value in original image and smallest pixel value in original image:", np.min(img), np.max(img))
    print("MSE between original and denoised average image:", mse)
    print("------------------------------------------------------------------------------------")

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(average_image_uint8, cv2.COLOR_BGR2RGB))
    plt.title('Denoised Average Image')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.2)
    plt.tight_layout()
    plt.show()
    
    # return average_image_uint8

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_file_name = "sombrero-galaxy-original.tif"
    image_path = os.path.join(script_directory, "images", image_file_name) # Assuming 'images' subdirectory

    number_of_iterations = int(input("Enter the number of noisy images to generate: "))
    output_file_name = os.path.join(script_directory, "output", f"denoised_average_image_iter_{number_of_iterations}.tif") # Assuming 'output' subdirectory
    noisy_image_output_path = os.path.join(script_directory, "noisy_galaxy_images") # Assuming 'images' subdirectory

    if not os.path.exists(noisy_image_output_path):
        os.makedirs(noisy_image_output_path)

    # Make sure the 'images' directory exists and contains the image
    output_image = denoise_by_averaging(image_path, noisy_image_output_path, number_of_iterations, output_file_name)