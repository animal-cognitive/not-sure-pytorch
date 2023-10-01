import cv2
import numpy as np
import random

def load_and_preprocess_image(image_path, target_size):
    """
    Load and preprocess an image from the given file path.

    Args:
        image_path (str): File path to the image.
        target_size (tuple): Target size (height, width) for resizing the image.

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, target_size)  # Resize to the target size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

def cutmix(image_path1, image_path2, output_path, beta=1.0):
    """
    Perform CutMix augmentation on two images and save the mixed image.

    Args:
        image_path1 (str): File path to the first image.
        image_path2 (str): File path to the second image.
        output_path (str): File path to save the mixed image.
        beta (float): Mixing ratio parameter (0.0 to 1.0). Default is 1.0.
    """
    # Load and preprocess the images
    image1 = load_and_preprocess_image(image_path1, (224, 224))
    image2 = load_and_preprocess_image(image_path2, (224, 224))

    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Generate random coordinates for the cutout region
    height, width, _ = image1.shape
    lam = np.random.beta(beta, beta)
    lam = max(lam, 1.0 - lam)
    cut_width = int(width * np.sqrt(1.0 - lam))
    cut_height = int(height * np.sqrt(1.0 - lam))
    cx = np.random.randint(0, width - cut_width + 1)
    cy = np.random.randint(0, height - cut_height + 1)

    # Create the mixed image
    mixed_image = image1.copy()
    mixed_image[cy:cy + cut_height, cx:cx + cut_width] = image2[cy:cy + cut_height, cx:cx + cut_width]

    # Save the mixed image
    mixed_image = (mixed_image * 255.0).astype(np.uint8)  # Convert back to uint8 format
    mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format
    cv2.imwrite(output_path, mixed_image)

# Example usage:
image_path2 = 'Linaeus5_1.jpg'
image_path1 = 'Linaeus5_2.jpg'
output_path = 'path_to_save_mixed_image.jpg'

# Specify the mixing parameter (beta)
beta = 0.5  # You can adjust this value

# Apply CutMix and save the mixed image
cutmix(image_path1, image_path2, output_path, beta)

# The mixed image is saved at the specified output_path.
