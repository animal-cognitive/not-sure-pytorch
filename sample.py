# Define file paths for two input images
image_path2 = 'Linaeus5_1.jpg'
image_path1 = 'Linaeus5_2.jpg'

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load your image (replace 'image_path' with the actual image path)
image = Image.open(image_path1)

# Define the desired image size and padding
img_size = 256
padding = 4

# Define the transformations
transform = transforms.Compose([
    transforms.RandomCrop(img_size, padding=padding),
    transforms.RandomHorizontalFlip(),
])

# Apply the transformations to the image
augmented_image = transform(image)

# # Convert the augmented image to a NumPy array
# augmented_image_np = np.array(augmented_image)

# # Save the augmented image using OpenCV
# cv2.imwrite('default_aug_.png', augmented_image_np)

# # Display the original and augmented images (optional)
# import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(image)
# ax1.set_title("Original Image")
# ax1.axis('off')
# ax2.imshow(augmented_image)
# ax2.set_title("Augmented Image")
# ax2.axis('off')
# plt.show()

# Apply the transformations to the image
augmented_image = transform(image)

# # Display the augmented image (optional)
# plt.imshow(augmented_image)
# plt.title("Augmented Image")
# plt.axis('off')

# # Define the path to save the augmented image
# output_path = 'augmented_image.jpg'

# # Save the augmented image using plt.savefig
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

# Convert the NumPy array to a PIL Image
# image = Image.fromarray((augmented_image ))  # Assuming it's a grayscale image

# Define the path to save the image
output_path = 'output_image.png'

# Save the image using PIL
augmented_image.save(output_path)

