import torch, cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Create a RandAugment transform with desired parameters
rand_augment = transforms.RandAugment(3, 10)

# Load an example image (you should replace this with your own image loading logic)
image_path = 'Linaeus5_1.jpg'
image = Image.open(image_path)

# Apply RandAugment to the image
augmented_image = rand_augment(image)

# Convert the augmented image to a NumPy array
augmented_image_np = np.array(augmented_image)

# Save the augmented image using OpenCV
cv2.imwrite('randaug_.png', augmented_image_np)

# Display the original and augmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')
ax2.imshow(augmented_image)
ax2.set_title("Augmented Image")
ax2.axis('off')
plt.show()