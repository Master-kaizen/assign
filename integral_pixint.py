import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image_path = 'sample_image.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at '{image_path}'. Please check the path and try again.")

# Display the image
plt.figure(figsize=(6, 6))
plt.title("Grayscale Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Calculate total pixel intensity (sum of all pixel values)
total_intensity = np.sum(image)

# Calculate average pixel intensity (mean of all pixel values)
average_intensity = np.mean(image)

# Print results
print(f"Total Pixel Intensity: {total_intensity:.2f}")
print(f"Average Pixel Intensity: {average_intensity:.2f}")
