import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image (grayscale)
image_path = 'sample_image.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at '{image_path}'. Please check the path and try again.")

# Display original image
plt.figure(figsize=(6, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()


# Function to scale the image
def scale_image(image, scale_factor):
    """Scales the image by the given factor using interpolation."""
    height, width = image.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return scaled_image


# Function to calculate pixel density changes using derivatives
def compute_pixel_density_change(original, scaled):
    """
    Computes the change in pixel density between the original and scaled images.
    The original image is resized to the size of the scaled image before comparison.
    """
    # Resize the original to match the scaled image dimensions
    resized_original = cv2.resize(original, (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Compute gradients for both images
    original_density = np.gradient(resized_original.astype(float))
    scaled_density = np.gradient(scaled.astype(float))

    # Calculate the change in density
    density_change = [scaled_density[i] - original_density[i] for i in range(2)]
    return density_change


# Scale the image (example: reduce by 50%, then increase by 200%)
scaled_down = scale_image(image, 0.5)  # Downscaled image
scaled_up = scale_image(image, 2.0)  # Upscaled image

# Compute pixel density changes
density_change_down = compute_pixel_density_change(image, scaled_down)
density_change_up = compute_pixel_density_change(image, scaled_up)

# Display scaled images and their pixel density changes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

# Scaled down image
axs[0, 1].imshow(scaled_down, cmap='gray')
axs[0, 1].set_title("Scaled Down (50%)")
axs[0, 1].axis('off')

# Scaled up image
axs[0, 2].imshow(scaled_up, cmap='gray')
axs[0, 2].set_title("Scaled Up (200%)")
axs[0, 2].axis('off')

# Density change (x-axis) for downscaled
axs[1, 0].imshow(np.abs(density_change_down[0]), cmap='viridis')
axs[1, 0].set_title("Density Change (X-axis, Downscaled)")
axs[1, 0].axis('off')

# Density change (y-axis) for downscaled
axs[1, 1].imshow(np.abs(density_change_down[1]), cmap='viridis')
axs[1, 1].set_title("Density Change (Y-axis, Downscaled)")
axs[1, 1].axis('off')

# Density change (x-axis) for upscaled
axs[1, 2].imshow(np.abs(density_change_up[0]), cmap='viridis')
axs[1, 2].set_title("Density Change (X-axis, Upscaled)")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
