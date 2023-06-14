import numpy as np
from PIL import Image

# Load the image
image_path = "data/original_text/text_image0.png"  # Replace with the path to your image
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Define the region of interest (ROI) using a mask
roi_mask = np.zeros_like(image_array, dtype=np.uint8)
start_x, start_y = 10, 10  # Coordinates of the top-left corner of the ROI
end_x, end_y = 2000, 2000  # Coordinates of the bottom-right corner of the ROI
roi_mask[start_y:end_y, start_x:end_x] = 1

# Generate random noise for the ROI
roi_noise = np.random.normal(loc=0, scale=50, size=image_array.shape[:2])
roi_noise = np.expand_dims(roi_noise, axis=-1)  # Add a channel dimension for broadcasting
roi_noise = roi_noise.astype(np.uint8)

# Apply the noise to the ROI only
noisy_image_array = np.where(roi_mask, np.clip(image_array + roi_noise, 0, 255), image_array)

# Convert the NumPy array back to an image
noisy_image = Image.fromarray(noisy_image_array)

# Save the noisy image
noisy_image.save("noisy_image.png")