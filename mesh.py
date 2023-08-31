import cv2
import numpy as np
import os

def combine_mask_and_depth(mask_path, depth_path, save_path):
    # Read mask and depth image from disk
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    print(f"Unique values in mask: {np.unique(mask)}")  # Debug line
    print(f"Unique values in depth: {np.unique(depth)}")  # Debug line
    
    # Convert both images to float type
    mask = mask.astype(np.float32) / 255.0
    depth = depth.astype(np.float32)

    # Combine the mask and depth image through element-wise multiplication
    combined = cv2.multiply(mask, depth)
    
    # Re-scale to 0-255 and convert to uint8
    combined = np.uint8(combined * 255)

    # Save the combined image
    cv2.imwrite(save_path, combined)

# Specify the directory where the mask and depth images are stored
folder_path = "data"

# List of image names
image_names = ['left', 'right', 'front']

# Loop through each image
for image_name in image_names:
    mask_path = os.path.join(folder_path, f"{image_name}_mask.png")
    depth_path = os.path.join(folder_path, f"depth_{image_name}.png")
    save_path = os.path.join(folder_path, f"mesh_{image_name}.png")  # Changed from "combined" to "mesh"

    if os.path.exists(mask_path) and os.path.exists(depth_path):
        combine_mask_and_depth(mask_path, depth_path, save_path)
        print(f"Mesh for {image_name} saved at {save_path}.")
    else:
        print(f"Mask or depth image missing for {image_name}.")
