# # Import Dependencies
# import cv2
# import torch
# import os
# from matplotlib import pyplot as plt

# # Download the Midas model
# midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
# midas.to('cuda')
# midas.eval()

# # Input transformation Pipeline
# transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
# transform = transforms.small_transform

# # Define the folder and image names
# folder_path = "data"
# image_names = ["front.jpg", "left.jpg", "right.jpg"]

# # Loop through the list of image names
# for image_name in image_names:
#     image_path = os.path.join(folder_path, image_name)
    
#     # Read the image
#     frame = cv2.imread(image_path)
    
#     # Check if the image is loaded successfully
#     if frame is None:
#         print(f"Failed to load image from {image_path}.")
#         continue

#     # Transform input for Midas
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_batch = transform(img).to('cuda')

#     # Make a prediction
#     with torch.no_grad():
#         prediction = midas(img_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode='bicubic',
#             align_corners=False
#         ).squeeze()

#         output = prediction.cpu().numpy()

#     # Display the depth map
#     plt.figure()
#     plt.title(f"Depth Map for {image_name}")
#     plt.imshow(output, cmap='inferno')
#     plt.axis('off')
#     plt.show()


# Import Dependencies
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np

# Load the Midas model
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas.eval()

# Input transformation Pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# List of image names
image_names = ['front.jpg', 'left.jpg', 'right.jpg']

# Loop through the list of images to perform depth estimation
for image_name in image_names:
    image_path = f"data/{image_name}"
    
    # Read the image from disk
    img = cv2.imread(image_path)

    # Resize the image to 640x480
    img_resized = cv2.resize(img, (640, 480))

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Transform input for midas
    img_batch = transform(img_rgb)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(img_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(480, 640),
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Normalize the output to the range 0-255
    output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save or show the depth map
    plt.imshow(output_normalized, cmap='plasma')
    plt.title(f"Depth Map - {image_name}")
    plt.axis("off")
    plt.show()

    # Save as PNG
    cv2.imwrite(f"data/depth_{image_name.split('.')[0]}.png", output_normalized)

