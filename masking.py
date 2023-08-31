# import json
# import numpy as np
# import cv2
# import os

# def read_mask_from_json(json_path, img_shape):
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # Get the original dimensions from JSON data if available, else assume they're the same as img_shape
#     original_shape = (data.get('imageHeight', img_shape[0]), data.get('imageWidth', img_shape[1]))
    
#     # Scale the points
#     points = np.array(data['shapes'][0]['points'])
#     scaling_factor_x = img_shape[1] / original_shape[1]
#     scaling_factor_y = img_shape[0] / original_shape[0]
#     points[:, 0] *= scaling_factor_x
#     points[:, 1] *= scaling_factor_y
    
#     # Convert to integers
#     points = points.astype(np.int32)
    
#     mask = np.zeros(img_shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [points], 255)
#     return mask

# folder_path = "data"
# image_paths = ["left.jpg", "right.jpg", "front.jpg"]
# json_paths = ["left_mask.json", "right_mask.json", "front_mask.json"]

# for image_name, json_name in zip(image_paths, json_paths):
#     image_path = os.path.join(folder_path, image_name)
#     json_path = os.path.join(folder_path, json_name)
    
#     image = cv2.imread(image_path)
#     resized_image = cv2.resize(image, (640, 480))
    
#     mask = read_mask_from_json(json_path, resized_image.shape)
#     resized_mask = cv2.resize(mask, (640, 480))

#     cv2.imshow(f"Image {image_name}", resized_image)
#     cv2.imshow(f"Mask {json_name}", resized_mask)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()


import json
import numpy as np
import cv2
import os

def read_mask_from_json(json_path, img_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get the original dimensions from JSON data if available, else assume they're the same as img_shape
    original_shape = (data.get('imageHeight', img_shape[0]), data.get('imageWidth', img_shape[1]))
    
    # Scale the points
    points = np.array(data['shapes'][0]['points'])
    scaling_factor_x = img_shape[1] / original_shape[1]
    scaling_factor_y = img_shape[0] / original_shape[0]
    points[:, 0] *= scaling_factor_x
    points[:, 1] *= scaling_factor_y
    
    # Convert to integers
    points = points.astype(np.int32)
    
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask

folder_path = "data"
image_paths = ["left.jpg", "right.jpg", "front.jpg"]
json_paths = ["left_mask.json", "right_mask.json", "front_mask.json"]

for image_name, json_name in zip(image_paths, json_paths):
    image_path = os.path.join(folder_path, image_name)
    json_path = os.path.join(folder_path, json_name)
    
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (640, 480))
    
    mask = read_mask_from_json(json_path, resized_image.shape)
    resized_mask = cv2.resize(mask, (640, 480))

    # Save the mask as a .png file
    mask_save_path = os.path.join(folder_path, f"{json_name.split('.')[0]}_mask.png")
    cv2.imwrite(mask_save_path, resized_mask)

    cv2.imshow(f"Image {image_name}", resized_image)
    cv2.imshow(f"Mask {json_name}", resized_mask)
    cv2.waitKey(0)

cv2.destroyAllWindows()

