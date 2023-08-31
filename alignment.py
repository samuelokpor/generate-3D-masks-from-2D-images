import json
import cv2
import numpy as np

def read_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['shapes'][0]['points'], dtype=np.float32)

def align_images_using_keypoints(img1, img2, points1, points2):
    if len(points1) != len(points2):
        print("Mismatched keypoints, returning original image")
        return img1
    
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, h, (width, height))
    return aligned_img1

# Initialize to None
aligned_mesh_left = aligned_mesh_right = None

# Read mesh images
mesh_front = cv2.imread("data/mesh_front.png", cv2.IMREAD_GRAYSCALE)
mesh_left = cv2.imread("data/mesh_left.png", cv2.IMREAD_GRAYSCALE)
mesh_right = cv2.imread("data/mesh_right.png", cv2.IMREAD_GRAYSCALE)

# Read keypoints
keypoints_front = read_keypoints("data/mesh_front_keypoints.json")
keypoints_left = read_keypoints("data/mesh_left_keypoints.json")
keypoints_right = read_keypoints("data/mesh_right_keypoints.json")

# Debug info
print(f"Number of keypoints Front: {len(keypoints_front)}")
print(f"Number of keypoints Left: {len(keypoints_left)}")
print(f"Number of keypoints Right: {len(keypoints_right)}")

# Align and save images, if possible
if keypoints_front is not None and keypoints_left is not None and keypoints_right is not None:
    if len(keypoints_left) == len(keypoints_front) and len(keypoints_right) == len(keypoints_front):
        aligned_mesh_left = align_images_using_keypoints(mesh_left, mesh_front, keypoints_left, keypoints_front)
        aligned_mesh_right = align_images_using_keypoints(mesh_right, mesh_front, keypoints_right, keypoints_front)
    else:
        print("Mismatched keypoints")

if aligned_mesh_left is not None:
    cv2.imwrite("data/aligned_mesh_left.png", aligned_mesh_left)

if aligned_mesh_right is not None:
    cv2.imwrite("data/aligned_mesh_right.png", aligned_mesh_right)
