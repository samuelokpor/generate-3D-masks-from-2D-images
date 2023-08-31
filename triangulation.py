import cv2
import open3d as o3d
import numpy as np
import os

def create_point_cloud(image_path, mask_path, intrinsic):
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Image or mask not found!")
        return None

    cv_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming mask is grayscale

    if cv_image is None or mask is None:
        print(f"Failed to load image or mask")
        return None

    mask = mask / 255  # Convert to 0 and 1
    cv_image = (cv_image * mask).astype(np.uint16)  # Apply mask

    o3d_image = o3d.geometry.Image(cv_image)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_image, intrinsic)
    return pcd

# Camera intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 200, 200, 219.5, 139.5)

# Create point clouds from depth images
pcd_front = create_point_cloud("data/depth_front.png", "data/front_mask.png", intrinsic)
pcd_left = create_point_cloud("data/depth_left.png", "data/left_mask.png", intrinsic)
pcd_right = create_point_cloud("data/depth_right.png", "data/right_mask.png", intrinsic)

if pcd_front is None or pcd_left is None or pcd_right is None:
    exit()

# Registration (Alignment)
# You may need to adjust the initial transformation guess
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

# Align left to front
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_left, pcd_front, 0.03, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_left.transform(reg_p2p.transformation)

# Align right to front
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_right, pcd_front, 0.03, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_right.transform(reg_p2p.transformation)

# Merge point clouds
merged_pcd = pcd_front + pcd_left + pcd_right

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pcd])
