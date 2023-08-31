# import open3d as o3d
# import numpy as np
# import cv2

# def read_and_convert_image(image_path):
#     cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if cv_image is None:
#         print(f"Failed to load {image_path}")
#         return None
#     cv_image = cv2.convertScaleAbs(cv_image, alpha=(65535.0/255.0))  # Convert to 16-bit
#     return o3d.geometry.Image(cv_image)

# # Load the depth maps
# depth_front = read_and_convert_image("data/depth_front.png")
# depth_left = read_and_convert_image("data/depth_left.png")
# depth_right = read_and_convert_image("data/depth_right.png")

# # Check if any image failed to load
# if depth_front is None or depth_left is None or depth_right is None:
#     exit()

# # Create point clouds
# pcd_front = o3d.geometry.PointCloud.create_from_depth_image(depth_front, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# pcd_left = o3d.geometry.PointCloud.create_from_depth_image(depth_left, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# pcd_right = o3d.geometry.PointCloud.create_from_depth_image(depth_right, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# # Apply ICP to align the point clouds
# threshold = 1.0
# trans_init = np.asarray([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]) 

# # Align left view with the front view
# reg_p2p = o3d.pipelines.registration.registration_icp(pcd_left, pcd_front, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
# pcd_left.transform(reg_p2p.transformation)

# # Align right view with the front view
# reg_p2p = o3d.pipelines.registration.registration_icp(pcd_right, pcd_front, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
# pcd_right.transform(reg_p2p.transformation)

# # Combine point clouds
# combined_pcd = pcd_front + pcd_left + pcd_right

# # Create a mesh
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(combined_pcd, 0.02)
# mesh.compute_vertex_normals()

# # Save as a 3D object
# o3d.io.write_triangle_mesh("final_mesh.ply", mesh)


import open3d as o3d
import numpy as np
import cv2

# Load the depth maps using OpenCV
depth_front_cv = cv2.imread("data/depth_front.png", cv2.IMREAD_UNCHANGED)
depth_left_cv = cv2.imread("data/depth_left.png", cv2.IMREAD_UNCHANGED)
depth_right_cv = cv2.imread("data/depth_right.png", cv2.IMREAD_UNCHANGED)

# Create Open3D Image objects
depth_front = o3d.geometry.Image(depth_front_cv)
depth_left = o3d.geometry.Image(depth_left_cv)
depth_right = o3d.geometry.Image(depth_right_cv)

# Create point clouds
pcd_front = o3d.geometry.PointCloud.create_from_depth_image(depth_front, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd_left = o3d.geometry.PointCloud.create_from_depth_image(depth_left, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd_right = o3d.geometry.PointCloud.create_from_depth_image(depth_right, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Apply ICP to align the point clouds
threshold = 1.0
trans_init = np.asarray([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]) 

# Align left view with the front view
reg_p2p = o3d.pipelines.registration.registration_icp(pcd_left, pcd_front, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_left.transform(reg_p2p.transformation)

# Align right view with the front view
reg_p2p = o3d.pipelines.registration.registration_icp(pcd_right, pcd_front, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_right.transform(reg_p2p.transformation)

# Combine point clouds
combined_pcd = pcd_front + pcd_left + pcd_right

# Create a mesh
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(combined_pcd, 0.02)
mesh.compute_vertex_normals()

# Save as a 3D object
o3d.io.write_triangle_mesh("final_mesh.ply", mesh)

