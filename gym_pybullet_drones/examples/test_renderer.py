import open3d as o3d
import numpy as np

def get_view_matrix(eye, center, up):
    """
    Computes a camera extrinsic (4x4) matrix given the camera position, a target point, and the up vector.
    """
    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)
    
    # Compute the new camera coordinate system:
    z = (eye - center)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    
    extrinsic = np.eye(4)
    extrinsic[0, :3] = x
    extrinsic[1, :3] = y
    extrinsic[2, :3] = z
    extrinsic[:3, 3] = eye
    return extrinsic

# Create a cylinder.
cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=30, split=4)
cylinder.compute_vertex_normals()

# Create a Visualizer.
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(cylinder)

# Retrieve the view control.
view_ctl = vis.get_view_control()

# Define the desired camera parameters.
eye = [0, -3, 0]     # Camera positioned 3 meters away along the negative Y axis.
center = [0, 0, 0]   # Looking at the origin (center of the cylinder).
up = [0, 0, 1]       # Z-up coordinate system.

# Compute the extrinsic view matrix.
extrinsic = get_view_matrix(eye, center, up)

# Get current camera parameters, update its extrinsic matrix, then apply it.
cam_params = view_ctl.convert_to_pinhole_camera_parameters()
cam_params.extrinsic = extrinsic
view_ctl.convert_from_pinhole_camera_parameters(cam_params)

# Run the viewer.
vis.run()
vis.destroy_window()