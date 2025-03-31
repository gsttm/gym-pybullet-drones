"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import torch

import pyvista as pv
from torchvision.models.optical_flow import raft_small 

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")

DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


class OpticalFlowTracker:
    def __init__(self, dt, model_size='small'):
        """
        Initializes the RAFT-based optical flow tracker.
        Note: parameters max_corners, quality_level, etc. are kept for API compatibility.
        """
        self.dt = dt
        # RAFT model initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_size == 'small':
            self.model = raft_small(pretrained=True).to(self.device).eval()
        else:
            raise NotImplementedError("Only 'small' model is implemented in this example.")
        self.prev_rgb = None
        # History of average flows for sensor fusion (store last 5 values)
        self.flow_history = []

    def update(self, frame_rgb, draw_on=None, attitude=None, altitude=None,lin_accel=None):
        """
        Parameters:
            frame_rgb: Current image (H x W X 3) as a uint8 numpy array.
            draw_on: (Optional) Color image (BGR) on which to draw optical flow arrows and key points.
            attitude: (Optional) Drone's attitude as a tuple (roll, pitch, yaw) in radians.
            altitude: (Optional) Drone's altitude in meters.
            lin_accel: (Optional) Drone's current acceleration (ax, ay, az) in the same units as used in fused state.
        Returns:
            fused_vel: numpy array representing fused velocity (in pixels per second),
                       computed from optical flow and an acceleration correction.
        """
        if self.prev_rgb is None:
            self.prev_rgb = frame_rgb.copy()
            return np.array([0.0, 0.0])

        # Convert grayscale images to 3-channel RGB since RAFT expects 3-channel input.
        prev_rgb = self.prev_rgb
        curr_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        # Convert images to torch tensors with shape (1, 3, H, W) and normalize to [0,1]
        prev_tensor = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        curr_tensor = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        prev_tensor = prev_tensor.to(self.device)
        curr_tensor = curr_tensor.to(self.device)

        with torch.no_grad():
            # Call the RAFT model without additional keyword arguments.
            flow_up = self.model(prev_tensor, curr_tensor)
            # If the output tensor is sparse, convert it to dense.
            flow_tensor = flow_up[0][0]
            if flow_tensor.is_sparse:
                flow_tensor = flow_tensor.to_dense()
            # Reshape flow to (H, W, 2)
            flow = flow_tensor.permute(1, 2, 0).cpu().numpy()
            # Average flow over the image yields an estimated average displacement (in pixels).
            average_flow = np.mean(flow, axis=(0, 1))
            # Compute instantaneous optical flow velocity.
            optical_flow_vel = average_flow / self.dt
            optical_flow_vel[0] = -optical_flow_vel[0]  # Invert x-component for correct direction.
            # Append current optical flow velocity to history and keep the latest 5 values.
            self.flow_history.append(optical_flow_vel)
            if len(self.flow_history) > 10:
                self.flow_history = self.flow_history[-10:]
            avg_flow_history = np.mean(np.array(self.flow_history), axis=0)

            # Compensation using attitude if provided.
            if attitude is not None:
                # Unpack roll, pitch, and yaw.
                roll, pitch, yaw = attitude
                # Create rotation matrices.
                R_x = np.array([[1, 0, 0],
                                [0, math.cos(roll), -math.sin(roll)],
                                [0, math.sin(roll), math.cos(roll)]])
                R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                                [0, 1, 0],
                                [-math.sin(pitch), 0, math.cos(pitch)]])
                R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                [math.sin(yaw),  math.cos(yaw), 0],
                                [0, 0, 1]])
                # Compose the full rotation matrix (order may be adjusted based on your convention).
                R_full = R_z.dot(R_y).dot(R_x)
                # Extend the 2D flow vector to 3D.
                of_vec3 = np.array([avg_flow_history[0], avg_flow_history[1], 0])
                comp_flow_vec3 = R_full.dot(of_vec3)
                # Use the x and y components.
                compensated_flow = comp_flow_vec3[:2]
            else:
                compensated_flow = avg_flow_history
            
            fov = np.deg2rad(70)
            image_height = 224
            focal_length_pixels = (image_height / 2) / np.tan(fov / 2)
            meters_per_pixel = altitude / focal_length_pixels
            est_vel = compensated_flow * meters_per_pixel

            fused_vel = est_vel
            if draw_on is not None:
                h, w = draw_on.shape[:2]
                grid_step = 16
                small_factor = 2
                # Draw green arrows for local flow vectors.
                for y in range(0, h, grid_step):
                    for x in range(0, w, grid_step):
                        dx, dy = flow[y, x]
                        start_point = (x, y)
                        end_point = (int(x + -dx * small_factor), int(y + -dy * small_factor))
                        cv2.arrowedLine(draw_on, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)
                # Draw a red arrow showing the fused velocity at the image center.
                center = (w // 2, h // 2)
                scale_factor = 20  # adjust for visualization if needed
                fused_end = (int(center[0] + fused_vel[0] * scale_factor),
                             int(center[1] + -fused_vel[1] * scale_factor))
                cv2.arrowedLine(draw_on, center, fused_end, (0, 0, 255), 2, tipLength=0.3)

        self.prev_rgb = frame_rgb.copy()
        return fused_vel

u_history = []
plotter = None

def lines_from_points(points):
    """Given an array of points, make a line set"""
    #apply small z offset to points
    points = points - np.array([0, 0, 0.1])
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

def initViz():
    global plotter
    plotter = pv.Plotter(off_screen=True, window_size=(224, 224))
    plotter.set_background("black")
def overlayWaypoint3DCylinder(img, target_pos, drone_pos, attitude, fov_diag_deg, traj):
    """
    Overlays a red 3D cylinder as a waypoint marker that is clipped by a ground plane,
    and a blue 'road' derived from the 3 dimensional trajectory.
    The cylinder is 2 m tall and 0.3 m in radius; it is clipped so that nothing appears below z=0.
    The road is rendered as a tube along the provided 3D trajectory (from current waypoint onward),
    with a color gradient based on the Euclidean distance from the current drone position.
    The rendered scene is then blended with the input image.

    Parameters:
        img (np.ndarray): Input image (H x W x 3, uint8).
        target_pos (list/tuple): [x, y, z] where the cylinder’s bottom should be placed.
                                  Note: the z-value is ignored for the cylinder placement (ground is at z=0).
        drone_pos (list/tuple): [x, y, z] representing the camera (drone) position.
        attitude (tuple): (roll, pitch, yaw) in radians.
        fov_diag_deg (float): Field-of-view in degrees for the camera.
        traj (array-like): Nx3 array of waypoints (3D trajectory) from the current point onward.
        
    Returns:
        np.ndarray: Blended image (uint8).
    """
    global plotter
    # Define ground plane height.
    ground_z = 0.0

    # Cylinder dimensions.
    cylinder_height = 200.0   # meters
    cylinder_radius = 1.0   # meters

    # Get image dimensions.
    h, w, _ = img.shape

    # Create the cylinder so that its bottom touches the ground.
    cylinder_bottom = np.array([target_pos[0], target_pos[1], ground_z])
    cylinder_center = cylinder_bottom + np.array([0, 0, cylinder_height / 2])
    cylinder = pv.Cylinder(center=cylinder_center, direction=(0, 0, 1),
                           radius=cylinder_radius, height=cylinder_height, resolution=32)
    # Clip the cylinder with the ground plane. Only the portion with z>=ground_z is kept.
    clipped_cylinder = cylinder.clip(normal=(0, 0, 1),
                                     origin=(cylinder_center[0], cylinder_center[1], ground_z),
                                     invert=False)
    cyl_actor = plotter.add_mesh(clipped_cylinder, color="red")

    # Create the road based on the provided 3D trajectory.
    traj = np.array(traj)  # assume traj is Nx3 (3D)
    line = lines_from_points(traj)

    # Convert the polyline to a tube to give it a visible width.
    line['distance'] = range(traj.shape[0])
    road_tube = line.ribbon(width=0.02, normal=[0, 0, 1])
   

    road_actor = plotter.add_mesh(road_tube, cmap="Blues", show_scalar_bar=False)
    # Compute camera pose from attitude.
    roll, pitch, yaw = attitude
    forward_vec = np.array([1, 0, 0])
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw),  math.cos(yaw), 0],
                    [0, 0, 1]])
    R = R_z.dot(R_y).dot(R_x)
    cam_forward = R.dot(forward_vec)
    camera_position = np.array(drone_pos)
    camera_target = camera_position + cam_forward

    # Set the camera position with global Z as the up direction.
    plotter.camera_position = [camera_position.tolist(), camera_target.tolist(), [0, 0, 1]]
    plotter.camera.SetViewAngle(fov_diag_deg)

    # Render the scene.
    plotter.render()
    rendered_img = plotter.screenshot(transparent_background=True)
    
    # Ensure the rendered image has an alpha channel.
    if rendered_img.shape[2] == 3:
        alpha_channel = 255 * np.ones(
            (rendered_img.shape[0], rendered_img.shape[1], 1), dtype=rendered_img.dtype)
        rendered_img = np.concatenate((rendered_img, alpha_channel), axis=2)

    # Resize the rendered overlay to match the input image dimensions.
    rendered_img = cv2.resize(rendered_img, (w, h))

    # Blend the rendered overlay with the original image.
    overlay_rgb = rendered_img[..., :3].astype(np.float32)
    alpha = rendered_img[..., 3:4].astype(np.float32) / 255.0
    base_rgb = img.astype(np.float32)
    blended = base_rgb * (1 - 0.5 * alpha) + overlay_rgb * (0.5 * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Clean up the added actors.
    plotter.remove_actor(cyl_actor)
    plotter.remove_actor(road_actor)
    return blended
def initializeTrajectoryTowardsObject(num_drones, control_freq_hz):
    # Randomly choose a ground position for the target cube.
    # For example, choose x and y uniformly between -50 and 50; z is 0 for ground.
    cube_x = 0
    cube_y = 0
    cube_z = 0.5
    cube_pos = [cube_x, cube_y, cube_z]
    
    # Place the drone 100 meters away horizontally and 100 meters above the cube.
    # For simplicity, we place the drone along the x-direction from the cube.
    # You can change the offset direction as desired.
    offset_horizontal = 20.0  # horizontal distance in meters
    offset_vertical = 3.0      # vertical distance in meters
    # Drone initial horizontal position: cube_x + offset
    init_x = cube_x + offset_horizontal
    # For y, we can leave it the same (or add an offset if desired).
    init_y = cube_y
    init_z = cube_z + offset_vertical  # 100 m above ground
    
    # Compute yaw so that the drone's forward axis points toward the cube.
    # The yaw is the angle from the drone to the cube in the horizontal plane.
    yaw = math.atan2(cube_y - init_y, cube_x - init_x)
    
    # Set the initial positions and orientations for each drone.
    # If you have only one drone, simply replicate the initial condition.
    INIT_XYZS = np.array([[init_x, init_y, init_z] for _ in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, yaw] for _ in range(num_drones)])
    
    # Generate a straight-line trajectory from the drone initial position to the cube.
    # Let's assume we want a total of 100 waypoints.
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD  # for example, a 10-second approach
    TARGET_POS = np.zeros((NUM_WP, 3))
    # Linearly interpolate between the initial position and the cube position.
    for i in range(NUM_WP):
        alpha = i / (NUM_WP - 1)
        TARGET_POS[i, :] = (1 - alpha) * np.array([init_x, init_y, init_z]) + alpha * np.array([cube_x, cube_y, cube_z])
    
    # Initialize the waypoint counter for each drone.
    wp_counters = np.array([0 for _ in range(num_drones)])
    
    return INIT_XYZS, INIT_RPYS, TARGET_POS, NUM_WP, wp_counters, cube_pos
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
   
    # Get initial trajectory and cube position.
    INIT_XYZS, INIT_RPYS, TARGET_POS, NUM_WP, wp_counters, cube_pos = initializeTrajectoryTowardsObject(num_drones, control_freq_hz)

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=False,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=False,
                     vision_attributes=True
                     )
    
    ### Init visualization
    initViz()

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    # Load the cube URDF once and store its unique ID.
    cube_id = p.loadURDF("cube_no_rotation.urdf",
                         cube_pos,
                         p.getQuaternionFromEuler([0, 0, 0]),
                         physicsClientId=PYB_CLIENT)

    ## Initialize the optical flow tracker
    dt = env.CTRL_TIMESTEP
    of_tracker = OpticalFlowTracker(dt)

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    PERIOD = 10  # seconds for a full trajectory generation
    NUM_WP = control_freq_hz * PERIOD  # number of waypoints in trajectory

    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        #### Update the cube position: move left/right by ±1 meter.
        # Randomly choose a shift in x-direction (±1 meter)
        if i < 50:
            delta_x = 0.0
            cube_pos[1] += delta_x
            # Reset the cube's position in the simulation.
            p.resetBasePositionAndOrientation(cube_id,
                                            cube_pos,
                                            p.getQuaternionFromEuler([0, 0, 0]),
                                            physicsClientId=PYB_CLIENT)

        #### Regenerate the trajectory from the current drone position to the new cube position.
        # First, get the current drone position.
        current_drone = obs[0][0:3] if i > 0 else INIT_XYZS[0]
        # Use current_drone[2] as the starting altitude and cube_pos[2] as the landing altitude.
        start_z = current_drone[2]
        target_z = cube_pos[2]
        new_TARGET_POS = np.zeros((NUM_WP, 3))
        for j in range(NUM_WP):
            # Linear interpolation for xy components.
            alpha = j / (NUM_WP - 1)
            new_TARGET_POS[j, :2] = (1 - alpha) * np.array(current_drone[:2]) + alpha * np.array(cube_pos[:2])
            # Compute 2D distance from this waypoint to the target.
            d = np.linalg.norm(np.array(cube_pos[:2]) - new_TARGET_POS[j, :2])
            if d > 10:
                # Stay at the high (start) altitude.
                new_TARGET_POS[j, 2] = start_z
            else:
                # When within 10 m, interpolate altitude with smoothstep function.
                # t = 0 when d=10, and t = 1 when d=0.
                t = (10 - d) / 10
                # Smoothstep: f(t) = start_z + (target_z - start_z)*(3*t^2 - 2*t^3)
                new_TARGET_POS[j, 2] = start_z + (target_z - start_z) * (3 * t**2 - 2 * t**3)
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        altitude = obs[0][2]
        attitude = obs[0][7:10] 
        lin_accel = obs[0][20:23]
        rgb, rgb_down, rgb_fixed = env._getDroneImages(0, False)
        rgb_disp = rgb[:, :, :3].astype(np.uint8)
        rgb_down_disp = rgb_down[:, :, :3].astype(np.uint8)
        rgb_fixed_disp = rgb_fixed[:, :, :3].astype(np.uint8)
        
        rgb_down_disp_copy = rgb_down_disp.copy()
        
        est_vel = of_tracker.update(rgb_down_disp_copy, draw_on=rgb_down_disp,
                                    attitude=attitude, altitude=altitude, lin_accel=lin_accel)

        actual_vel = obs[0][10:12]
        diff = (est_vel - actual_vel)**2
        
        # Use the regenerated trajectory for overlay drawing.
        rgb_disp = overlayWaypoint3DCylinder(rgb_disp,
                                             target_pos=cube_pos,
                                             fov_diag_deg=70,
                                             drone_pos=obs[0][0:3],
                                             attitude=attitude,
                                             traj=new_TARGET_POS)
        # Combine the images horizontally.
        combined_img = np.hstack((rgb_disp, rgb_down_disp))
        cv2.imshow("Combined Drone Views", combined_img)
        cv2.waitKey(1)

        #### Compute control for the current waypoint #############
        # Update target orientation so that the drone points directly toward the cube.
        # Calculate yaw = math.atan2(delta_y, delta_x) from current drone position to cube.
        current_drone = obs[0][0:3]
        target_yaw = math.atan2(cube_pos[1] - current_drone[1], cube_pos[0] - current_drone[0])
        new_target_rpy = [0, 0, target_yaw]  # roll and pitch remain 0.
    
        #### Compute control for the current waypoint #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                  state=obs[j],
                                                                  target_pos=new_TARGET_POS[wp_counters[j], :],
                                                                  target_rpy=new_target_rpy)

        #### Go to the next waypoint and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

    #### Close the environment #################################
    env.close()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
