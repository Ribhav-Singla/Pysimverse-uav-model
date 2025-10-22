# UAV Navigation Simulation
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
import torch
import os
import sys  # For error handling
from ppo_agent import PPOAgent
from uav_env import UAVEnv, EnvironmentGenerator

# Define the path to your XML model
MODEL_PATH = "environment.xml"

# Configuration parameters - MUST match training environment
CONFIG = {
    'start_pos': np.array([-3.0, -3.0, 1.0]),  # Default start position (will be updated dynamically)
    'goal_pos': np.array([3.0, 3.0, 1.0]),     # Default goal position (will be updated dynamically)
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,
    'static_obstacles': 4,
    'min_obstacle_size': 0.05,
    'max_obstacle_size': 0.12,
    'collision_distance': 0.1,
    'control_dt': 0.05,
    'boundary_penalty': -100,
    'lidar_range': 2.9,
    'lidar_num_rays': 16,
    'step_reward': -0.1,
    
    # Render-specific parameters (do not affect agent logic)
    'kp_pos': 1.5,
    'path_trail_length': 500,
}

class EnvironmentGenerator:
    @staticmethod
    def get_random_corner_position():
        """Get a random start position from one of the four corners"""
        half_world = CONFIG['world_size'] / 2 - 1.0  # 1m margin from boundary
        corners = [
            np.array([-half_world, -half_world, CONFIG['uav_flight_height']]),  # Bottom-left
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),   # Bottom-right
            np.array([-half_world, half_world, CONFIG['uav_flight_height']]),   # Top-left
            np.array([half_world, half_world, CONFIG['uav_flight_height']])     # Top-right
        ]
        return random.choice(corners)
    
    @staticmethod
    def get_random_goal_position():
        """Get a random goal position anywhere in the map (not just corners)"""
        half_world = CONFIG['world_size'] / 2 - 1.0  # 1m margin from boundary
        x = random.uniform(-half_world, half_world)
        y = random.uniform(-half_world, half_world)
        return np.array([x, y, CONFIG['uav_flight_height']])
    
    @staticmethod
    def get_random_goal_position_legacy():
        """Legacy method: Select a random goal position from the three available corners (excluding start position)"""
        # Define the four corners of the world
        half_world = CONFIG['world_size'] / 2
        corners = [
            np.array([half_world, half_world, CONFIG['uav_flight_height']]),    # Top-right
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),   # Bottom-right
            np.array([-half_world, half_world, CONFIG['uav_flight_height']])    # Top-left
        ]
        # Start position is bottom-left: [-half_world, -half_world, height]
        # So we exclude it and randomly select from the other three corners
        return random.choice(corners)
    
    @staticmethod
    def generate_obstacles():
        """Generate both static and dynamic obstacles with random non-overlapping positions"""
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        # Generate a grid of possible positions for uniform distribution
        grid_size = int(math.sqrt(CONFIG['static_obstacles'])) + 1
        cell_size = world_size / grid_size
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = -half_world + (i + 0.5) * cell_size
                y = -half_world + (j + 0.5) * cell_size
                positions.append((x, y))
        
        # Shuffle positions for random assignment
        random.shuffle(positions)
        
        # Generate static obstacles with random placement
        for i in range(CONFIG['static_obstacles']):
            x, y = positions[i]
            
            # Random size and shape
            size_x = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
            size_y = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
            height = CONFIG['obstacle_height']
            
            obstacle_type = random.choice(['box', 'cylinder'])
            color = [random.uniform(0.1, 0.9) for _ in range(3)] + [1.0]
            
            obstacles.append({
                'type': 'static',
                'shape': obstacle_type,
                'pos': [x, y, height/2],
                'size': [size_x, size_y, height/2] if obstacle_type == 'box' else [min(size_x, size_y), height/2],
                'color': color,
                'id': f'static_obs_{i}'
            })
        
        return obstacles
    
    @staticmethod
    def create_xml_with_obstacles(obstacles):
        """Create XML model with dynamically generated obstacles"""
        xml_template = f'''<mujoco model="complex_uav_env">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="2560" offheight="1440"/>
  </visual>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="{CONFIG['world_size']/2} {CONFIG['world_size']/2} 0.1" material="grid"/>
    <light name="light1" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
    
    <!-- Start position marker (green) -->
    <geom name="start_marker" type="cylinder" size="0.15 0.05" pos="{CONFIG['start_pos'][0]} {CONFIG['start_pos'][1]} 0.05" rgba="0 1 0 0.8"/>
    <geom name="start_pole" type="box" size="0.03 0.03 0.4" pos="{CONFIG['start_pos'][0]} {CONFIG['start_pos'][1]} 0.4" rgba="0 1 0 1"/>
    
    <!-- Goal position marker (blue) -->
    <geom name="goal_marker" type="cylinder" size="0.15 0.05" pos="{CONFIG['goal_pos'][0]} {CONFIG['goal_pos'][1]} 0.05" rgba="0 0 1 0.8"/>
    <geom name="goal_pole" type="box" size="0.03 0.03 0.4" pos="{CONFIG['goal_pos'][0]} {CONFIG['goal_pos'][1]} 0.4" rgba="0 0 1 1"/>
    
    <!-- UAV starting position -->
    <body name="chassis" pos="{CONFIG['start_pos'][0]} {CONFIG['start_pos'][1]} {CONFIG['start_pos'][2]}">
      <joint type="free" name="root"/>
      <geom name="uav_body" type="box" size="0.12 0.12 0.03" rgba="1.0 0.0 0.0 1.0" mass="0.8"/>
      
      <!-- Propeller arms and motors -->
      <geom type="box" size="0.06 0.008 0.004" pos="0 0 0.008" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" size="0.008 0.06 0.004" pos="0 0 0.008" rgba="0.3 0.3 0.3 1"/>
      
      <!-- Motor visual geometry -->
      <geom type="cylinder" size="0.012 0.015" pos="0.06 0.06 0.012" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.012 0.015" pos="-0.06 0.06 0.012" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.012 0.015" pos="0.06 -0.06 0.012" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.012 0.015" pos="-0.06 -0.06 0.012" rgba="0.2 0.2 0.2 1"/>
      
      <!-- Propellers -->
      <geom name="prop1" type="cylinder" size="0.04 0.008" pos="0.06 0.06 0.04" rgba="0.0 0.0 0.0 1.0"/>
      <geom name="prop2" type="cylinder" size="0.04 0.008" pos="-0.06 0.06 0.04" rgba="0.0 0.0 0.0 1.0"/>
      <geom name="prop3" type="cylinder" size="0.04 0.008" pos="0.06 -0.06 0.04" rgba="0.0 0.0 0.0 1.0"/>
      <geom name="prop4" type="cylinder" size="0.04 0.008" pos="-0.06 -0.06 0.04" rgba="0.0 0.0 0.0 1.0"/>
      
      <!-- Sites for motor force application -->
      <site name="motor1" pos="0.06 0.06 0" size="0.01"/>
      <site name="motor2" pos="-0.06 0.06 0" size="0.01"/>
      <site name="motor3" pos="0.06 -0.06 0" size="0.01"/>
      <site name="motor4" pos="-0.06 -0.06 0" size="0.01"/>
    </body>'''
        
        # Add obstacles - use the passed obstacles parameter
        for obs in obstacles:
            if obs['shape'] == 'box':
                xml_template += f'''
    <geom name="{obs['id']}" type="box" size="{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
            elif obs['shape'] == 'cylinder':
                xml_template += f'''
    <geom name="{obs['id']}" type="cylinder" size="{obs['size'][0]} {obs['size'][1]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
            elif obs['shape'] == 'sphere':
                xml_template += f'''
    <geom name="{obs['id']}" type="sphere" size="{obs['size'][0]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
        
        # Add dynamic trail geometries that will follow the UAV's actual path
        for i in range(CONFIG['path_trail_length']):
            xml_template += f'''
    <geom name="trail_{i}" type="sphere" size="0.02" pos="0 0 -10" rgba="0 1 0 0.6" contype="0" conaffinity="0"/>'''
        
        xml_template += '''
  </worldbody>

  <actuator>
    <motor name="m1" gear="0 0 1 0 0 0" site="motor1"/>
    <motor name="m2" gear="0 0 1 0 0 0" site="motor2"/>
    <motor name="m3" gear="0 0 1 0 0 0" site="motor3"/>
    <motor name="m4" gear="0 0 1 0 0 0" site="motor4"/>
  </actuator>
</mujoco>'''
        
        # Write to file
        with open(MODEL_PATH, 'w') as f:
            f.write(xml_template)
        
        return obstacles
    
    @staticmethod
    def check_collision(uav_pos, obstacles):
        """Check if UAV collides with any obstacle with precise 0.2m threshold"""
        collision_dist = CONFIG['collision_distance']  # This should be 0.2
        
        for obs in obstacles:
            obs_pos = np.array(obs['pos'])
            
            # For box obstacles, check distance to each face
            if obs['shape'] == 'box':
                # Calculate minimum distance to any face of the box
                dx = max(obs_pos[0] - obs['size'][0] - uav_pos[0], 
                         uav_pos[0] - (obs_pos[0] + obs['size'][0]), 0)
                dy = max(obs_pos[1] - obs['size'][1] - uav_pos[1], 
                         uav_pos[1] - (obs_pos[1] + obs['size'][1]), 0)
                dz = max(obs_pos[2] - obs['size'][2] - uav_pos[2], 
                         uav_pos[2] - (obs_pos[2] + obs['size'][2]), 0)
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
            # For cylinder obstacles, check horizontal distance and vertical overlap
            elif obs['shape'] == 'cylinder':
                # Horizontal distance
                horizontal_dist = math.sqrt((uav_pos[0]-obs_pos[0])**2 + (uav_pos[1]-obs_pos[1])**2)
                # Vertical distance
                vertical_dist = max(0, abs(uav_pos[2]-obs_pos[2]) - obs['size'][1])
                
                if horizontal_dist <= obs['size'][0] and vertical_dist == 0:
                    distance = 0  # Inside cylinder
                else:
                    distance = max(horizontal_dist - obs['size'][0], vertical_dist)
                    
            # For sphere obstacles, check center-to-center distance
            elif obs['shape'] == 'sphere':
                distance = math.sqrt((uav_pos[0]-obs_pos[0])**2 + 
                                    (uav_pos[1]-obs_pos[1])**2 + 
                                    (uav_pos[2]-obs_pos[2])**2) - obs['size'][0]
            
            # Check if distance is less than collision threshold
            if distance < collision_dist:
                return True, obs['id'], distance
        
        return False, None, float('inf')

def get_lidar_readings(pos, obstacles):
    """Generate LIDAR readings in 360 degrees around the UAV - MUST match training"""
    lidar_readings = []
    
    for i in range(CONFIG['lidar_num_rays']):
        angle = (2 * math.pi * i) / CONFIG['lidar_num_rays']
        ray_dir = np.array([math.cos(angle), math.sin(angle), 0])
        
        min_distance = CONFIG['lidar_range']
        
        boundary_dist = ray_boundary_intersection(pos, ray_dir)
        if boundary_dist < min_distance:
            min_distance = boundary_dist
        
        for obs in obstacles:
            obs_dist = ray_obstacle_intersection(pos, ray_dir, obs)
            if obs_dist < min_distance:
                min_distance = obs_dist
        
        # Normalize LIDAR reading to [0, 1] range - MUST match training
        normalized_distance = min_distance / CONFIG['lidar_range']
        lidar_readings.append(normalized_distance)
    
    return np.array(lidar_readings)

def ray_boundary_intersection(pos, ray_dir):
    """Calculate intersection of ray with world boundaries - MUST match training"""
    half_world = CONFIG['world_size'] / 2
    min_dist = CONFIG['lidar_range']
    
    boundaries = [
        (half_world, np.array([1, 0, 0])),
        (-half_world, np.array([-1, 0, 0])),
        (half_world, np.array([0, 1, 0])),
        (-half_world, np.array([0, -1, 0]))
    ]
    
    for boundary_pos, boundary_normal in boundaries:
        denominator = np.dot(ray_dir, boundary_normal)
        if abs(denominator) > 1e-6:
            if boundary_normal[0] != 0:
                t = (boundary_pos - pos[0]) / ray_dir[0]
            else:
                t = (boundary_pos - pos[1]) / ray_dir[1]
            
            if t > 0:
                intersection = pos + t * ray_dir
                if (-half_world <= intersection[0] <= half_world and 
                    -half_world <= intersection[1] <= half_world):
                    min_dist = min(min_dist, t)
    
    return min_dist

def ray_obstacle_intersection(pos, ray_dir, obstacle):
    """Calculate intersection of ray with obstacle - MUST match training"""
    obs_pos = np.array(obstacle['pos'])
    min_dist = CONFIG['lidar_range']
    
    if obstacle['shape'] == 'box':
        to_obs = obs_pos - pos
        proj_length = np.dot(to_obs, ray_dir)
        
        if proj_length > 0:
            closest_point = pos + proj_length * ray_dir
            lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])
            
            if lateral_dist < max(obstacle['size'][0], obstacle['size'][1]):
                surface_dist = max(0, proj_length - max(obstacle['size'][0], obstacle['size'][1]))
                min_dist = min(min_dist, surface_dist)
    
    elif obstacle['shape'] == 'cylinder':
        to_obs = obs_pos - pos
        proj_length = np.dot(to_obs, ray_dir)
        
        if proj_length > 0:
            closest_point = pos + proj_length * ray_dir
            lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])
            
            if lateral_dist < obstacle['size'][0]:
                surface_dist = max(0, proj_length - obstacle['size'][0])
                min_dist = min(min_dist, surface_dist)
    
    return min_dist

import os

# ... (rest of the imports)

# ... (CONFIG and EnvironmentGenerator class)

# Initialize UAV environment first to use its exact methods
print("🏗️ Initializing UAV environment for rendering...")
env = UAVEnv()
print("🏗️ Generating obstacles using training environment methods...")
obstacles = env.obstacles  # Use the same obstacles as the training environment

# Helper function to ensure positions are safe from obstacles
def check_position_safety(position, obstacles, safety_radius=0.5):
    """Check if a position is safe from obstacle collisions"""
    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        
        # For box obstacles
        if obs['shape'] == 'box':
            # Calculate minimum distance to any face of the box
            dx = max(obs_pos[0] - obs['size'][0] - position[0], 
                     position[0] - (obs_pos[0] + obs['size'][0]), 0)
            dy = max(obs_pos[1] - obs['size'][1] - position[1], 
                     position[1] - (obs_pos[1] + obs['size'][1]), 0)
            dz = max(obs_pos[2] - obs['size'][2] - position[2], 
                     position[2] - (obs_pos[2] + obs['size'][2]), 0)
            
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
        # For cylinder obstacles
        elif obs['shape'] == 'cylinder':
            # Horizontal distance
            horizontal_dist = math.sqrt((position[0]-obs_pos[0])**2 + (position[1]-obs_pos[1])**2)
            # Vertical distance
            vertical_dist = max(0, abs(position[2]-obs_pos[2]) - obs['size'][1])
            
            if horizontal_dist <= obs['size'][0]:
                # Inside cylinder horizontally
                if vertical_dist == 0:
                    distance = 0  # Inside cylinder
                else:
                    distance = vertical_dist
            else:
                # Outside cylinder horizontally
                if vertical_dist == 0:
                    distance = horizontal_dist - obs['size'][0]
                else:
                    distance = math.sqrt((horizontal_dist-obs['size'][0])**2 + vertical_dist**2)
        
        # Check if distance is less than safety radius
        if distance < safety_radius:
            return False
            
    return True

# Set up safe start and goal positions
max_attempts = 50
safety_radius = 0.2  # Keep at least 0.2m from obstacles

# Use the EXACT same position generation as training environment
print("🏠 Using training environment's start and goal positions...")
print(f"✅ Start position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
print(f"✅ Goal position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
start_goal_distance = np.linalg.norm(CONFIG['goal_pos'][:2] - CONFIG['start_pos'][:2])
print(f"📏 Start-Goal distance: {start_goal_distance:.1f}m")

# Create environment XML using training environment's method
EnvironmentGenerator.create_xml_with_obstacles(obstacles)

# Load the model and create the simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"🚁 Complex UAV Navigation Environment Loaded!")
print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
print(f"🎯 Mission: Navigate from START (green) to GOAL (blue)")
print(f"� Start Position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
print(f"🏁 Goal Position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
print(f"�🚧 Static obstacles: {CONFIG['static_obstacles']}")
print(f"📏 Obstacle height: {CONFIG['obstacle_height']}m")
print(f"✈️ UAV flight height: {CONFIG['uav_flight_height']}m")
print(f"🛣️ Green path trail: {CONFIG['path_trail_length']} points")

# Initialize PPO agent with the EXACT same architecture as training
state_dim = env.observation_space.shape[0]  # Will be 36 with our enhanced features
action_dim = env.action_space.shape[0]

# Use EXACT parameters from training.py
action_std = 1.0            # Start with 100% exploration
lr_actor = 0.0001           # learning rate for actor
lr_critic = 0.0004          # learning rate for critic
gamma = 0.999               # increased for longer-term planning
K_epochs = 12               # update policy for K epochs
eps_clip = 0.1              # clip parameter for PPO

# Initialize PPO agent
ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

# Load the trained model
try:
    # Load the single weight file
    model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth" 
    
    if os.path.exists(model_path):
        ppo_agent.load(model_path)
        print(f"🤖 Trained PPO agent loaded successfully from {model_path}!")
    else:
        raise FileNotFoundError(f"Weight file not found: {model_path}")
        
except Exception as e:
    print(f"⚠️ Could not load trained agent: {str(e)}")
    print("⚠️ Using random actions (model will behave unpredictably).")
    print("💡 Make sure to run training.py first to generate the weight file.")

path_history = []
# ... (rest of the script)


def update_path_trail(model, data, current_pos):
    """Update the path trail visualization."""
    path_history.append(current_pos.copy())
    if len(path_history) > CONFIG['path_trail_length']:
        path_history.pop(0)
    
    trail_count = len(path_history)
    total_geoms = model.ngeom
    trail_start_idx = total_geoms - CONFIG['path_trail_length']
    
    for i in range(CONFIG['path_trail_length']):
        geom_idx = trail_start_idx + i
        if 0 <= geom_idx < total_geoms and i < trail_count:
            # Only modify trail geometries, not UAV geometries
            try:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx) or ""
            except:
                geom_name = ""
            if "trail_" in geom_name:
                model.geom_pos[geom_idx] = path_history[i]
                alpha = 0.3 + 0.5 * (i / max(1, trail_count))
                model.geom_rgba[geom_idx] = [0.0, 1.0, 0.0, alpha]
        elif 0 <= geom_idx < total_geoms:
            try:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx) or ""
            except:
                geom_name = ""
            if "trail_" in geom_name:
                model.geom_pos[geom_idx] = [0, 0, -10]
                model.geom_rgba[geom_idx] = [0, 0, 0, 0]

# Function to ensure UAV visibility

def ensure_uav_visibility(model):
    """Makes sure the UAV remains visible by forcing color settings"""
    # Find all UAV-related geometries and protect them from color changes
    for i in range(model.ngeom):
        try:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        except:
            geom_name = ""
        
        # Set UAV body (solid red)
        if "uav_body" in geom_name:
            model.geom_rgba[i] = [1.0, 0.0, 0.0, 1.0]  # Solid red
        
        # Set propellers (solid black)
        elif "prop" in geom_name:
            model.geom_rgba[i] = [0.0, 0.0, 0.0, 1.0]  # Solid black

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position
    data.qpos[:3] = CONFIG['start_pos']
    data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation (quaternion)
    
    print("Mission started! Watch the PPO agent control the UAV!")
    time.sleep(2)
    
    step_count = 0
    mission_complete = False
    collision_occurred = False
    
    # Goal tracking variable
    goal_reached = False
    
    try:
        while viewer.is_running() and not mission_complete and not collision_occurred:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Create observation EXACTLY as in training environment
            pos = current_pos
            vel = current_vel
            goal_dist = CONFIG['goal_pos'] - pos
            
            # Use the EXACT same LIDAR function as training
            lidar_readings = env._get_lidar_readings(pos)
            
            # === EXACT LIDAR FEATURE ENGINEERING from uav_env.py ===
            # 1. Minimum distance (closest obstacle)
            min_lidar = np.min(lidar_readings)
            
            # 2. Mean distance (overall clearance)
            mean_lidar = np.mean(lidar_readings)
            
            # 3. Direction to closest obstacle (unit vector)
            closest_idx = np.argmin(lidar_readings)
            closest_angle = (2 * np.pi * closest_idx) / CONFIG['lidar_num_rays']
            obstacle_direction = np.array([np.cos(closest_angle), np.sin(closest_angle)])
            
            # 4. Danger level (ratio of close obstacles)
            danger_threshold = 0.36  # 1.0m / 2.8m normalized
            num_close_obstacles = np.sum(lidar_readings < danger_threshold)
            danger_level = num_close_obstacles / CONFIG['lidar_num_rays']
            
            # 5. Directional clearance (front/right/back/left sectors)
            sector_size = len(lidar_readings) // 4
            front_clear = np.mean(lidar_readings[0:sector_size])
            right_clear = np.mean(lidar_readings[sector_size:2*sector_size])
            back_clear = np.mean(lidar_readings[2*sector_size:3*sector_size])
            left_clear = np.mean(lidar_readings[3*sector_size:4*sector_size])
            
            # 6. Goal direction alignment (unit vector toward goal)
            goal_direction_norm = goal_dist[:2] / (np.linalg.norm(goal_dist[:2]) + 1e-8)
            
            # Combine all LIDAR features (11 dimensions)
            lidar_features = np.array([
                min_lidar,                    # 1D
                mean_lidar,                   # 1D
                obstacle_direction[0],        # 1D
                obstacle_direction[1],        # 1D
                danger_level,                 # 1D
                front_clear,                  # 1D
                right_clear,                  # 1D
                back_clear,                   # 1D
                left_clear,                   # 1D
                goal_direction_norm[0],       # 1D
                goal_direction_norm[1]        # 1D
            ])
            
            # Create EXACT state vector as training environment
            state = np.concatenate([pos, vel, goal_dist, lidar_readings, lidar_features])
            
            # Convert to torch tensor
            state_tensor = torch.FloatTensor(state.reshape(1, -1))

            # Agent selects action
            raw_action, _ = ppo_agent.select_action(state_tensor)
            action = raw_action.flatten() # Ensure action is a 1D array
            
            # Ensure action is within expected bounds [-1, 1] (same as training)
            action = np.clip(action, -1.0, 1.0)
            
            # Calculate goal distance first
            goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
            
            # Debug: Print action values occasionally
            if step_count % 100 == 0:
                print(f"🎮 Action: [{action[0]:.3f}, {action[1]:.3f}] | "
                      f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                      f"Goal dist: {goal_distance:.2f}m")

            # Apply RDR Clear Path Rule with velocity clamping
            target_vel = action[:2]  # Desired velocity from action
            
            # Check boundary distance for R2 rule
            half_world = CONFIG['world_size'] / 2
            distances_to_boundaries = np.array([
                half_world + current_pos[0],  # Distance to west boundary
                half_world - current_pos[0],  # Distance to east boundary
                half_world + current_pos[1],  # Distance to south boundary
                half_world - current_pos[1]   # Distance to north boundary
            ])
            distance_to_boundary = float(np.min(distances_to_boundaries))
            
            # Check if clear path to goal (simple line-of-sight)
            has_clear_path = env.has_line_of_sight_to_goal() if hasattr(env, 'has_line_of_sight_to_goal') else False
            
            # RDR Rule R2: Boundary Safety (higher priority than clear path)
            if distance_to_boundary < 0.8:
                # Apply boundary safety rule: vel = max(0.5, vel - 0.25)
                boundary_escape_dir = np.zeros(2)
                
                # Calculate direction away from nearest boundary
                if current_pos[0] > half_world - 0.8:  # Close to east boundary
                    boundary_escape_dir[0] = -1.0  # Move west
                elif current_pos[0] < -(half_world - 0.8):  # Close to west boundary
                    boundary_escape_dir[0] = 1.0   # Move east
                    
                if current_pos[1] > half_world - 0.8:  # Close to north boundary
                    boundary_escape_dir[1] = -1.0  # Move south
                elif current_pos[1] < -(half_world - 0.8):  # Close to south boundary
                    boundary_escape_dir[1] = 1.0   # Move north
                
                # Normalize escape direction
                if np.linalg.norm(boundary_escape_dir) > 0:
                    boundary_escape_dir = boundary_escape_dir / np.linalg.norm(boundary_escape_dir)
                
                # Apply velocity reduction: vel = max(0.5, vel - 0.25)
                reduced_vel = target_vel - 0.25 * boundary_escape_dir
                
                # Ensure minimum speed of 0.5
                current_speed = np.linalg.norm(reduced_vel)
                if current_speed < 0.5 and current_speed > 0:
                    reduced_vel = (reduced_vel / current_speed) * 0.5
                elif current_speed == 0:
                    # If stopped, move away from boundary at min speed
                    reduced_vel = boundary_escape_dir * 0.5
                    
                target_vel = reduced_vel
                
                # Debug print for boundary safety activation
                if step_count % 30 == 0:
                    print(f"⚠️ RDR Boundary Safety Active | Distance: {distance_to_boundary:.2f}m | Escape vel: {np.linalg.norm(target_vel):.3f}")
                    
            elif has_clear_path and goal_distance > 0.5:
                # RDR Rule R1: Clear path - increase velocity toward goal
                goal_direction = (CONFIG['goal_pos'][:2] - current_pos[:2]) / (goal_distance + 1e-8)
                
                # Apply velocity boost: vel = max(1, vel + 0.2) as requested
                boosted_vel = target_vel + 0.2 * goal_direction
                
                # Ensure velocity magnitude doesn't exceed maximum of 1.0
                vel_magnitude = np.linalg.norm(boosted_vel)
                if vel_magnitude > 1.0:
                    target_vel = boosted_vel / vel_magnitude  # Normalize to unit magnitude
                else:
                    target_vel = boosted_vel
                    
                # Debug print for RDR activation
                if step_count % 50 == 0:
                    print(f"🎯 RDR Clear Path Active | Boost: {0.2:.1f} | Final vel mag: {np.linalg.norm(target_vel):.3f}")
            
            # Ensure velocity is strictly within [-1, 1] bounds
            target_vel = np.clip(target_vel, -1.0, 1.0)
            
            # Apply velocity to simulation
            data.qvel[0] = float(target_vel[0])  # X velocity  
            data.qvel[1] = float(target_vel[1])  # Y velocity
            data.qvel[2] = 0.0                   # Z velocity = 0 (no vertical movement)
            
            # Maintain constant height
            data.qpos[2] = CONFIG['uav_flight_height']
            
            # Check if goal reached for logging
            if goal_distance < 0.5 and not goal_reached:
                print(f"\n🎯 GOAL REACHED! Distance: {goal_distance:.3f}m")
                goal_reached = True
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Post-step velocity monitoring and STRICT safety clamping
            current_vel_magnitude = np.linalg.norm(data.qvel[:2])
            
            # STRICT velocity enforcement - NEVER allow velocity > 1.0
            if current_vel_magnitude > 1.0:
                # Emergency velocity clamping if simulation causes velocity escalation
                data.qvel[0] = np.clip(data.qvel[0], -1.0, 1.0)
                data.qvel[1] = np.clip(data.qvel[1], -1.0, 1.0)
                
                if step_count % 10 == 0:  # More frequent reporting
                    print(f"🚨 STRICT velocity clamp! Mag: {current_vel_magnitude:.3f} → {np.linalg.norm(data.qvel[:2]):.3f}")
            
            # Double-check: Absolute hard limit enforcement
            data.qvel[0] = max(-1.0, min(1.0, data.qvel[0]))
            data.qvel[1] = max(-1.0, min(1.0, data.qvel[1]))
            
            # Update path trail visualization
            if step_count % 5 == 0:
                update_path_trail(model, data, current_pos)
                
            # Force UAV visibility every frame to prevent transparency
            ensure_uav_visibility(model)
            
            # Forward model to update rendering
            viewer.sync()
            
            # HARD boundary enforcement - immediately stop UAV if at boundary
            half_world = CONFIG['world_size'] / 2
            boundary_margin = 0.1  # 10cm from actual boundary
            
            if (abs(current_pos[0]) > half_world - boundary_margin or 
                abs(current_pos[1]) > half_world - boundary_margin):
                # Stop UAV immediately and pull back from boundary
                data.qvel[0] = 0.0
                data.qvel[1] = 0.0
                
                # Pull UAV back to safe zone
                if current_pos[0] > half_world - boundary_margin:
                    data.qpos[0] = half_world - boundary_margin
                elif current_pos[0] < -(half_world - boundary_margin):
                    data.qpos[0] = -(half_world - boundary_margin)
                    
                if current_pos[1] > half_world - boundary_margin:
                    data.qpos[1] = half_world - boundary_margin  
                elif current_pos[1] < -(half_world - boundary_margin):
                    data.qpos[1] = -(half_world - boundary_margin)
                
                print(f"\n🛑 HARD BOUNDARY HIT - UAV STOPPED!")
                print(f"📍 Position corrected to: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
            
            # Check for complete boundary violation (simulation failure)
            if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                current_pos[2] < 0.1 or current_pos[2] > 5.0):
                collision_occurred = True
                final_vel_magnitude = np.linalg.norm(data.qvel[:2])
                print(f"\n🚨 COMPLETE BOUNDARY VIOLATION!")
                print(f"📍 UAV position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                print(f"🏃 Final velocity: ({data.qvel[0]:.3f}, {data.qvel[1]:.3f}) | Magnitude: {final_vel_magnitude:.3f}")
                print(f"🎯 Goal distance: {goal_distance:.2f}m")
                print(f"📊 Episode length: {step_count} steps")
                print(f"🚫 UAV went out of bounds!")
                print(f"⚠️ MISSION FAILED - RESTARTING SIMULATION!")
                break
            
            # Check for collision
            has_collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(data.qpos[:3], obstacles)
            if has_collision:
                collision_occurred = True
                print(f"\n💥 COLLISION DETECTED!")
                print(f"💀 UAV crashed into obstacle: {obstacle_id}")
                print(f"📏 Collision distance: {collision_dist:.3f}m")
                print(f"📍 UAV position at crash: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
                print(f"⚠️ MISSION FAILED!")
                break
            
            # Goal checking is now handled in the velocity control section above
            
            # Status updates
            if step_count % 100 == 0:
                print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) | Goal dist: {goal_distance:.2f}m")
            
            step_count += 1
            time.sleep(CONFIG['control_dt'])
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Final mission status
    if collision_occurred:
        print("\n" + "="*50)
        print("💥 MISSION RESULT: FAILURE - COLLISION DETECTED")
        print("🚨 The UAV crashed into an obstacle!")
        print("💡 Try adjusting flight path or obstacle avoidance")
        print("="*50)
        
        # Show crash scene for a moment
        for _ in range(200):
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)
            
    elif mission_complete:
        # Celebration hover
        print("\n" + "="*50)
        print("🎉 MISSION RESULT: SUCCESS!")
        print("🏆 UAV successfully navigated to goal without collision!")
        print(f"📍 Start: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}] → Goal: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}]")
        print("="*50)
        print("🎊 Celebration hover sequence...")
        
        for _ in range(500):
            data.ctrl[:] = [4.0, 4.0, 4.0, 4.0]  # Stronger hover thrust
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    else:
        print("\n" + "="*50)
        print("⏹️ SIMULATION TERMINATED")
        print("="*50)

    print("Simulation finished.")