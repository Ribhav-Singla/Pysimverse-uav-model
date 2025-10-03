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
    'static_obstacles': 9,
    'min_obstacle_size': 0.05,
    'max_obstacle_size': 0.12,
    'collision_distance': 0.1,
    'control_dt': 0.05,
    'boundary_penalty': -100,
    'step_reward': -0.01,
    
    # Camera configuration - MUST match uav_env.py
    'camera_enabled': True,
    'camera_width': 224,
    'camera_height': 224,
    'camera_fov': 90,
    'camera_fps': 2.0,
    'use_vision_model': True,
    
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

# LIDAR functions removed - now using vision-based system
# All obstacle and goal detection is handled by the UAVEnv vision system

# LIDAR ray intersection functions removed - now using vision-based system

import os

# ... (rest of the imports)

# ... (CONFIG and EnvironmentGenerator class)

# Generate obstacles first so we can check positions against them
print("🏗️ Generating complex environment with static obstacles...")
obstacles = EnvironmentGenerator.generate_obstacles()

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
safety_radius = 0.8  # Keep at least 0.8m from obstacles

# Generate dynamic start position from one of the corners
start_pos_safe = False
start_safety_check_attempts = 0

print("🏠 Generating random start position from corners...")
while not start_pos_safe and start_safety_check_attempts < max_attempts:
    start_safety_check_attempts += 1
    # Get a random corner position
    CONFIG['start_pos'] = EnvironmentGenerator.get_random_corner_position()
    
    # Check if it's safe
    if check_position_safety(CONFIG['start_pos'], obstacles, safety_radius):
        start_pos_safe = True
        print(f"✅ Safe start position set to: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
        break

if not start_pos_safe:
    print("⚠️ Warning: Could not find safe start position after multiple attempts!")
    print("⚠️ Using default corner position [-3, -3, 1]")
    CONFIG['start_pos'] = np.array([-3.0, -3.0, 1.0])

# Generate dynamic goal position anywhere in the map
goal_pos_safe = False
goal_safety_check_attempts = 0

print("🎯 Generating random goal position anywhere in map...")
while not goal_pos_safe and goal_safety_check_attempts < max_attempts:
    goal_safety_check_attempts += 1
    # Get a random goal position anywhere in the map
    CONFIG['goal_pos'] = EnvironmentGenerator.get_random_goal_position()
    
    # Ensure goal is not too close to start position
    start_goal_distance = np.linalg.norm(CONFIG['goal_pos'][:2] - CONFIG['start_pos'][:2])
    
    # Check if it's safe and far enough from start
    if (check_position_safety(CONFIG['goal_pos'], obstacles, safety_radius) and 
        start_goal_distance > 2.0):  # At least 2m apart
        goal_pos_safe = True
        print(f"✅ Safe goal position set to: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
        print(f"📏 Start-Goal distance: {start_goal_distance:.1f}m")
        break
    
if not goal_pos_safe:
    print("⚠️ Warning: Could not find a safe goal position after multiple attempts!")
    print("⚠️ Goal may be too close to obstacles.")
    print(f"� Using potentially unsafe goal: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")

# Create environment XML with the obstacles
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
env = UAVEnv()
state_dim = env.observation_space.shape[0]  # Vision-based: 41D [pos(3), vel(3), goal_dist(3), visual_features(32)]
action_dim = env.action_space.shape[0]

print(f"🧠 VISION-BASED ARCHITECTURE LOADED:")
print(f"   - Observation Space: {state_dim}D (Vision-Based)")
print(f"   - UAV State: 9D (position, velocity, goal distance)")
print(f"   - Visual Features: 32D (camera-based obstacle & goal detection)")
print(f"   - Action Space: {action_dim}D (velocity control)")

# Use EXACT parameters from training.py
action_std = 1.0            # Start with 100% exploration
lr_actor = 0.0001           # learning rate for actor
lr_critic = 0.0004          # learning rate for critic
gamma = 0.999               # increased for longer-term planning
K_epochs = 11               # Match training.py exactly (was 12)
eps_clip = 0.1              # clip parameter for PPO

# Initialize PPO agent
ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

# Load the trained model
try:
    # Load the single weight file
    model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights.pth"
    
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
    
    # Goal stabilization variables
    goal_reached = False
    goal_stabilization_steps = 0
    goal_hold_duration = 30  # Steps before confirming stabilization (1 second)
    stay_at_goal_indefinitely = True  # Keep UAV at goal after reaching it
    
    try:
        while viewer.is_running() and not mission_complete and not collision_occurred:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Use environment to get proper observation - IDENTICAL to training
            # This ensures we get the exact same 41D vision-based observation
            obs = env._get_obs()
            
            # Convert to torch tensor
            state_tensor = torch.FloatTensor(obs.reshape(1, -1))

            # Agent selects action
            raw_action, _ = ppo_agent.select_action(state_tensor)
            action = raw_action.flatten() # Ensure action is a 1D array

            # --- Height Control ---
            # The agent controls X and Y velocity, but we override Z for stable height.
            current_height = data.qpos[2]
            desired_height = CONFIG['uav_flight_height']
            
            # Simple P-controller to maintain height
            height_error = desired_height - current_height
            vz_correction = CONFIG['kp_pos'] * height_error
            
            # Check if UAV is at goal position BEFORE applying velocity
            goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
            
            # GOAL STABILIZATION: If UAV reaches goal, apply strong braking to keep it there
            if goal_distance < 0.5:  # Same threshold as training environment
                if not goal_reached:
                    print(f"\n🎯 GOAL REACHED! Stabilizing UAV at goal position...")
                    goal_reached = True
                    goal_stabilization_steps = 0
                
                # Apply VERY strong braking forces to LOCK UAV at goal
                goal_vector = CONFIG['goal_pos'] - current_pos
                
                # Check if we're very close to goal center (within 0.1m)
                if goal_distance < 0.1:
                    # LOCK MODE: Virtually stop all movement
                    # Apply extremely strong velocity damping (99% reduction)
                    stabilization_vel = -data.qvel[:2] * 0.99
                    
                    # Add tiny position correction to keep centered
                    stabilization_vel += goal_vector[:2] * 5.0
                else:
                    # APPROACH MODE: Strong position correction with damping
                    position_correction = goal_vector[:2] * 3.0  # Very strong pull toward goal
                    
                    # Strong velocity damping to eliminate momentum
                    velocity_damping = -data.qvel[:2] * 0.9  # Reduce current velocity by 90%
                    
                    # Combine corrections with priority on stopping motion
                    stabilization_vel = position_correction + velocity_damping
                
                # Limit stabilization velocity to prevent overshooting
                stabilization_speed = np.linalg.norm(stabilization_vel)
                max_stabilization_speed = 0.2 if goal_distance < 0.1 else 0.4
                if stabilization_speed > max_stabilization_speed:
                    stabilization_vel = (stabilization_vel / stabilization_speed) * max_stabilization_speed
                
                # Apply stabilization velocity instead of action-based velocity
                data.qvel[0] = float(stabilization_vel[0])
                data.qvel[1] = float(stabilization_vel[1])
                data.qvel[2] = vz_correction  # Height controller still manages Z-velocity
                
                goal_stabilization_steps += 1
                
                # Print stabilization confirmation (only once after sufficient steps)
                if goal_stabilization_steps == goal_hold_duration:
                    print(f"\n✅ GOAL SUCCESSFULLY REACHED AND STABILIZED!")
                    print(f"📍 UAV position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                    print(f"📍 Goal position: ({CONFIG['goal_pos'][0]:.2f}, {CONFIG['goal_pos'][1]:.2f}, {CONFIG['goal_pos'][2]:.2f})")
                    print(f"📏 Distance to goal: {goal_distance:.3f}m")
                    print(f"🔒 UAV is now LOCKED at goal position")
                    print(f"🏆 MISSION COMPLETE! (Press ESC or close window to exit)")
                
                # Optional: Allow mission to complete but don't set flag if we want to stay indefinitely
                # Keep UAV at goal instead of ending simulation
                # mission_complete = stay_at_goal_indefinitely is False
                    
            else:
                # Normal velocity control when not at goal
                data.qvel[0] = action[0]      # Agent controls X-velocity
                data.qvel[1] = action[1]      # Agent controls Y-velocity
                data.qvel[2] = vz_correction  # Height controller manages Z-velocity
                
                # Reset goal status if UAV moves too far from goal during stabilization
                if goal_reached and goal_distance > 1.0:
                    goal_reached = False
                    goal_stabilization_steps = 0
                    print(f"\n⚠️ UAV left goal area during stabilization. Resetting...")
            
            # Step the simulation
            mujoco.mj_step(model, data)

            # Enforce velocity constraints (match training velocity limits)
            vel = data.qvel[:3]
            speed = np.linalg.norm(vel)
            # Using the same velocity limits as in the velocity curriculum
            min_vel = 0.1
            max_vel = 1.0  # Medium velocity for demonstration
            
            if speed < min_vel and speed > 0:
                data.qvel[:3] = (vel / speed) * min_vel if speed > 0 else np.zeros(3)
            elif speed > max_vel:
                data.qvel[:3] = (vel / speed) * max_vel
            
            # Update path trail visualization
            if step_count % 5 == 0:
                update_path_trail(model, data, current_pos)
                
            # Force UAV visibility every frame to prevent transparency
            ensure_uav_visibility(model)
            
            # Forward model to update rendering
            viewer.sync()
            
            # Check for boundary violation
            half_world = CONFIG['world_size'] / 2
            if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                current_pos[2] < 0.1 or current_pos[2] > 5.0):
                collision_occurred = True
                print(f"\n🚨 BOUNDARY VIOLATION DETECTED!")
                print(f"📍 UAV position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                print(f"🚫 UAV flew outside simulation boundaries!")
                print(f"⚠️ MISSION FAILED!")
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