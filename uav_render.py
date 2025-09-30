# UAV Navigation Simulation
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
import torch
from ppo_agent import PPOAgent
from uav_env import UAVEnv, EnvironmentGenerator

# Define the path to your XML model
MODEL_PATH = "environment.xml"

# Configuration parameters - MUST match training environment
CONFIG = {
    'start_pos': np.array([-4.0, -4.0, 1.0]),
    'goal_pos': np.array([4.0, 4.0, 1.0]),
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,
    'static_obstacles': 10,
    'min_obstacle_size': 0.05,
    'max_obstacle_size': 0.12,
    'collision_distance': 0.1,
    'control_dt': 0.05,
    'boundary_penalty': -10,
    'lidar_range': 2.75,
    'lidar_num_rays': 16,
    'step_reward': 0.01,
    
    # Render-specific parameters (do not affect agent logic)
    'kp_pos': 1.5,
    'path_trail_length': 200,
}

class EnvironmentGenerator:
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
    def create_xml_with_obstacles():
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
        
        # Add obstacles
        obstacles = EnvironmentGenerator.generate_obstacles()
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
        
        lidar_readings.append(min_distance)
    
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

# Generate complex environment
print("🏗️ Generating complex environment with static obstacles...")
obstacles = EnvironmentGenerator.create_xml_with_obstacles()

# Load the model and create the simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"🚁 Complex UAV Navigation Environment Loaded!")
print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
print(f"🎯 Mission: Navigate from START (green) to GOAL (blue)")
print(f"🚧 Static obstacles: {CONFIG['static_obstacles']}")
print(f"📏 Obstacle height: {CONFIG['obstacle_height']}m")
print(f"✈️ UAV flight height: {CONFIG['uav_flight_height']}m")
print(f"🛣️ Green path trail: {CONFIG['path_trail_length']} points")

# Initialize PPO agent
env = UAVEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
ppo_agent = PPOAgent(state_dim, action_dim, 0, 0, 0, 0, 0)

# Load the latest trained model
try:
    checkpoint_dir = "PPO_preTrained/UAVEnv/"
    files = os.listdir(checkpoint_dir)
    paths = [os.path.join(checkpoint_dir, basename) for basename in files]
    latest_model = max(paths, key=os.path.getctime)
    ppo_agent.load(latest_model)
    print(f"🤖 Trained PPO agent loaded successfully from {latest_model}!")
except:
    print("⚠️ Could not load trained agent. Using random actions.")

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
    
    try:
        while viewer.is_running() and not mission_complete and not collision_occurred:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Create observation for PPO agent (MUST match training environment)
            goal_dist = CONFIG['goal_pos'] - current_pos
            lidar_readings = get_lidar_readings(current_pos, obstacles)
            state = np.concatenate([current_pos, current_vel, goal_dist, lidar_readings])

            # Agent selects action
            raw_action, _ = ppo_agent.select_action(state)
            action = raw_action.flatten() # Ensure action is a 1D array

            # --- Height Control ---
            # The agent controls X and Y velocity, but we override Z for stable height.
            current_height = data.qpos[2]
            desired_height = CONFIG['uav_flight_height']
            
            # Simple P-controller to maintain height
            height_error = desired_height - current_height
            vz_correction = CONFIG['kp_pos'] * height_error
            
            # Apply velocity control
            data.qvel[0] = action[0]      # Agent controls X-velocity
            data.qvel[1] = action[1]      # Agent controls Y-velocity
            data.qvel[2] = vz_correction  # Height controller manages Z-velocity
            
            # Step the simulation
            mujoco.mj_step(model, data)

            # Enforce velocity constraints
            vel = data.qvel[:3]
            speed = np.linalg.norm(vel)
            if speed < 0.5:
                data.qvel[:3] = (vel / speed) * 0.5 if speed > 0 else np.zeros(3)
            elif speed > 1.5:
                data.qvel[:3] = (vel / speed) * 1.5
            
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
            
            # Check if mission is complete
            goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
            if goal_distance < 0.5:
                mission_complete = True
                print(f"\n🎉 MISSION COMPLETE! UAV reached the goal!")
                print(f"Final position: {current_pos}")
                print(f"Distance to goal: {goal_distance:.2f}m")
                print(f"🏆 SUCCESS! No collisions occurred!")
            
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