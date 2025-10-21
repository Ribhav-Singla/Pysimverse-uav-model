# UAV Manual Control Simulation
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
import torch
import os
import sys
import threading
import keyboard  # You may need to install: pip install keyboard
import pickle
from datetime import datetime
from collections import deque

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
    'lidar_range': 2.9,
    'lidar_num_rays': 16,
    'step_reward': -0.01,
    
    # Render-specific parameters (do not affect agent logic)
    'kp_pos': 1.5,
    'path_trail_length': 500,
    
    # Manual control parameters
    'manual_speed': 0.8,  # Base movement speed for manual control
    'manual_acceleration': 0.1,  # How quickly to reach target speed
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
        
        # Add obstacles
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
        collision_dist = CONFIG['collision_distance']
        
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

class ManualControlSystem:
    """System to handle keyboard input and data recording"""
    def __init__(self):
        self.target_velocity = np.array([0.0, 0.0])  # Target velocity from keyboard
        self.current_velocity = np.array([0.0, 0.0])  # Smoothed current velocity
        self.recording_buffer = []  # Buffer to store recorded data
        self.is_recording = True
        self.controls_active = True
        
        # Control instructions
        self.print_controls()
        
    def print_controls(self):
        print("\n" + "="*60)
        print("🎮 MANUAL UAV CONTROL SYSTEM")
        print("="*60)
        print("🔵 MOVEMENT CONTROLS:")
        print("   ↑ Arrow Up    - Move Forward  (+Y direction)")
        print("   ↓ Arrow Down  - Move Backward (-Y direction)")
        print("   ← Arrow Left  - Move Left     (-X direction)")
        print("   → Arrow Right - Move Right    (+X direction)")
        print("")
        print("🔵 OTHER CONTROLS:")
        print("   SPACE         - Stop/Brake (zero velocity)")
        print("   R             - Reset UAV to start position")
        print("   ESC           - Exit simulation")
        print("")
        print("📊 DATA RECORDING:")
        print("   S             - Save recorded data to file")
        print("   C             - Clear recording buffer")
        print("")
        print("💡 MISSION: Navigate from START (green) to GOAL (blue)")
        print("⚠️  Avoid obstacles and stay within boundaries!")
        print("="*60)
    
    def update_controls(self):
        """Update target velocity based on keyboard input"""
        target_vel = np.array([0.0, 0.0])
        
        if keyboard.is_pressed('up'):
            target_vel[1] += CONFIG['manual_speed']  # Forward
        if keyboard.is_pressed('down'):
            target_vel[1] -= CONFIG['manual_speed']  # Backward
        if keyboard.is_pressed('left'):
            target_vel[0] -= CONFIG['manual_speed']  # Left
        if keyboard.is_pressed('right'):
            target_vel[0] += CONFIG['manual_speed']  # Right
        if keyboard.is_pressed('space'):
            target_vel = np.array([0.0, 0.0])  # Stop
            
        self.target_velocity = target_vel
        
        # Smooth velocity transition for realistic control
        vel_diff = self.target_velocity - self.current_velocity
        self.current_velocity += vel_diff * CONFIG['manual_acceleration']
        
        # Return the current smoothed velocity
        return self.current_velocity.copy()
    
    def handle_special_keys(self, data, model, start_pos):
        """Handle special keyboard commands"""
        if keyboard.is_pressed('r'):
            # Reset UAV position
            data.qpos[:3] = start_pos
            data.qpos[3:7] = [1, 0, 0, 0]  # Reset orientation
            data.qvel[:] = 0  # Zero velocity
            self.current_velocity = np.array([0.0, 0.0])
            self.target_velocity = np.array([0.0, 0.0])
            print("🔄 UAV reset to start position")
            time.sleep(0.2)  # Prevent repeated resets
            
        if keyboard.is_pressed('s'):
            self.save_data()
            time.sleep(0.5)  # Prevent repeated saves
            
        if keyboard.is_pressed('c'):
            self.clear_buffer()
            time.sleep(0.3)  # Prevent repeated clears
            
        if keyboard.is_pressed('esc'):
            self.controls_active = False
    
    def record_step(self, position, velocity, goal_pos, lidar_readings, action, collision, goal_reached):
        """Record a step of data for later analysis"""
        if self.is_recording:
            step_data = {
                'timestamp': time.time(),
                'position': position.copy(),
                'velocity': velocity.copy(),
                'goal_position': goal_pos.copy(),
                'goal_distance': np.linalg.norm(position - goal_pos),
                'lidar_readings': lidar_readings.copy(),
                'action': action.copy(),
                'collision': collision,
                'goal_reached': goal_reached
            }
            self.recording_buffer.append(step_data)
    
    def save_data(self):
        """Save recorded data to file"""
        if not self.recording_buffer:
            print("📝 No data to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_control_data_{timestamp}.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.recording_buffer, f)
            print(f"💾 Data saved to {filename} ({len(self.recording_buffer)} steps)")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def clear_buffer(self):
        """Clear the recording buffer"""
        self.recording_buffer.clear()
        print("🗑️ Recording buffer cleared")
    
    def get_buffer_info(self):
        """Get information about the current buffer"""
        return {
            'steps': len(self.recording_buffer),
            'size_mb': len(pickle.dumps(self.recording_buffer)) / (1024 * 1024) if self.recording_buffer else 0
        }

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

# === MAIN EXECUTION ===

# Generate obstacles first so we can check positions against them
print("🏗️ Generating complex environment with static obstacles...")
obstacles = EnvironmentGenerator.generate_obstacles()

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
    print(f"🎯 Using potentially unsafe goal: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")

# Create environment XML with the obstacles
EnvironmentGenerator.create_xml_with_obstacles(obstacles)

# Load the model and create the simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"🚁 Manual UAV Navigation Environment Loaded!")
print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
print(f"🎯 Mission: Navigate from START (green) to GOAL (blue) using arrow keys")
print(f"🏠 Start Position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
print(f"🏁 Goal Position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
print(f"🚧 Static obstacles: {CONFIG['static_obstacles']}")
print(f"📏 Obstacle height: {CONFIG['obstacle_height']}m")
print(f"✈️ UAV flight height: {CONFIG['uav_flight_height']}m")
print(f"🛣️ Green path trail: {CONFIG['path_trail_length']} points")

# Initialize manual control system
control_system = ManualControlSystem()
path_history = []

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position
    data.qpos[:3] = CONFIG['start_pos']
    data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation (quaternion)
    
    print("\n🎮 Manual Control Active! Use arrow keys to navigate!")
    time.sleep(2)
    
    step_count = 0
    mission_complete = False
    collision_occurred = False
    goal_reached = False
    
    try:
        while viewer.is_running() and control_system.controls_active and not mission_complete and not collision_occurred:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Update manual controls
            manual_action = control_system.update_controls()
            control_system.handle_special_keys(data, model, CONFIG['start_pos'])
            
            # Create observation for data recording (same as PPO agent would see)
            goal_dist = CONFIG['goal_pos'] - current_pos
            lidar_readings = get_lidar_readings(current_pos, obstacles)
            
            # === LIDAR FEATURE ENGINEERING (for recording compatibility) ===
            min_lidar = np.min(lidar_readings)
            mean_lidar = np.mean(lidar_readings)
            
            closest_idx = np.argmin(lidar_readings)
            closest_angle = (2 * np.pi * closest_idx) / CONFIG['lidar_num_rays']
            obstacle_direction = np.array([np.cos(closest_angle), np.sin(closest_angle)])
            
            danger_threshold = 0.36
            num_close_obstacles = np.sum(lidar_readings < danger_threshold)
            danger_level = num_close_obstacles / CONFIG['lidar_num_rays']
            
            sector_size = len(lidar_readings) // 4
            front_clear = np.mean(lidar_readings[0:sector_size])
            right_clear = np.mean(lidar_readings[sector_size:2*sector_size])
            back_clear = np.mean(lidar_readings[2*sector_size:3*sector_size])
            left_clear = np.mean(lidar_readings[3*sector_size:4*sector_size])
            
            goal_direction_norm = goal_dist[:2] / (np.linalg.norm(goal_dist[:2]) + 1e-8)
            
            lidar_features = np.array([
                min_lidar, mean_lidar, obstacle_direction[0], obstacle_direction[1],
                danger_level, front_clear, right_clear, back_clear, left_clear,
                goal_direction_norm[0], goal_direction_norm[1]
            ])
            
            # --- Height Control ---
            current_height = data.qpos[2]
            desired_height = CONFIG['uav_flight_height']
            
            # Simple P-controller to maintain height
            height_error = desired_height - current_height
            vz_correction = CONFIG['kp_pos'] * height_error
            
            # Apply manual control actions
            data.qvel[0] = manual_action[0]  # X-velocity from manual control
            data.qvel[1] = manual_action[1]  # Y-velocity from manual control
            data.qvel[2] = vz_correction     # Height controller manages Z-velocity
            
            # Check if UAV is at goal position
            goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
            if goal_distance < 0.5 and not goal_reached:
                goal_reached = True
                mission_complete = True
                print(f"\n🎉 GOAL REACHED!")
                print(f"📍 Final position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                print(f"📏 Distance to goal: {goal_distance:.3f}m")
                print(f"🏆 MISSION COMPLETE!")
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
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
            
            # Record data for this step
            control_system.record_step(
                position=current_pos,
                velocity=current_vel,
                goal_pos=CONFIG['goal_pos'],
                lidar_readings=lidar_readings,
                action=manual_action,
                collision=has_collision,
                goal_reached=goal_reached
            )
            
            # Status updates
            if step_count % 100 == 0:
                buffer_info = control_system.get_buffer_info()
                print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) | "
                      f"Goal dist: {goal_distance:.2f}m | Buffer: {buffer_info['steps']} steps")
            
            step_count += 1
            time.sleep(CONFIG['control_dt'])
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Final mission status and data handling
    buffer_info = control_system.get_buffer_info()
    print(f"\n📊 Total data recorded: {buffer_info['steps']} steps ({buffer_info['size_mb']:.1f} MB)")
    
    if collision_occurred:
        print("\n" + "="*50)
        print("💥 MISSION RESULT: FAILURE - COLLISION DETECTED")
        print("🚨 The UAV crashed into an obstacle!")
        print("💡 Try different navigation strategy")
        print("="*50)
        
    elif mission_complete:
        print("\n" + "="*50)
        print("🎉 MISSION RESULT: SUCCESS!")
        print("🏆 UAV successfully navigated to goal under manual control!")
        print(f"📍 Start: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}] → Goal: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}]")
        print("="*50)
        
    else:
        print("\n" + "="*50)
        print("⏹️ SIMULATION TERMINATED")
        print("="*50)
    
    # Ask user if they want to save recorded data
    if buffer_info['steps'] > 0:
        print(f"\n💾 You have {buffer_info['steps']} recorded steps.")
        print("📝 Press 'S' to save data, or 'ESC' to exit without saving...")
        
        while viewer.is_running():
            if keyboard.is_pressed('s'):
                control_system.save_data()
                break
            elif keyboard.is_pressed('esc'):
                print("📤 Exiting without saving data.")
                break
            time.sleep(0.1)

    print("Simulation finished.")