# UAV Performance Comparison System
# Integrated comparison of Human Expert, Neural Only, and Neurosymbolic approaches
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
import argparse
import csv
from datetime import datetime
from collections import deque, defaultdict

# Try to import keyboard for manual control
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: 'keyboard' module not available. Install with: pip install keyboard")
    print("Manual control trials will be disabled.")
    KEYBOARD_AVAILABLE = False

# Import custom modules for agent comparison
try:
    from uav_env import UAVEnv, EnvironmentGenerator as UAVEnvGenerator
    from ppo_agent import PPOAgent
    from uav_agent_runner import UAVAgentRunner
    AGENTS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Agent modules not available. Only manual control will work.")
    AGENTS_AVAILABLE = False

# Add argument parsing for different execution modes
parser = argparse.ArgumentParser(description='UAV Performance Comparison System')
parser.add_argument('--mode', type=str, choices=['manual', 'comparison', 'level'], 
                   default='manual', help='Execution mode: manual (original manual control), comparison (single level), level (multi-level range)')
parser.add_argument('--level', type=int, default=1, 
                   help='Curriculum level for single comparison (1-10)')
parser.add_argument('--start-level', type=int, default=1, 
                   help='Starting curriculum level for multi-level comparison')
parser.add_argument('--end-level', type=int, default=10, 
                   help='Ending curriculum level for multi-level comparison')

args = parser.parse_args()

# Define the path to your XML model
MODEL_PATH = "environment.xml"

# Performance Tracking Classes
class PerformanceTracker:
    """Track performance metrics for each approach"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new trial"""
        self.start_time = None
        self.end_time = None
        self.path_length = 0.0
        self.step_count = 0
        self.success = False
        self.collision = False
        self.out_of_bounds = False
        self.timeout = False
        self.final_distance = None
        self.path_positions = []
    
    def start_trial(self, start_pos):
        """Start tracking a new trial"""
        self.reset()
        self.start_time = time.time()
        self.path_positions = [start_pos.copy()]
    
    def update_step(self, position, goal_pos):
        """Update metrics for each step"""
        if len(self.path_positions) > 0:
            step_distance = np.linalg.norm(position - self.path_positions[-1])
            self.path_length += step_distance
        
        self.path_positions.append(position.copy())
        self.step_count += 1
        self.final_distance = np.linalg.norm(position - goal_pos)
    
    def end_trial(self, success=False, collision=False, out_of_bounds=False, timeout=False):
        """End the trial and calculate final metrics"""
        self.end_time = time.time()
        self.success = success
        self.collision = collision
        self.out_of_bounds = out_of_bounds
        self.timeout = timeout
    
    def get_metrics(self):
        """Get performance metrics dictionary"""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            'path_length': self.path_length,
            'step_count': self.step_count,
            'success': self.success,
            'collision': self.collision,
            'out_of_bounds': self.out_of_bounds,
            'timeout': self.timeout,
            'final_distance': self.final_distance,
            'duration': duration,
            'path_efficiency': self.calculate_path_efficiency()
        }
    
    def calculate_path_efficiency(self):
        """Calculate path efficiency (direct distance / actual path length)"""
        if len(self.path_positions) < 2 or self.path_length == 0:
            return 0.0
        
        direct_distance = np.linalg.norm(self.path_positions[-1] - self.path_positions[0])
        return direct_distance / self.path_length if self.path_length > 0 else 0.0

# Parse command line arguments for configurable parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='UAV Performance Comparison System')
    parser.add_argument('--mode', choices=['manual', 'comparison', 'level'], default='comparison',
                       help='Run mode: manual (manual control only), comparison (single level comparison), level (multi-level range)')
    parser.add_argument('--level', type=int, default=1, 
                       help='Curriculum level for comparison (default: 1)')
    parser.add_argument('--start_level', '--start-level', type=int, default=1,
                       help='Starting curriculum level (default: 1)')  
    parser.add_argument('--end_level', '--end-level', type=int, default=10,
                       help='Ending curriculum level (default: 10)')
    parser.add_argument('--static_obstacles', type=int, default=10, 
                       help='Number of static obstacles for manual mode (default: 10)')
    parser.add_argument('--start_pos', nargs=3, type=float, default=[-3.0, -3.0, 1.0],
                       help='Start position [x, y, z] for manual mode (default: [-3.0, -3.0, 1.0])')
    parser.add_argument('--goal_pos', nargs=3, type=float, default=[3.0, 3.0, 1.0],
                       help='Goal position [x, y, z] for manual mode (default: [3.0, 3.0, 1.0])')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Configuration parameters - MUST match training environment
CONFIG = {
    'start_pos': np.array(args.start_pos),  # Configurable start position
    'goal_pos': np.array(args.goal_pos),    # Configurable goal position
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,
    'static_obstacles': args.static_obstacles,  # Configurable obstacle count
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
    def generate_obstacles(num_obstacles=None):
        """Generate obstacles with specified count for curriculum learning"""
        if num_obstacles is None:
            num_obstacles = CONFIG['static_obstacles']
            
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        # Generate potential positions with grid-based distribution
        grid_size = int(math.sqrt(max(num_obstacles, 9))) + 1
        cell_size = world_size / grid_size
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = -half_world + (i + 0.5) * cell_size
                y = -half_world + (j + 0.5) * cell_size
                positions.append((x, y))
        
        random.shuffle(positions)
        
        for i in range(num_obstacles):
            x, y = positions[i % len(positions)]
            
            # Add some randomness to avoid perfect grid alignment
            x += random.uniform(-cell_size/4, cell_size/4)
            y += random.uniform(-cell_size/4, cell_size/4)
            
            # Ensure obstacles stay within bounds
            x = max(-half_world + 0.5, min(half_world - 0.5, x))
            y = max(-half_world + 0.5, min(half_world - 0.5, y))
            
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

class ComparisonEnvironment:
    """Environment wrapper for running comparison trials"""
    
    def __init__(self, level=1):
        self.level = level
        self.obstacles = []
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment for the current level with deterministic obstacle generation"""
        # Use level-based seed for reproducible obstacle generation
        random.seed(1000 + self.level)
        np.random.seed(1000 + self.level)
        
        # Generate obstacles for current level (now deterministic)
        self.obstacles = EnvironmentGenerator.generate_obstacles(self.level)
        
        # Generate safe start and goal positions (also deterministic)
        self.generate_safe_positions()
        
        # Store positions for reuse across trials
        self.start_pos = CONFIG['start_pos'].copy()
        self.goal_pos = CONFIG['goal_pos'].copy()
        
        # Create XML file
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
    
    def generate_safe_positions(self):
        """Generate safe start and goal positions for the current level (deterministic)"""
        half_world = CONFIG['world_size'] / 2 - 1.0
        
        # Deterministic corner selection based on level
        corners = [
            np.array([-half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([-half_world, half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, half_world, CONFIG['uav_flight_height']])
        ]
        
        # Use deterministic start corner (level-based)
        start_corner_idx = self.level % len(corners)
        CONFIG['start_pos'] = corners[start_corner_idx].copy()
        
        # Use deterministic goal corner (different from start, level-based)
        available_goal_corners = [c for i, c in enumerate(corners) if i != start_corner_idx]
        goal_corner_idx = (self.level + 1) % len(available_goal_corners)
        CONFIG['goal_pos'] = available_goal_corners[goal_corner_idx].copy()
        
        print(f"📍 Level {self.level}: Start {CONFIG['start_pos'][:2]}, Goal {CONFIG['goal_pos'][:2]} (deterministic)")
    
    def reset_uav(self):
        """Reset UAV to start position"""
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']
        self.data.qvel[:] = 0

class ManualControlSystem:
    """System to handle keyboard input and data recording"""
    def __init__(self, comparison_mode=False):
        self.target_velocity = np.array([0.0, 0.0])  # Target velocity from keyboard
        self.current_velocity = np.array([0.0, 0.0])  # Smoothed current velocity
        self.recording_buffer = []  # Buffer to store recorded data
        self.is_recording = True
        self.controls_active = True
        self.comparison_mode = comparison_mode
        self.trial_complete = False
        self.collision_occurred = False
        self.goal_reached = False
        
        # Control instructions
        if not comparison_mode:
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

class PerformanceComparison:
    """Main comparison class"""
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.save_path = "performance_comparison_results.csv"
        self.check_model_files()
    
    def check_model_files(self):
        """Check which model files are available"""
        if not AGENTS_AVAILABLE:
            self.neural_available = False
            self.neurosymbolic_available = False
            return
            
        print("🤖 Checking available PPO models...")
        
        self.neural_model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth"
        self.neurosymbolic_model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_1_0.pth"
        
        self.neural_available = os.path.exists(self.neural_model_path)
        self.neurosymbolic_available = os.path.exists(self.neurosymbolic_model_path)
        
        print(f"✅ Neural model: {'Available' if self.neural_available else 'Missing'}")
        print(f"✅ Neurosymbolic model: {'Available' if self.neurosymbolic_available else 'Missing'}")
        
        if not (self.neural_available or self.neurosymbolic_available):
            print("❌ No trained models found! Only manual control will be available.")
    
    def run_agent_trial(self, env, model_path, agent_type="Neural", max_steps=5000):
        """Run an agent trial using the UAVAgentRunner"""
        if not AGENTS_AVAILABLE:
            print(f"❌ {agent_type} agent not available - missing agent modules")
            return None
            
        print(f"🤖 Starting {agent_type} agent trial...")
        
        # Ensure CONFIG uses the same start/goal positions as environment
        CONFIG['start_pos'] = env.start_pos.copy()
        CONFIG['goal_pos'] = env.goal_pos.copy()
        
        # Setup neurosymbolic configuration based on agent type
        if agent_type == "Neurosymbolic":
            ns_cfg = {'use_neurosymbolic': True, 'lambda': 1.0}
        else:
            ns_cfg = {'use_neurosymbolic': False, 'lambda': 0.0}
        
        # Create agent runner
        runner = UAVAgentRunner(
            model_path=model_path,
            ns_cfg=ns_cfg,
            max_steps=max_steps,
            show_viewer=True,
            verbose=False
        )
        
        try:
            # Setup environment with current obstacles
            runner.setup_environment(obstacles=env.obstacles)
            
            # Run trial
            results = runner.run_trial()
            
            # Convert results to our format
            direct_distance = np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos'])
            metrics = {
                'path_length': results['path_length'],
                'step_count': results['step_count'],
                'success': results['success'],
                'collision': results['collision'],
                'out_of_bounds': results['out_of_bounds'],
                'timeout': results['timeout'],
                'final_distance': results['final_distance'],
                'duration': 0.0,  # Not tracked in runner
                'path_efficiency': direct_distance / results['path_length'] if results['path_length'] > 0 else 0.0
            }
            
            # Print result
            if results['success']:
                print("🎯 Goal reached!")
            elif results['collision']:
                print("💥 Collision detected!")
            elif results['out_of_bounds']:
                print("🚫 Out of bounds!")
            elif results['timeout']:
                print("⏰ Trial timed out!")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Agent trial failed: {e}")
            return None
    
    def run_level_comparison(self, level):
        """Run comparison for a specific level"""
        print(f"\n🎯 LEVEL {level} - {level} OBSTACLES")
        print("="*50)
        
        # Generate environment with deterministic obstacles
        env = ComparisonEnvironment(level)
        level_results = {}
        
        print(f"🗺️  Generated level {level} environment with {len(env.obstacles)} obstacles")
        print(f"📍 Start: [{env.start_pos[0]:.1f}, {env.start_pos[1]:.1f}] → Goal: [{env.goal_pos[0]:.1f}, {env.goal_pos[1]:.1f}]")
        print(f"⚖️  ALL APPROACHES will use IDENTICAL configuration for fair comparison")
        
        # 1. Human Expert Trial
        if KEYBOARD_AVAILABLE:
            print(f"\n--- Human Expert Trial ---")
            try:
                metrics = self.run_manual_trial(env)
                level_results["Human Expert"] = metrics
                self.print_trial_results("Human Expert", metrics)
            except KeyboardInterrupt:
                print(f"❌ Human Expert trial cancelled by user")
                level_results["Human Expert"] = None
            except Exception as e:
                print(f"❌ Human Expert trial failed: {e}")
                level_results["Human Expert"] = None
        else:
            print("⚠️  Skipping Human Expert trial - keyboard module not available")
        
        # 2. Neural Only Trial
        if self.neural_available:
            print(f"\n--- Neural Only Trial ---")
            try:
                metrics = self.run_agent_trial(env, self.neural_model_path, "Neural")
                level_results["Neural Only"] = metrics
                self.print_trial_results("Neural Only", metrics)
            except KeyboardInterrupt:
                print(f"❌ Neural Only trial cancelled by user")
                level_results["Neural Only"] = None
            except Exception as e:
                print(f"❌ Neural Only trial failed: {e}")
                level_results["Neural Only"] = None
        
        # 3. Neurosymbolic Trial  
        if self.neurosymbolic_available:
            print(f"\n--- Neurosymbolic Trial ---")
            try:
                metrics = self.run_agent_trial(env, self.neurosymbolic_model_path, "Neurosymbolic")
                level_results["Neurosymbolic"] = metrics
                self.print_trial_results("Neurosymbolic", metrics)
            except KeyboardInterrupt:
                print(f"❌ Neurosymbolic trial cancelled by user")
                level_results["Neurosymbolic"] = None
            except Exception as e:
                print(f"❌ Neurosymbolic trial failed: {e}")
                level_results["Neurosymbolic"] = None
        
        # Store results
        self.results[level] = level_results
        
        # Print level summary
        self.print_level_summary(level, level_results)
        
        return level_results
    
    def print_trial_results(self, approach, metrics):
        """Print results for a single trial"""
        if metrics is None:
            print(f"❌ {approach}: Failed")
            return
        
        status = "✅ SUCCESS" if metrics['success'] else "❌ FAILED"
        failure_reason = ""
        if not metrics['success']:
            if metrics['collision']:
                failure_reason = " (Collision)"
            elif metrics['out_of_bounds']:
                failure_reason = " (Out of Bounds)"
            elif metrics['timeout']:
                failure_reason = " (Timeout)"
        
        print(f"{status}{failure_reason}")
        print(f"  Path Length: {metrics['path_length']:.2f}m")
        print(f"  Steps: {metrics['step_count']}")
        print(f"  Final Distance: {metrics['final_distance']:.2f}m")
        print(f"  Duration: {metrics['duration']:.1f}s")
        print(f"  Path Efficiency: {metrics['path_efficiency']:.2f}")
    
    def print_level_summary(self, level, results):
        """Print summary for a level"""
        print(f"\n📊 LEVEL {level} SUMMARY")
        print("-" * 30)
        
        for approach, metrics in results.items():
            if metrics:
                status = "SUCCESS" if metrics['success'] else "FAILED"
                print(f"{approach:15}: {status:7} | {metrics['path_length']:6.2f}m | {metrics['step_count']:4d} steps")
            else:
                print(f"{approach:15}: ERROR")
    
    def save_results_to_csv(self):
        """Save all results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['level', 'approach', 'success', 'path_length', 'step_count', 
                         'final_distance', 'duration', 'path_efficiency', 'collision', 
                         'out_of_bounds', 'timeout']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for level, level_results in self.results.items():
                for approach, metrics in level_results.items():
                    if metrics:
                        row = {
                            'level': level,
                            'approach': approach,
                            'success': metrics['success'],
                            'path_length': metrics['path_length'],
                            'step_count': metrics['step_count'],
                            'final_distance': metrics['final_distance'],
                            'duration': metrics['duration'],
                            'path_efficiency': metrics['path_efficiency'],
                            'collision': metrics['collision'],
                            'out_of_bounds': metrics['out_of_bounds'],
                            'timeout': metrics['timeout']
                        }
                        writer.writerow(row)
        
        print(f"📄 Results saved to {filename}")
    
    def run_manual_trial(self, env, max_steps=5000):
        """Run a manual control trial using the same environment as other approaches"""
        if not KEYBOARD_AVAILABLE:
            print("❌ Manual control not available - keyboard module not installed")
            return None
        
        # Ensure CONFIG uses the same start/goal positions as environment
        CONFIG['start_pos'] = env.start_pos.copy()
        CONFIG['goal_pos'] = env.goal_pos.copy()
        
        print("👤 Starting Human Expert trial with integrated manual control...")
        print(f"📍 Navigate from START {env.start_pos[:2]} to GOAL {env.goal_pos[:2]}")
        print(f"🚧 Level {env.level} with {len(env.obstacles)} obstacles")
        
        # Regenerate XML with correct start/goal markers
        EnvironmentGenerator.create_xml_with_obstacles(env.obstacles)
        env.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        env.data = mujoco.MjData(env.model)
        
        # Create manual controller in comparison mode
        control_system = ManualControlSystem(comparison_mode=True)
        control_system.controls_active = True
        
        # Initialize performance tracker
        tracker = PerformanceTracker()
        tracker.start_trial(env.start_pos)
        
        # Initialize path_history for trail visualization
        path_history = []
        
        # Reset UAV to start position
        env.reset_uav()
        
        print("🚀 Starting manual control simulation...")
        print(f"🟢 GREEN markers: START position {env.start_pos[:2]}")
        print(f"🔵 BLUE markers: GOAL position {env.goal_pos[:2]}")
        print("🎮 Controls: Arrow Keys (↑↓←→), SPACE (stop), R (reset), ESC (exit)")
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            step_count = 0
            control_dt = CONFIG.get('control_dt', 0.05)
            kp_pos = CONFIG.get('kp_pos', 1.5)
            
            start_time = time.time()
            
            try:
                while (viewer.is_running() and control_system.controls_active and 
                       not control_system.trial_complete and step_count < max_steps):
                    
                    # Get current UAV state
                    current_pos = env.data.qpos[:3].copy()
                    
                    # Update manual controls
                    manual_action = control_system.update_controls()
                    
                    # Handle special keys
                    control_system.handle_special_keys(env.data, env.model, CONFIG['start_pos'])
                    
                    # Height control
                    desired_height = CONFIG['uav_flight_height']
                    height_error = desired_height - current_pos[2]
                    vz_correction = kp_pos * height_error
                    
                    # Apply manual control actions
                    env.data.qvel[0] = manual_action[0]  # X-velocity
                    env.data.qvel[1] = manual_action[1]  # Y-velocity  
                    env.data.qvel[2] = vz_correction    # Z-velocity
                    
                    # Check goal reached
                    goal_distance = np.linalg.norm(current_pos - env.goal_pos)
                    if goal_distance < 0.2 and not control_system.goal_reached:
                        control_system.goal_reached = True
                        control_system.trial_complete = True
                        print(f"\n🎉 GOAL REACHED! Distance: {goal_distance:.3f}m")
                        break
                    
                    # Check for collision
                    has_collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(current_pos, env.obstacles)
                    if has_collision:
                        control_system.collision_occurred = True
                        control_system.trial_complete = True
                        print(f"\n💥 COLLISION with {obstacle_id}! Distance: {collision_dist:.3f}m")
                        break
                    
                    # Check boundaries
                    half_world = CONFIG['world_size'] / 2
                    if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                        current_pos[2] < 0.1 or current_pos[2] > 5.0):
                        control_system.collision_occurred = True
                        control_system.trial_complete = True
                        print(f"\n🚨 BOUNDARY VIOLATION!")
                        break
                    
                    # Step simulation
                    mujoco.mj_step(env.model, env.data)
                    
                    # Update path trail
                    if step_count % 5 == 0:
                        update_path_trail(env.model, env.data, current_pos, path_history)
                    
                    # Ensure visibility
                    ensure_uav_visibility(env.model)
                    ensure_markers_visibility(env.model)
                    
                    # Update performance tracking
                    tracker.update_step(current_pos, env.goal_pos)
                    
                    viewer.sync()
                    
                    step_count += 1
                    time.sleep(control_dt)
                    
            except KeyboardInterrupt:
                print("🛑 Manual control interrupted")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Finalize tracking
            tracker.end_trial(
                success=control_system.goal_reached,
                collision=control_system.collision_occurred,
                timeout=(step_count >= max_steps)
            )
            
            # Get final metrics
            metrics = tracker.get_metrics()
            metrics['duration'] = duration
            
            return metrics

def ensure_markers_visibility(model):
    """Ensure start/goal markers remain visible"""
    for i in range(model.ngeom):
        try:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if "start_marker" in geom_name:
                model.geom_rgba[i] = [0.0, 1.0, 0.0, 0.8]  # Green start marker
            elif "start_pole" in geom_name:
                model.geom_rgba[i] = [0.0, 1.0, 0.0, 1.0]  # Green start pole
            elif "goal_marker" in geom_name:
                model.geom_rgba[i] = [0.0, 0.0, 1.0, 0.8]  # Blue goal marker
            elif "goal_pole" in geom_name:
                model.geom_rgba[i] = [0.0, 0.0, 1.0, 1.0]  # Blue goal pole
        except:
            pass

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

def update_path_trail(model, data, current_pos, path_history=None):
    """Update the path trail visualization."""
    if path_history is None:
        # Use global path_history if available, otherwise create empty list
        global _global_path_history
        if '_global_path_history' not in globals():
            _global_path_history = []
        path_history = _global_path_history
        
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

def run_manual_only():
    """Run original manual control mode"""
    # Generate obstacles first
    print("🏗️ Generating complex environment with static obstacles...")
    obstacles = EnvironmentGenerator.generate_obstacles()

    # Set up safe start and goal positions
    max_attempts = 50
    safety_radius = 0.8

    # Generate dynamic start position from corners
    start_pos_safe = False
    start_safety_check_attempts = 0

    print("🏠 Generating random start position from corners...")
    while not start_pos_safe and start_safety_check_attempts < max_attempts:
        start_safety_check_attempts += 1
        CONFIG['start_pos'] = EnvironmentGenerator.get_random_corner_position()
        
        if check_position_safety(CONFIG['start_pos'], obstacles, safety_radius):
            start_pos_safe = True
            print(f"✅ Safe start position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
            break

    if not start_pos_safe:
        print("⚠️ Using default position [-3, -3, 1]")
        CONFIG['start_pos'] = np.array([-3.0, -3.0, 1.0])

    # Generate dynamic goal position
    goal_pos_safe = False
    goal_safety_check_attempts = 0

    print("🎯 Generating random goal position...")
    while not goal_pos_safe and goal_safety_check_attempts < max_attempts:
        goal_safety_check_attempts += 1
        CONFIG['goal_pos'] = EnvironmentGenerator.get_random_goal_position()
        
        start_goal_distance = np.linalg.norm(CONFIG['goal_pos'][:2] - CONFIG['start_pos'][:2])
        
        if (check_position_safety(CONFIG['goal_pos'], obstacles, safety_radius) and 
            start_goal_distance > 2.0):
            goal_pos_safe = True
            print(f"✅ Safe goal position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
            print(f"📏 Start-Goal distance: {start_goal_distance:.1f}m")
            break
        
    if not goal_pos_safe:
        print("⚠️ Using potentially unsafe goal position")

    # Create environment
    EnvironmentGenerator.create_xml_with_obstacles(obstacles)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"🚁 Manual UAV Navigation Environment Loaded!")
    print(f"🎯 Navigate from START (green) to GOAL (blue)")
    print(f"🏠 Start: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
    print(f"🏁 Goal: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")

    # Initialize manual control
    control_system = ManualControlSystem()
    path_history = []

    # Run simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = CONFIG['start_pos']
        data.qpos[3:7] = [1, 0, 0, 0]
        
        print("\n🎮 Manual Control Active! Use arrow keys to navigate!")
        
        step_count = 0
        mission_complete = False
        collision_occurred = False
        goal_reached = False
        
        try:
            while viewer.is_running() and control_system.controls_active and not mission_complete and not collision_occurred:
                current_pos = data.qpos[:3].copy()
                current_vel = data.qvel[:3].copy()
                
                manual_action = control_system.update_controls()
                control_system.handle_special_keys(data, model, CONFIG['start_pos'])
                
                # Height control
                height_error = CONFIG['uav_flight_height'] - data.qpos[2]
                vz_correction = CONFIG['kp_pos'] * height_error
                
                # Apply actions
                data.qvel[0] = manual_action[0]
                data.qvel[1] = manual_action[1]
                data.qvel[2] = vz_correction
                
                # Check goal
                goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
                if goal_distance < 0.1 and not goal_reached:
                    goal_reached = True
                    mission_complete = True
                    print(f"\n🎉 GOAL REACHED! Distance: {goal_distance:.3f}m")
                
                mujoco.mj_step(model, data)
                
                if step_count % 5 == 0:
                    update_path_trail(model, data, current_pos, path_history)
                    
                ensure_uav_visibility(model)
                viewer.sync()
                
                # Check boundaries
                half_world = CONFIG['world_size'] / 2
                if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                    current_pos[2] < 0.1 or current_pos[2] > 5.0):
                    collision_occurred = True
                    print(f"\n🚨 BOUNDARY VIOLATION!")
                    break
                
                # Check collision
                has_collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(data.qpos[:3], obstacles)
                if has_collision:
                    collision_occurred = True
                    print(f"\n💥 COLLISION with {obstacle_id}!")
                    break
                
                # Record data
                lidar_readings = get_lidar_readings(current_pos, obstacles)
                control_system.record_step(
                    position=current_pos,
                    velocity=current_vel,
                    goal_pos=CONFIG['goal_pos'],
                    lidar_readings=lidar_readings,
                    action=manual_action,
                    collision=has_collision,
                    goal_reached=goal_reached
                )
                
                if step_count % 100 == 0:
                    print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) | Goal dist: {goal_distance:.2f}m")
                
                step_count += 1
                time.sleep(CONFIG['control_dt'])
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted")
        
        # Results
        if mission_complete:
            print("\n🎉 MISSION SUCCESS!")
        elif collision_occurred:
            print("\n💥 MISSION FAILED - COLLISION")
        else:
            print("\n⏹️ SIMULATION TERMINATED")
        
        # Save data option
        buffer_info = control_system.get_buffer_info()
        if buffer_info['steps'] > 0:
            print(f"\n💾 Recorded {buffer_info['steps']} steps. Press 'S' to save or 'ESC' to exit...")
            while viewer.is_running():
                if keyboard.is_pressed('s'):
                    control_system.save_data()
                    break
                elif keyboard.is_pressed('esc'):
                    break
                time.sleep(0.1)

def main():
    """Main function with different execution modes"""
    print("🎮 UAV Performance Comparison System")
    print("=" * 50)
    
    if args.mode == 'manual':
        print("📋 Mode: Manual Control Only")
        if not KEYBOARD_AVAILABLE:
            print("❌ Keyboard module not available!")
            return
        run_manual_only()
        
    elif args.mode == 'comparison':
        print(f"📋 Mode: Single Level Comparison (Level {args.level})")
        comparison = PerformanceComparison()
        try:
            comparison.run_level_comparison(args.level)
            comparison.save_results_to_csv()
        except KeyboardInterrupt:
            print("\n👋 Comparison cancelled")
        
    elif args.mode == 'level':
        print(f"📋 Mode: Multi-Level Comparison (Levels {args.start_level}-{args.end_level})")
        comparison = PerformanceComparison()
        
        try:
            for level in range(args.start_level, args.end_level + 1):
                comparison.run_level_comparison(level)
                
                if level < args.end_level:
                    print(f"\n🔄 Level {level} completed!")
                    input("Press Enter to continue to next level, or Ctrl+C to stop...")
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Comparison stopped")
        
        # Final summary
        print("\n" + "="*80)
        print("🏆 FINAL PERFORMANCE SUMMARY")
        print("="*80)
        
        approaches = set()
        for level_results in comparison.results.values():
            approaches.update(level_results.keys())
        
        for approach in approaches:
            print(f"\n🤖 {approach}")
            successes = sum(1 for level, results in comparison.results.items() 
                          if approach in results and results[approach] and results[approach]['success'])
            total = len([level for level, results in comparison.results.items() 
                        if approach in results and results[approach]])
            if total > 0:
                success_rate = successes / total * 100
                print(f"Success Rate: {success_rate:.1f}% ({successes}/{total})")
        
        comparison.save_results_to_csv()

if __name__ == "__main__":
    main()