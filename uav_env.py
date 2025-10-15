import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import math
import random
import csv
import os
from datetime import datetime

# Configuration from uav_render.py, adapted for the environment
CONFIG = {
    'start_pos': np.array([-3.0, -3.0, 1.0]),  # Default start position (will be updated dynamically)
    'goal_pos': np.array([3.0, 3.0, 1.0]),     # Default goal position (will be updated dynamically)
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,  # Half of obstacle height
    'static_obstacles': 9,
    'min_obstacle_size': 0.05,
    'max_obstacle_size': 0.12,
    'collision_distance': 0.1,
    'control_dt': 0.05,
    'max_steps': 50000,  # Max steps per episode
    'boundary_penalty': -100,  # Penalty for going out of bounds
    'lidar_range': 2.9,  # LIDAR maximum detection range
    'lidar_num_rays': 16,  # Number of LIDAR rays (360 degrees)
    'step_reward': -0.01,    # Survival bonus per timestep
}

class EnvironmentGenerator:
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
    def check_position_safety(position, obstacles, safety_radius=0.8):
        """Check if a position is safe from obstacles"""
        for obs in obstacles:
            obs_pos = np.array(obs['pos'])
            distance = np.linalg.norm(position[:2] - obs_pos[:2])  # Only check horizontal distance
            
            if obs['shape'] == 'box':
                # For box obstacles, consider the size
                min_safe_distance = max(obs['size'][0], obs['size'][1]) + safety_radius
            elif obs['shape'] == 'cylinder':
                # For cylinder obstacles
                min_safe_distance = obs['size'][0] + safety_radius
            else:  # sphere
                min_safe_distance = obs['size'][0] + safety_radius
            
            if distance < min_safe_distance:
                return False
        return True

    @staticmethod
    def generate_curriculum_maps():
        """Generate 50 maps each for obstacle counts 1-10 with versatile start/goal positions"""
        curriculum_maps = {}
        
        for obstacle_count in range(1, 11):  # 1 to 10 obstacles
            maps = []
            for map_id in range(50):  # 50 maps per obstacle count (increased from 20)
                # Set random seed for reproducible map generation
                random.seed(obstacle_count * 1000 + map_id)
                
                # Generate obstacles first
                obstacles = EnvironmentGenerator.generate_obstacles(obstacle_count)
                
                # Generate safe start and goal positions
                max_attempts = 50
                safety_radius = 0.8
                
                # Get random start position from corners
                start_pos = None
                for attempt in range(max_attempts):
                    candidate_start = EnvironmentGenerator.get_random_corner_position()
                    if EnvironmentGenerator.check_position_safety(candidate_start, obstacles, safety_radius):
                        start_pos = candidate_start
                        break
                
                # If no safe corner found, use default corner and warn
                if start_pos is None:
                    start_pos = np.array([-3.0, -3.0, 1.0])
                    print(f"⚠️ Warning: Using default start position for map {obstacle_count}-{map_id}")
                
                # Get random goal position anywhere in map
                goal_pos = None
                for attempt in range(max_attempts):
                    candidate_goal = EnvironmentGenerator.get_random_goal_position()
                    # Ensure goal is not too close to start position
                    start_goal_distance = np.linalg.norm(candidate_goal[:2] - start_pos[:2])
                    if (EnvironmentGenerator.check_position_safety(candidate_goal, obstacles, safety_radius) and 
                        start_goal_distance > 2.0):  # At least 2m apart
                        goal_pos = candidate_goal
                        break
                
                # If no safe goal found, use opposite corner from start
                if goal_pos is None:
                    goal_pos = np.array([3.0, 3.0, 1.0])
                    print(f"⚠️ Warning: Using default goal position for map {obstacle_count}-{map_id}")
                
                maps.append({
                    'obstacles': obstacles,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'obstacle_count': obstacle_count,
                    'map_id': map_id
                })
            curriculum_maps[obstacle_count] = maps
        
        # Reset random seed
        random.seed()
        return curriculum_maps

    @staticmethod
    def create_xml_with_obstacles(obstacles):
        xml_template = f'''<mujoco model="uav_env">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="{CONFIG['world_size']/2} {CONFIG['world_size']/2} 0.1" material="grid"/>
    <light name="light1" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
    <body name="chassis" pos="{CONFIG['start_pos'][0]} {CONFIG['start_pos'][1]} {CONFIG['start_pos'][2]}">
      <joint type="free" name="root"/>
      <geom type="box" size="0.12 0.12 0.02" rgba="1.0 0.0 0.0 1.0" mass="0.8"/>
      <site name="motor1" pos="0.08 0.08 0" size="0.01"/>
      <site name="motor2" pos="-0.08 0.08 0" size="0.01"/>
      <site name="motor3" pos="0.08 -0.08 0" size="0.01"/>
      <site name="motor4" pos="-0.08 -0.08 0" size="0.01"/>
    </body>'''
        for obs in obstacles:
            if obs['shape'] == 'box':
                xml_template += f'''
    <geom name="{obs['id']}" type="box" size="{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
            elif obs['shape'] == 'cylinder':
                xml_template += f'''
    <geom name="{obs['id']}" type="cylinder" size="{obs['size'][0]} {obs['size'][1]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
        xml_template += '''
  </worldbody>
  <actuator>
    <motor name="m1" gear="0 0 1 0 0 0" site="motor1"/>
    <motor name="m2" gear="0 0 1 0 0 0" site="motor2"/>
    <motor name="m3" gear="0 0 1 0 0 0" site="motor3"/>
    <motor name="m4" gear="0 0 1 0 0 0" site="motor4"/>
  </actuator>
</mujoco>'''
        with open("environment.xml", 'w') as f:
            f.write(xml_template)
        return obstacles

class UAVEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, curriculum_learning=False, ns_cfg=None):
        super().__init__()
        self.render_mode = render_mode
        self.curriculum_learning = curriculum_learning
        # Neurosymbolic configuration (optional, with safe defaults)
        self.ns_cfg = ns_cfg if ns_cfg is not None else {
            'use_neurosymbolic': False,
            'lambda': 0.0,
            'warmup_steps': 20,
            'high_speed': 0.9,
            'blocked_strength': 0.1
        }
        
        # Initialize curriculum learning system
        if self.curriculum_learning:
            self.curriculum_maps = EnvironmentGenerator.generate_curriculum_maps()
            self.current_obstacle_level = 1
            self.current_map_pool = self.curriculum_maps[1]
            print(f"🎓 Curriculum Learning Initialized: Generated maps for obstacle levels 1-10")
            print(f"   - 50 maps per obstacle level (500 total maps)")
            print(f"   - Random start positions from corners")
            print(f"   - Random goal positions anywhere in map")
            print(f"   - Safe positioning with 0.8m obstacle clearance")
            print(f"   - Starting with obstacle level: {self.current_obstacle_level}")
        
        # Generate initial obstacles
        self.obstacles = EnvironmentGenerator.generate_obstacles() if not curriculum_learning else []
        self.load_curriculum_map() if curriculum_learning else EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Observation space: [pos(3), vel(3), goal_dist(3), lidar_readings(16), lidar_features(11)]
        # LIDAR features: min, mean, closest_dir(2), danger_level, clearances(4), goal_alignment(2)
        obs_dim = 3 + 3 + 3 + CONFIG['lidar_num_rays'] + 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: 3D velocity control (vx, vy, vz=0) - no Z-axis movement
        self.action_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.viewer = None
        self.step_count = 0
        
        # Initialize CSV logging for obstacle detection
        self.csv_log_path = "obstacle_detection_log.csv"
        self.init_csv_logging()
        
        # Store previous velocity for trajectory change calculation
        self.prev_velocity = np.zeros(3)
        
        # Episode counter for adaptive thresholds
        self.current_episode = 0
        
        # Goal stabilization tracking
        self.goal_reached = False
        self.goal_stabilization_steps = 0
        self.goal_hold_duration = 50  # Steps to hold at goal before episode ends (more steps for better stabilization)

        # Episode-local timestep counter (for neurosymbolic warmup logic)
        self._episode_timestep = 0

    def _get_velocity_limits(self):
        """Get adaptive velocity limits based on training progress (velocity curriculum)"""
        episode = self.current_episode
        
        # EVEN MORE CONSERVATIVE velocity curriculum (very slow start)
        # This helps agent learn proper direction before moving quickly
        if episode < 100:
            return 0.05, 0.2   # SUPER SLOW: Nearly stationary for initial learning
        elif episode < 300:
            return 0.05, 0.3   # Very slow
        elif episode < 500:
            return 0.08, 0.4   # Still slow but slightly faster
        elif episode < 1000:
            return 0.1, 0.5    # Min: 0.1 m/s, Max: 0.5 m/s (SLOW - learning phase)
        elif episode < 1500:
            return 0.12, 0.7   # Gradually increasing
        elif episode < 2000:
            return 0.15, 0.9   # Min: 0.15 m/s, Max: 0.9 m/s
        elif episode < 3000:
            return 0.2, 1.2    # Min: 0.2 m/s, Max: 1.2 m/s
        elif episode < 4000:
            return 0.25, 1.5   # Min: 0.25 m/s, Max: 1.5 m/s
        else:
            return 0.3, 2.0    # Min: 0.3 m/s, Max: 2.0 m/s (FULL SPEED)
    
    def step(self, action):
        # Track per-episode timestep
        self._episode_timestep += 1
        # Store previous velocity for trajectory change calculation
        self.prev_velocity = self.data.qvel[:3].copy()
        
        # Action is now 3D: [vx, vy, vz] but we ignore vz and keep constant height
        # Convert action to numpy array if it's a tensor
        if hasattr(action, 'numpy'):
            action = action.numpy()
        action = np.array(action).flatten()
        
        # Add goal-directed bias to action in early training
        # This helps the agent initially learn to move toward goal
        if self.current_episode < 200:  # Only in early training
            pos = self.data.qpos[:3]
            goal_vector = CONFIG['goal_pos'] - pos
            goal_direction = goal_vector / (np.linalg.norm(goal_vector) + 1e-8)  # Normalized goal direction
            
            # Blend action with goal direction (50% agent action, 50% goal bias)
            bias_strength = max(0, (200 - self.current_episode) / 200)  # Fade out bias over first 200 episodes
            action[:2] = (1 - bias_strength * 0.5) * action[:2] + (bias_strength * 0.5) * goal_direction[:2]
        
        # VELOCITY CURRICULUM: Get current velocity limits
        min_vel, max_vel = self._get_velocity_limits()
        
        # Apply velocity control with GRADUAL ACCELERATION
        # Instead of directly setting velocity from action, we gradually adjust it
        current_vel = self.data.qvel[:2].copy()
        target_vel = action[:2]  # Desired velocity from action
        
        # ANTI-WESTWARD BIAS: Special case for western boundary problems
        # If agent is trying to move west (negative X), reduce that component significantly
        # This helps prevent the boundary issues we're seeing
        if target_vel[0] < 0 and self.current_episode < 500:
            target_vel[0] *= 0.5  # Reduce westward velocity by 50%
        
        # Limit target velocity magnitude
        target_speed = np.linalg.norm(target_vel)
        if target_speed > max_vel:
            target_vel = (target_vel / target_speed) * max_vel
        elif target_speed < min_vel and target_speed > 0:
            target_vel = (target_vel / target_speed) * min_vel
        
        # GRADUAL ACCELERATION: Smoothly transition from current to target velocity
        acceleration_rate = 0.2  # REDUCED: How quickly to change velocity (0-1)
        new_vel = current_vel + acceleration_rate * (target_vel - current_vel)
        
        # Check if UAV is at goal position BEFORE applying velocity
        pos = self.data.qpos[:3]
        goal_dist = np.linalg.norm(CONFIG['goal_pos'] - pos)
        
        # GOAL STABILIZATION: If UAV reaches goal, apply strong braking to keep it there
        if goal_dist < 0.5:  # Same threshold as reward function
            if not self.goal_reached:
                print(f"🎯 GOAL REACHED! Stabilizing UAV at goal position...")
                self.goal_reached = True
                self.goal_stabilization_steps = 0
            
            # Apply VERY strong braking forces to LOCK UAV at goal
            goal_vector = CONFIG['goal_pos'] - pos
            
            # Check if we're very close to goal center (within 0.1m)
            if goal_dist < 0.1:
                # LOCK MODE: Virtually stop all movement
                # Apply extremely strong velocity damping (99% reduction)
                stabilization_vel = -self.data.qvel[:2] * 0.99
                
                # Add tiny position correction to keep centered
                stabilization_vel += goal_vector[:2] * 5.0
            else:
                # APPROACH MODE: Strong position correction with damping
                position_correction = goal_vector[:2] * 3.0  # Very strong pull toward goal
                
                # Strong velocity damping to eliminate momentum
                velocity_damping = -self.data.qvel[:2] * 0.9  # Reduce current velocity by 90%
                
                # Combine corrections with priority on stopping motion
                stabilization_vel = position_correction + velocity_damping
            
            # Limit stabilization velocity to prevent overshooting
            stabilization_speed = np.linalg.norm(stabilization_vel)
            max_stabilization_speed = 0.2 if goal_dist < 0.1 else 0.4
            if stabilization_speed > max_stabilization_speed:
                stabilization_vel = (stabilization_vel / stabilization_speed) * max_stabilization_speed
            
            # Apply stabilization velocity instead of action-based velocity
            self.data.qvel[0] = float(stabilization_vel[0])
            self.data.qvel[1] = float(stabilization_vel[1])
            self.data.qvel[2] = 0.0
            
            self.goal_stabilization_steps += 1
        else:
            # Normal velocity control when not at goal
            self.data.qvel[0] = float(new_vel[0])  # X velocity
            self.data.qvel[1] = float(new_vel[1])  # Y velocity
            self.data.qvel[2] = 0.0                # Z velocity = 0 (no vertical movement)
        
        # Maintain constant height
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Enforce velocity constraints (with curriculum limits)
        vel_xy = self.data.qvel[:2]
        speed_xy = np.linalg.norm(vel_xy)
        
        if speed_xy < min_vel and speed_xy > 0:
            self.data.qvel[:2] = (vel_xy / speed_xy) * min_vel
        elif speed_xy > max_vel:
            self.data.qvel[:2] = (vel_xy / speed_xy) * max_vel
        
        # Always ensure Z position stays constant
        self.data.qpos[2] = CONFIG['uav_flight_height']
        self.data.qvel[2] = 0.0
        
        obs = self._get_obs()
        reward, termination_info = self._get_reward_and_termination_info(obs)
        terminated = termination_info['terminated']
        truncated = self.step_count >= CONFIG['max_steps']
        
        # Store termination info for logging
        self.last_termination_info = termination_info
        if truncated and not terminated:
            self.last_termination_info['termination_reason'] = 'max_steps_reached'
            self.last_termination_info['terminated'] = True
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._episode_timestep = 0
        
        # Set dynamic goal position (randomly select from three available corners)
        CONFIG['goal_pos'] = self._get_random_goal_position()
        
        # Reset UAV position and state with constant height
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']  # Ensure constant height
        self.data.qvel[:] = 0
        
        # Reset velocity tracking
        self.prev_velocity = np.zeros(3)
        
        # Initialize termination info
        self.last_termination_info = {
            'terminated': False,
            'termination_reason': 'none',
            'final_position': None,
            'goal_distance': None,
            'collision_detected': False,
            'out_of_bounds': False
        }
        
        # Reset goal stabilization tracking
        self.goal_reached = False
        self.goal_stabilization_steps = 0
        
        # Regenerate obstacles for each episode
        if self.curriculum_learning:
            self.load_curriculum_map()
        else:
            self.obstacles = EnvironmentGenerator.generate_obstacles()
            EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Set initial position again after model reload
        self.data.qpos[:3] = CONFIG['start_pos']
        
        # INITIALIZE WITH SMALL POSITIVE X VELOCITY to avoid early westward movement
        if self.current_episode < 300:  # Only for early training
            # Small initial eastward velocity (will help avoid immediate west boundary issues)
            self.data.qvel[0] = 0.05  # Small positive X velocity (eastward)
            self.data.qvel[1] = 0.0   # No Y velocity
            self.data.qvel[2] = 0.0   # No Z velocity
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        # Initialize previous goal distance
        self.prev_goal_dist = np.linalg.norm(CONFIG['goal_pos'] - self.data.qpos[:3])
        
        return self._get_obs(), {}

    # =====================
    # Neurosymbolic helpers
    # =====================
    def get_goal_vector(self):
        """Return unit vector from UAV to goal (2D XY) and distance."""
        pos = self.data.qpos[:3]
        vec = CONFIG['goal_pos'] - pos
        dist = float(np.linalg.norm(vec[:2]) + 1e-8)
        dir_xy = vec[:2] / dist
        return dir_xy, dist

    def has_line_of_sight_to_goal(self):
        """Approximate LOS by raycasting from UAV to goal against obstacles in XY plane."""
        pos = self.data.qpos[:3]
        goal_vec = CONFIG['goal_pos'] - pos
        goal_dist = float(np.linalg.norm(goal_vec[:2]))
        if goal_dist <= 1e-6:
            return True
        ray_dir = np.array([goal_vec[0], goal_vec[1], 0.0]) / (goal_dist + 1e-8)
        min_obs_dist = CONFIG['lidar_range']
        for obs in self.obstacles:
            obs_dist = self._ray_obstacle_intersection(pos, ray_dir, obs)
            if obs_dist < min_obs_dist:
                min_obs_dist = obs_dist
        # LOS if closest obstacle along ray is farther than goal distance
        return min_obs_dist >= goal_dist - 1e-6

    def symbolic_action(self, t_step=None):
        """Compute a simple goal-directed action in env action space (vx, vy, vz=0)."""
        # Read cfg with fallbacks
        warmup_steps = int(self.ns_cfg.get('warmup_steps', 20))
        high_speed = float(self.ns_cfg.get('high_speed', 0.9))
        blocked_strength = float(self.ns_cfg.get('blocked_strength', 0.1))

        if t_step is None:
            t_step = self._episode_timestep

        dir_xy, _ = self.get_goal_vector()

        if t_step < warmup_steps:
            speed = min(0.3, high_speed)  # moderate speed at start
        elif self.has_line_of_sight_to_goal():
            speed = min(high_speed, 1.0)  # cap to action_space.high
        else:
            speed = max(0.0, min(blocked_strength, 1.0))  # gentle nudge when blocked

        vx = float(speed * dir_xy[0])
        vy = float(speed * dir_xy[1])
        vz = 0.0
        a = np.array([vx, vy, vz], dtype=np.float32)
        # Clip to action bounds
        low = np.broadcast_to(self.action_space.low, a.shape)
        high = np.broadcast_to(self.action_space.high, a.shape)
        return np.clip(a, low, high)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
    
    def _get_random_goal_position(self):
        """Select a random goal position from the three available corners (excluding start position)"""
        # Define the four corners of the world with LARGER SAFETY MARGIN
        half_world = CONFIG['world_size'] / 2
        safety_margin = 1.0  # INCREASED: Keep 1.0m away from boundary
        
        corners = [
            np.array([half_world - safety_margin, half_world - safety_margin, CONFIG['uav_flight_height']]),    # Top-right [3, 3]
            np.array([half_world - safety_margin, -half_world + safety_margin, CONFIG['uav_flight_height']]),   # Bottom-right [3, -3]
            np.array([-half_world + safety_margin, half_world - safety_margin, CONFIG['uav_flight_height']])    # Top-left [-3, 3]
        ]
        # Start position is bottom-left: [-3, -3, 1.0]
        # So we exclude it and randomly select from the other three corners
        return random.choice(corners)
    
    def init_csv_logging(self):
        """Initialize CSV file for obstacle detection logging"""
        # Create headers for the CSV file
        headers = [
            'timestamp', 'episode', 'step', 'curriculum_level', 'map_id', 'obstacle_count',
            'start_x', 'start_y', 'start_z', 'goal_x', 'goal_y', 'goal_z',
            'uav_x', 'uav_y', 'uav_z', 'obstacle_x', 'obstacle_y', 'obstacle_z', 
            'obstacle_type', 'obstacle_id', 'detection_distance', 'detection_angle', 
            'prev_velocity_x', 'prev_velocity_y', 'new_velocity_x', 'new_velocity_y', 
            'trajectory_change_angle'
        ]
        
        # Always create/overwrite the CSV file to clear previous logs
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
        # Initialize curriculum episode logging
        self.curriculum_log_path = "curriculum_learning_log.csv"
        curriculum_headers = [
            'timestamp', 'episode', 'curriculum_level', 'episode_in_level', 'map_id',
            'obstacle_count', 'start_x', 'start_y', 'start_z', 'goal_x', 'goal_y', 'goal_z',
            'episode_reward', 'episode_length', 'termination_reason',
            'goal_reached', 'collision_detected', 'out_of_bounds', 'final_position_x',
            'final_position_y', 'final_position_z', 'goal_distance', 'final_velocity'
        ]
        
        # Always create/overwrite the curriculum log file
        with open(self.curriculum_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(curriculum_headers)
                
    def log_obstacle_detection(self, uav_pos, obstacle_pos, obstacle_type, obstacle_id, 
                             detection_distance, detection_angle, prev_vel, new_vel):
        """Log obstacle detection event to CSV file"""
        # Calculate trajectory change angle
        prev_vel_2d = prev_vel[:2]
        new_vel_2d = new_vel[:2]
        
        trajectory_change_angle = 0.0
        if np.linalg.norm(prev_vel_2d) > 0.01 and np.linalg.norm(new_vel_2d) > 0.01:
            # Normalize vectors
            prev_vel_norm = prev_vel_2d / np.linalg.norm(prev_vel_2d)
            new_vel_norm = new_vel_2d / np.linalg.norm(new_vel_2d)
            
            # Calculate angle between vectors
            dot_product = np.clip(np.dot(prev_vel_norm, new_vel_norm), -1.0, 1.0)
            trajectory_change_angle = np.degrees(np.arccos(dot_product))
        
        # Write to CSV with curriculum information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        curriculum_info = self.get_curriculum_info() if self.curriculum_learning else None
        
        row = [
            timestamp, getattr(self, 'current_episode', 0), self.step_count,
            curriculum_info['current_level'] if curriculum_info else 0,
            curriculum_info['current_map_info']['map_id'] if curriculum_info and curriculum_info['current_map_info'] else 0,
            curriculum_info['current_map_info']['obstacle_count'] if curriculum_info and curriculum_info['current_map_info'] else 0,
            CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['start_pos'][2],
            CONFIG['goal_pos'][0], CONFIG['goal_pos'][1], CONFIG['goal_pos'][2],
            uav_pos[0], uav_pos[1], uav_pos[2],
            obstacle_pos[0], obstacle_pos[1], obstacle_pos[2],
            obstacle_type, obstacle_id, detection_distance, detection_angle,
            prev_vel[0], prev_vel[1], new_vel[0], new_vel[1], trajectory_change_angle
        ]
        
        with open(self.csv_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    
    def log_curriculum_episode(self, episode_num, episode_in_level, episode_reward, 
                             episode_length, termination_info):
        """Log curriculum learning episode data"""
        if not self.curriculum_learning:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        curriculum_info = self.get_curriculum_info()
        map_info = curriculum_info['current_map_info'] if curriculum_info else {}
        
        # Extract termination information
        final_pos = termination_info.get('final_position', [0, 0, 0])
        goal_reached = termination_info.get('termination_reason') == 'goal_reached'
        collision = termination_info.get('collision_detected', False)
        out_of_bounds = termination_info.get('out_of_bounds', False)
        
        row = [
            timestamp, episode_num, 
            curriculum_info['current_level'] if curriculum_info else 0,
            episode_in_level,
            map_info.get('map_id', 0),
            map_info.get('obstacle_count', 0),
            CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['start_pos'][2],
            CONFIG['goal_pos'][0], CONFIG['goal_pos'][1], CONFIG['goal_pos'][2],
            episode_reward, episode_length,
            termination_info.get('termination_reason', 'unknown'),
            goal_reached, collision, out_of_bounds,
            final_pos[0] if len(final_pos) > 0 else 0,
            final_pos[1] if len(final_pos) > 1 else 0,
            final_pos[2] if len(final_pos) > 2 else 0,
            termination_info.get('goal_distance', 0),
            termination_info.get('final_velocity', 0)
        ]
        
        with open(self.curriculum_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    
    def set_curriculum_level(self, obstacle_level):
        """Set the current curriculum level (1-10 obstacles)"""
        if not self.curriculum_learning:
            print("⚠️  Curriculum learning not enabled for this environment")
            return
            
        if obstacle_level < 1 or obstacle_level > 10:
            print(f"⚠️  Invalid obstacle level: {obstacle_level}. Must be between 1-10")
            return
            
        self.current_obstacle_level = obstacle_level
        self.current_map_pool = self.curriculum_maps[obstacle_level]
        print(f"🎓 Curriculum Level Updated: Now using {obstacle_level} obstacle maps")
    
    def load_curriculum_map(self):
        """Load a random map from the current curriculum level"""
        if not self.curriculum_learning:
            return
            
        # Select random map from current level
        selected_map = random.choice(self.current_map_pool)
        self.obstacles = selected_map['obstacles']
        
        # Update CONFIG with map-specific start and goal positions
        CONFIG['start_pos'] = selected_map['start_pos'].copy()
        CONFIG['goal_pos'] = selected_map['goal_pos'].copy()
        
        self.current_map_info = {
            'obstacle_count': selected_map['obstacle_count'],
            'map_id': selected_map['map_id'],
            'start_pos': selected_map['start_pos'].copy(),
            'goal_pos': selected_map['goal_pos'].copy()
        }
        
        # Create XML with selected obstacles and positions
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
    
    def get_curriculum_info(self):
        """Get current curriculum information"""
        if not self.curriculum_learning:
            return None
            
        return {
            'current_level': self.current_obstacle_level,
            'current_map_info': getattr(self, 'current_map_info', None),
            'total_levels': 10
        }

    def _get_obs(self):
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        goal_dist = CONFIG['goal_pos'] - pos
        
        # Get LIDAR readings (normalized to [0, 1])
        lidar_readings = self._get_lidar_readings(pos)
        
        # === LIDAR FEATURE ENGINEERING ===
        # Extract meaningful features from raw LIDAR data
        
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
        
        return np.concatenate([pos, vel, goal_dist, lidar_readings, lidar_features])

    def _get_lidar_readings(self, pos):
        """Generate LIDAR readings in 360 degrees around the UAV (normalized to [0,1])"""
        lidar_readings = []
        
        # Define obstacle detection threshold (distance at which we consider obstacle "detected")
        obstacle_detection_threshold = 1.5  # meters
        
        for i in range(CONFIG['lidar_num_rays']):
            # Calculate ray direction (360 degrees divided by number of rays)
            angle = (2 * math.pi * i) / CONFIG['lidar_num_rays']
            ray_dir = np.array([math.cos(angle), math.sin(angle), 0])
            
            # Cast ray and find closest obstacle or boundary
            min_distance = CONFIG['lidar_range']
            closest_obstacle = None
            
            # Check boundary intersection
            boundary_dist = self._ray_boundary_intersection(pos, ray_dir)
            if boundary_dist < min_distance:
                min_distance = boundary_dist
            
            # Check obstacle intersections
            for obs in self.obstacles:
                obs_dist = self._ray_obstacle_intersection(pos, ray_dir, obs)
                if obs_dist < min_distance:
                    min_distance = obs_dist
                    closest_obstacle = obs
            
            # Log obstacle detection if obstacle is within detection threshold
            if (closest_obstacle is not None and 
                min_distance < obstacle_detection_threshold and 
                min_distance < CONFIG['lidar_range']):
                
                # Get current velocity from the data
                current_vel = self.data.qvel[:3].copy()
                
                # Log the obstacle detection
                self.log_obstacle_detection(
                    uav_pos=pos,
                    obstacle_pos=np.array(closest_obstacle['pos']),
                    obstacle_type=closest_obstacle['shape'],
                    obstacle_id=closest_obstacle['id'],
                    detection_distance=min_distance,
                    detection_angle=np.degrees(angle),
                    prev_vel=self.prev_velocity.copy(),
                    new_vel=current_vel
                )
            
            # Normalize LIDAR reading to [0, 1] range
            normalized_distance = min_distance / CONFIG['lidar_range']
            lidar_readings.append(normalized_distance)
        
        return np.array(lidar_readings)

    def _ray_boundary_intersection(self, pos, ray_dir):
        """Calculate intersection of ray with world boundaries"""
        half_world = CONFIG['world_size'] / 2
        min_dist = CONFIG['lidar_range']
        
        # Check intersection with each boundary
        boundaries = [
            (half_world, np.array([1, 0, 0])),   # +X boundary
            (-half_world, np.array([-1, 0, 0])), # -X boundary
            (half_world, np.array([0, 1, 0])),   # +Y boundary
            (-half_world, np.array([0, -1, 0]))  # -Y boundary
        ]
        
        for boundary_pos, boundary_normal in boundaries:
            # Ray-plane intersection
            denominator = np.dot(ray_dir, boundary_normal)
            if abs(denominator) > 1e-6:  # Ray not parallel to boundary
                if boundary_normal[0] != 0:  # X boundary
                    t = (boundary_pos - pos[0]) / ray_dir[0]
                else:  # Y boundary
                    t = (boundary_pos - pos[1]) / ray_dir[1]
                
                if t > 0:  # Ray goes forward
                    intersection = pos + t * ray_dir
                    # Check if intersection is within boundary limits
                    if (-half_world <= intersection[0] <= half_world and 
                        -half_world <= intersection[1] <= half_world):
                        min_dist = min(min_dist, t)
        
        return min_dist

    def _ray_obstacle_intersection(self, pos, ray_dir, obstacle):
        """Calculate intersection of ray with obstacle"""
        obs_pos = np.array(obstacle['pos'])
        min_dist = CONFIG['lidar_range']
        
        if obstacle['shape'] == 'box':
            # Simple box intersection (approximate)
            # Calculate distance to box center and subtract box size
            to_obs = obs_pos - pos
            proj_length = np.dot(to_obs, ray_dir)
            
            if proj_length > 0:  # Obstacle is in ray direction
                closest_point = pos + proj_length * ray_dir
                # Check if ray passes near the obstacle
                lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])  # Only X,Y
                
                if lateral_dist < max(obstacle['size'][0], obstacle['size'][1]):
                    # Approximate distance to obstacle surface
                    surface_dist = max(0, proj_length - max(obstacle['size'][0], obstacle['size'][1]))
                    min_dist = min(min_dist, surface_dist)
        
        elif obstacle['shape'] == 'cylinder':
            # Ray-cylinder intersection (simplified)
            to_obs = obs_pos - pos
            proj_length = np.dot(to_obs, ray_dir)
            
            if proj_length > 0:  # Obstacle is in ray direction
                closest_point = pos + proj_length * ray_dir
                lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])  # Only X,Y
                
                if lateral_dist < obstacle['size'][0]:  # Within cylinder radius
                    surface_dist = max(0, proj_length - obstacle['size'][0])
                    min_dist = min(min_dist, surface_dist)
        
        return min_dist

    def _get_reward_and_termination_info(self, obs):
        pos = obs[:3]
        vel = obs[3:6]
        goal_dist = np.linalg.norm(CONFIG['goal_pos'] - pos)
        
        # Extract LIDAR readings (normalized, indices 9:25)
        lidar_readings = obs[9:25]
        min_obstacle_dist_norm = np.min(lidar_readings)
        min_obstacle_dist = min_obstacle_dist_norm * CONFIG['lidar_range']  # Convert back to meters
        
        # Initialize termination info
        termination_info = {
            'terminated': False,
            'termination_reason': 'none',
            'final_position': pos.copy(),
            'goal_distance': goal_dist,
            'collision_detected': False,
            'out_of_bounds': False,
            'episode_length': self.step_count,
            'final_velocity': np.linalg.norm(vel[:2])  # Only horizontal velocity
        }
        
        # Initialize reward
        reward = 0.0
        
        # === TERMINAL REWARDS (Strong Signals) ===
        
        # Check if UAV is out of bounds
        if self._check_out_of_bounds(pos):
            reward = -100  # Strong penalty
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'out_of_bounds'
            termination_info['out_of_bounds'] = True
            return reward, termination_info
        
        # Check for collision with obstacles
        if self._check_collision(pos):
            reward = -100  # Strong penalty
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'collision'
            termination_info['collision_detected'] = True
            return reward, termination_info
            
        # Check if goal is reached and stabilized
        if goal_dist < 0.5:
            if not self.goal_reached:
                # First time reaching goal - give large reward but don't terminate yet
                reward = 500  # Large reward for reaching goal
                self.goal_reached = True
                self.goal_stabilization_steps = 0
            else:
                # UAV is stabilizing at goal - give smaller continuous rewards
                reward = 50  # Reward for staying at goal
                
                # Check if UAV has been stable at goal long enough
                if self.goal_stabilization_steps >= self.goal_hold_duration:
                    # Final bonus for successful goal stabilization
                    reward += 500  # Bonus for successful stabilization
                    termination_info['terminated'] = True
                    termination_info['termination_reason'] = 'goal_reached_and_stabilized'
                    print(f"✅ GOAL SUCCESSFULLY REACHED AND STABILIZED! ({self.goal_stabilization_steps} steps)")
                    return reward, termination_info
                elif goal_dist > 1.0:  # If UAV moves too far from goal during stabilization
                    reward = -200  # Penalty for leaving goal area
                    self.goal_reached = False  # Reset goal reached status
                    self.goal_stabilization_steps = 0
                    print(f"⚠️  UAV left goal area during stabilization. Resetting...")
        else:
            # Reset goal status if UAV is not near goal
            if self.goal_reached and goal_dist > 1.0:
                self.goal_reached = False
                self.goal_stabilization_steps = 0
        
        # === STEP REWARDS (Simplified) ===
        
        # 1. Progress reward (primary driving force)
        progress = self.prev_goal_dist - goal_dist
        reward = 10.0 * progress  # Scaled up for significance
        
        # 2. Add directional bias to encourage eastward movement if goal is eastward
        goal_vector = CONFIG['goal_pos'] - pos
        if goal_vector[0] > 0:  # If goal is to the east
            # Add a strong bias for eastward movement
            reward += 2.0 * max(0, self.data.qvel[0])  # Reward positive x velocity
            
            # Add stronger penalty for westward movement (going the wrong way)
            if self.data.qvel[0] < 0:  # If moving westward
                reward -= 5.0 * abs(self.data.qvel[0])  # Penalize negative x velocity
        
        # 3. Boundary awareness - add extra penalty when getting close to boundaries
        half_world = CONFIG['world_size'] / 2
        x_distance_to_west_boundary = abs(pos[0] + half_world)  # Distance to west boundary
        if x_distance_to_west_boundary < 1.0:  # Getting close to western boundary
            reward -= (1.0 - x_distance_to_west_boundary) * 10.0  # Stronger penalty closer to boundary
        
        # 4. LIDAR-based proximity penalties (collision avoidance)
        if min_obstacle_dist < 0.3:
            reward -= 5.0  # Very dangerous - strong penalty
        elif min_obstacle_dist < 0.5:
            reward -= 2.0  # Dangerous - moderate penalty
        elif min_obstacle_dist < 1.0:
            reward -= 0.5  # Caution zone - mild penalty
        
        # 3. Safe navigation bonus (reward for good behavior)
        if min_obstacle_dist > 1.5 and progress > 0:
            reward += 0.5  # Bonus for maintaining safe distance while progressing
        
        # Update previous goal distance for next step
        self.prev_goal_dist = goal_dist
        
        return reward, termination_info

    def _check_out_of_bounds(self, pos):
        """Check if UAV position is outside the world boundaries"""
        half_world = CONFIG['world_size'] / 2
        boundary_tolerance = 0.05  # REDUCED: Stricter tolerance for numerical errors
        
        # Create a smaller effective boundary to prevent getting too close to edges
        effective_boundary = half_world - 0.1
        
        # Apply strictest check to western boundary (x negative) where we have problems
        western_boundary = -half_world + 0.2  # Even stricter western boundary
        
        return (pos[0] < western_boundary or  # Stricter western boundary
                pos[0] > effective_boundary or 
                abs(pos[1]) > effective_boundary or 
                pos[2] < 0.5 or   # Too low (but shouldn't happen with constant height)
                pos[2] > 1.5)     # Too high (but shouldn't happen with constant height)

    def _get_collision_threshold(self):
        """Get adaptive collision threshold based on training progress (curriculum)"""
        episode = self.current_episode
        
        if episode < 500:
            return 0.15  # Lenient early training
        elif episode < 1500:
            return 0.13  # Moderate
        elif episode < 3000:
            return 0.11  # Getting stricter
        else:
            return 0.10  # Final strict threshold

    def _check_collision(self, uav_pos):
        collision_threshold = self._get_collision_threshold()
        
        for obs in self.obstacles:
            obs_pos = np.array(obs['pos'])
            if obs['shape'] == 'box':
                dx = max(abs(uav_pos[0] - obs_pos[0]) - obs['size'][0], 0)
                dy = max(abs(uav_pos[1] - obs_pos[1]) - obs['size'][1], 0)
                dz = max(abs(uav_pos[2] - obs_pos[2]) - obs['size'][2], 0)
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            elif obs['shape'] == 'cylinder':
                horizontal_dist = math.sqrt((uav_pos[0]-obs_pos[0])**2 + (uav_pos[1]-obs_pos[1])**2)
                vertical_dist = abs(uav_pos[2] - obs_pos[2])
                distance = max(horizontal_dist - obs['size'][0], vertical_dist - obs['size'][1])
            
            if distance < collision_threshold:
                return True
        return False

if __name__ == '__main__':
    # Example of using the environment
    env = UAVEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
    env.close()
