import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import math
import random

# Configuration from uav_render.py, adapted for the environment
CONFIG = {
    'start_pos': np.array([-4.0, -4.0, 1.0]),
    'goal_pos': np.array([4.0, 4.0, 1.0]),
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,  # Half of obstacle height
    'static_obstacles': 8,
    'min_obstacle_size': 0.2,
    'max_obstacle_size': 0.6,
    'collision_distance': 0.15,  # Slightly reduced for more forgiving collisions
    'control_dt': 0.05,
    'max_steps': 50000,  # Max steps per episode
    'boundary_penalty': -10,  # Penalty for going out of bounds
    'lidar_range': 3.0,  # LIDAR maximum detection range
    'lidar_num_rays': 16,  # Number of LIDAR rays (360 degrees)
    'step_reward': 0.01,    # Survival bonus per timestep
}

class EnvironmentGenerator:
    @staticmethod
    def generate_obstacles():
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        # Randomize number of obstacles for each episode (between 4 and 12)
        num_obstacles = random.randint(4, 12)
        
        for i in range(num_obstacles):
            # Generate completely random positions within the world boundaries
            # Keep obstacles away from start and goal positions
            x = random.uniform(-half_world + 1, half_world - 1)
            y = random.uniform(-half_world + 1, half_world - 1)
            
            # Ensure obstacles don't spawn too close to start or goal positions
            start_pos = CONFIG['start_pos'][:2]  # Only X,Y
            goal_pos = CONFIG['goal_pos'][:2]    # Only X,Y
            min_distance_from_points = 1.5
            
            # Regenerate position if too close to start or goal
            while (np.linalg.norm([x, y] - start_pos) < min_distance_from_points or 
                   np.linalg.norm([x, y] - goal_pos) < min_distance_from_points):
                x = random.uniform(-half_world + 1, half_world - 1)
                y = random.uniform(-half_world + 1, half_world - 1)
            
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
    def create_xml_content(obstacles):
        """Create XML content string for given obstacles (optimized version)"""
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
        return xml_template

    @staticmethod
    def create_xml_with_obstacles(obstacles):
        """Legacy method for backward compatibility"""
        xml_content = EnvironmentGenerator.create_xml_content(obstacles)
        with open("environment.xml", 'w') as f:
            f.write(xml_content)
        return obstacles

class UAVEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Pre-generate multiple environment configurations for fast switching
        self.num_env_configs = 20  # Number of pre-generated environments
        self.env_configs = []
        self.current_env_idx = 0
        
        print(f"🏗️ Pre-generating {self.num_env_configs} environment configurations...")
        
        # Generate multiple obstacle configurations and store them
        for i in range(self.num_env_configs):
            obstacles = EnvironmentGenerator.generate_obstacles()
            xml_content = EnvironmentGenerator.create_xml_content(obstacles)
            self.env_configs.append({
                'obstacles': obstacles,
                'xml_content': xml_content,
                'num_obstacles': len(obstacles)
            })
        
        print(f"✅ Environment configurations ready!")
        
        # Initialize with first configuration
        self.obstacles = self.env_configs[0]['obstacles']
        
        # Create initial model from first configuration
        with open("environment.xml", 'w') as f:
            f.write(self.env_configs[0]['xml_content'])
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Observation space: [pos(3), vel(3), goal_dist(3), lidar_readings(16)]
        obs_dim = 3 + 3 + 3 + CONFIG['lidar_num_rays']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: 3D velocity control (vx, vy, vz=0) - no Z-axis movement
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        
        self.viewer = None
        self.step_count = 0
        self.episode_count = 0  # Track episodes for periodic regeneration

    def _regenerate_env_pool(self):
        """Regenerate environment pool periodically for more variety"""
        print(f"🔄 Regenerating environment pool for fresh variety...")
        self.env_configs = []
        for i in range(self.num_env_configs):
            obstacles = EnvironmentGenerator.generate_obstacles()
            xml_content = EnvironmentGenerator.create_xml_content(obstacles)
            self.env_configs.append({
                'obstacles': obstacles,
                'xml_content': xml_content,
                'num_obstacles': len(obstacles)
            })
        print(f"✅ Environment pool regenerated!")

    def step(self, action):
        # Action is now 3D: [vx, vy, vz] but we ignore vz and keep constant height
        # Convert action to numpy array if it's a tensor
        if hasattr(action, 'numpy'):
            action = action.numpy()
        action = np.array(action).flatten()
        
        # Clip actions to reasonable velocity limits
        action = np.clip(action, -2.0, 2.0)
        
        # Apply velocity control with constant height
        self.data.qvel[0] = float(action[0])  # X velocity
        self.data.qvel[1] = float(action[1])  # Y velocity  
        self.data.qvel[2] = 0.0               # Z velocity = 0 (no vertical movement)
        
        # Maintain constant height
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        # Store current position before step
        prev_pos = self.data.qpos[:3].copy()
        
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Enforce boundary constraints by clamping position
        half_world = CONFIG['world_size'] / 2
        boundary_margin = 0.1  # Small margin to prevent exact boundary touching
        
        # Clamp X and Y positions to stay within bounds
        self.data.qpos[0] = np.clip(self.data.qpos[0], -half_world + boundary_margin, half_world - boundary_margin)
        self.data.qpos[1] = np.clip(self.data.qpos[1], -half_world + boundary_margin, half_world - boundary_margin)
        
        # If position was clamped, also reduce velocity in that direction
        if abs(self.data.qpos[0]) >= half_world - boundary_margin:
            self.data.qvel[0] = 0.0  # Stop X velocity when hitting X boundary
        if abs(self.data.qpos[1]) >= half_world - boundary_margin:
            self.data.qvel[1] = 0.0  # Stop Y velocity when hitting Y boundary

        # Enforce horizontal velocity constraints only
        vel_xy = self.data.qvel[:2]
        speed_xy = np.linalg.norm(vel_xy)
        if speed_xy < 0.3:
            self.data.qvel[:2] = (vel_xy / speed_xy) * 0.3 if speed_xy > 0 else np.zeros(2)
        elif speed_xy > 2.0:
            self.data.qvel[:2] = (vel_xy / speed_xy) * 2.0
        
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
        self.episode_count += 1
        
        # Regenerate environment pool every 100 episodes for variety
        if self.episode_count % 100 == 0:
            self._regenerate_env_pool()
        
        # Reset UAV position and state with constant height
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']  # Ensure constant height
        self.data.qvel[:] = 0
        
        # Initialize termination info
        self.last_termination_info = {
            'terminated': False,
            'termination_reason': 'none',
            'final_position': None,
            'goal_distance': None,
            'collision_detected': False,
            'out_of_bounds': False
        }
        
        # Initialize distance tracking for reward calculation
        initial_pos = CONFIG['start_pos']
        self.previous_goal_distance = np.linalg.norm(CONFIG['goal_pos'] - initial_pos)
        
        # Switch to next pre-generated environment configuration (ultra-fast!)
        self.current_env_idx = (self.current_env_idx + 1) % self.num_env_configs
        current_config = self.env_configs[self.current_env_idx]
        
        self.obstacles = current_config['obstacles']
        print(f"🏗️ Using pre-generated config {self.current_env_idx+1}/{self.num_env_configs} with {current_config['num_obstacles']} obstacles")
        
        # Load model directly from XML string (no file I/O - fastest!)
        self.model = mujoco.MjModel.from_xml_string(current_config['xml_content'])
        self.data = mujoco.MjData(self.model)
        
        # Set initial position again after model reload
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _get_obs(self):
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        goal_dist = CONFIG['goal_pos'] - pos
        
        # Get LIDAR readings
        lidar_readings = self._get_lidar_readings(pos)
            
        return np.concatenate([pos, vel, goal_dist, lidar_readings])

    def _get_lidar_readings(self, pos):
        """Generate LIDAR readings in 360 degrees around the UAV"""
        lidar_readings = []
        
        for i in range(CONFIG['lidar_num_rays']):
            # Calculate ray direction (360 degrees divided by number of rays)
            angle = (2 * math.pi * i) / CONFIG['lidar_num_rays']
            ray_dir = np.array([math.cos(angle), math.sin(angle), 0])
            
            # Cast ray and find closest obstacle or boundary
            min_distance = CONFIG['lidar_range']
            
            # Check boundary intersection
            boundary_dist = self._ray_boundary_intersection(pos, ray_dir)
            if boundary_dist < min_distance:
                min_distance = boundary_dist
            
            # Check obstacle intersections
            for obs in self.obstacles:
                obs_dist = self._ray_obstacle_intersection(pos, ray_dir, obs)
                if obs_dist < min_distance:
                    min_distance = obs_dist
            
            lidar_readings.append(min_distance)
        
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
        
        # Survival bonus
        reward = CONFIG.get('step_reward', 0.01)
        
        # Check if UAV is out of bounds
        if self._check_out_of_bounds(pos):
            reward = CONFIG['boundary_penalty']  # -10 penalty
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'out_of_bounds'
            termination_info['out_of_bounds'] = True
            return reward, termination_info
        
        # Check for collision with obstacles
        if self._check_collision(pos):
            # Reduced collision penalty to encourage learning
            reward = -20  # Reduced from -100 to -20
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'collision'
            termination_info['collision_detected'] = True
            return reward, termination_info
            
        # Check if goal is reached
        if goal_dist < 0.5:
            reward = 100
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'goal_reached'
            return reward, termination_info
        
        # Survival reward - encourage longer episodes
        reward += 0.02  # Small positive reward for each step
        
        # Reward 0.1 for every step that reduces distance to goal
        distance_reduction = self.previous_goal_distance - goal_dist
        if distance_reduction > 0:
            reward += 0.1  # Fixed reward for any distance reduction
        
        # Update previous distance for next step
        self.previous_goal_distance = goal_dist
        
        # Additional reward for moving towards the goal (only horizontal movement)
        goal_direction = (CONFIG['goal_pos'] - pos)[:2]  # Only X,Y components
        vel_horizontal = vel[:2]
        if np.dot(vel_horizontal, goal_direction) > 0:
            reward += 0.05 * np.dot(vel_horizontal, goal_direction) / np.linalg.norm(goal_direction)

        # Distance-based reward (closer to goal = higher reward)
        reward += max(0, (8.0 - goal_dist) / 80.0)  # Scale to small positive value
        
        # LIDAR-based obstacle avoidance rewards
        lidar_readings = obs[9:]  # LIDAR readings are the last 16 values
        min_obstacle_distance = np.min(lidar_readings)
        
        # Graduated penalty system based on obstacle proximity
        if min_obstacle_distance < 0.3:  # Very close to obstacle
            reward -= 0.5  # Strong penalty
        elif min_obstacle_distance < 0.6:  # Close to obstacle
            reward -= 0.2  # Medium penalty
        elif min_obstacle_distance < 1.0:  # Near obstacle
            reward -= 0.1  # Small penalty
        else:
            reward += 0.01  # Small bonus for maintaining distance
        
        # Penalty for getting too close to boundaries
        half_world = CONFIG['world_size'] / 2
        boundary_buffer = 1.0  # Start penalizing within 1m of boundary
        
        # Calculate distance to nearest boundary
        min_boundary_dist = min(
            half_world - abs(pos[0]),  # Distance to X boundaries
            half_world - abs(pos[1])   # Distance to Y boundaries
        )
        
        # Apply penalty if too close to boundary
        if min_boundary_dist < boundary_buffer:
            boundary_penalty = -0.5 * (boundary_buffer - min_boundary_dist) / boundary_buffer
            reward += boundary_penalty
        
        # LIDAR-based obstacle avoidance reward
        lidar_readings = obs[9:]  # LIDAR readings are the last 16 values in observation
        
        # Penalty for getting too close to obstacles detected by LIDAR
        min_obstacle_distance = np.min(lidar_readings)
        safe_distance = 0.5  # Minimum safe distance from obstacles
        
        if min_obstacle_distance < safe_distance:
            # Strong penalty for being too close to obstacles
            obstacle_penalty = -1.0 * (safe_distance - min_obstacle_distance) / safe_distance
            reward += obstacle_penalty
        
        # Reward for maintaining safe distance from obstacles while moving toward goal
        if min_obstacle_distance > safe_distance:
            # Small bonus for maintaining safe distance
            reward += 0.02
        
        # Directional obstacle avoidance: check if UAV is moving away from closest obstacles
        if min_obstacle_distance < 1.0:  # Only when obstacles are relatively close
            # Find the direction of the closest obstacle
            closest_obstacle_idx = np.argmin(lidar_readings)
            obstacle_angle = (closest_obstacle_idx * 360.0 / len(lidar_readings)) * np.pi / 180.0
            
            # Direction vector to the closest obstacle
            obstacle_direction = np.array([np.cos(obstacle_angle), np.sin(obstacle_angle)])
            
            # Reward for moving away from obstacles (velocity opposite to obstacle direction)
            vel_horizontal = vel[:2]
            if np.linalg.norm(vel_horizontal) > 0:
                vel_normalized = vel_horizontal / np.linalg.norm(vel_horizontal)
                # Negative dot product means moving away from obstacle (good)
                avoidance_score = -np.dot(vel_normalized, obstacle_direction)
                if avoidance_score > 0:
                    reward += 0.05 * avoidance_score
            
        return reward, termination_info

    def _check_out_of_bounds(self, pos):
        """Check if UAV position is outside the world boundaries"""
        half_world = CONFIG['world_size'] / 2
        return (abs(pos[0]) > half_world or 
                abs(pos[1]) > half_world or 
                pos[2] < 0.5 or   # Too low (but shouldn't happen with constant height)
                pos[2] > 1.5)     # Too high (but shouldn't happen with constant height)

    def _check_collision(self, uav_pos):
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
            
            if distance < CONFIG['collision_distance']:
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
