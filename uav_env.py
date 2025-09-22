# UAV RL Environment Wrapper
import gymnasium as gym
import numpy as np
import mujoco
import math
import random
from gymnasium import spaces
from uav_config import EnvironmentGenerator, CONFIG

class UAVNavigationEnv(gym.Env):
    """
    UAV Navigation Environment for RL Training
    
    Observation Space: [current_pos(3), current_vel(3), goal_pos(3)] = 9D
    Action Space: [vx, vy, vz] = 3D continuous velocity commands
    """
    
    def __init__(self, render_mode=None, max_episode_steps=50000):
        super().__init__()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.render_mode = render_mode
        
        # MuJoCo model and data
        self.model = None
        self.data = None
        self.obstacles = []
        
        # Define observation space: [pos(3), vel(3), goal(3)]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, 0.0, -2.0, -2.0, -2.0, -5.0, -5.0, 0.0]),
            high=np.array([5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 5.0, 5.0, 3.0]),
            dtype=np.float32
        )
        
        # Define action space: [vx, vy, vz] velocity commands
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5]),
            high=np.array([1.0, 1.0, 0.5]),
            dtype=np.float32
        )
        
        # State variables
        self.start_pos = CONFIG['start_pos'].copy()
        self.goal_pos = CONFIG['goal_pos'].copy()
        self.current_pos = self.start_pos.copy()
        self.current_vel = np.zeros(3)
        self.previous_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # Initialize environment
        self._init_environment()
    
    def _init_environment(self):
        """Initialize the MuJoCo environment"""
        # Generate obstacles and create XML
        self.obstacles = EnvironmentGenerator.create_xml_with_obstacles()
        
        # Load MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path("uav_model.xml")
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"Error loading MuJoCo model: {e}")
            raise
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Randomize goal position slightly for better generalization
        if options and options.get('random_goal', False):
            self.goal_pos = np.array([
                random.uniform(3.0, 4.5),
                random.uniform(3.0, 4.5),
                CONFIG['uav_flight_height']
            ])
        else:
            self.goal_pos = CONFIG['goal_pos'].copy()
        
        # Reset UAV to start position
        self.current_pos = self.start_pos.copy()
        self.current_vel = np.zeros(3)
        self.previous_distance = np.linalg.norm(self.goal_pos - self.current_pos)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.current_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Reset orientation
        self.data.qvel[:6] = 0  # Reset velocities
        
        # Get initial observation
        observation = self._get_observation()
        info = {'distance_to_goal': self.previous_distance}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Apply action (velocity commands)
        action = np.clip(action, -1.0, 1.0)  # Direct clipping to [-1, 1]
        
        # Convert velocity commands to position update
        dt = CONFIG['control_dt']
        velocity_command = action * CONFIG['velocity'] * 2.0  # Scale velocity
        
        # Update position based on velocity command
        new_pos = self.current_pos + velocity_command * dt
        
        # Keep UAV within world bounds
        world_bound = CONFIG['world_size'] / 2 - 0.5
        new_pos[0] = np.clip(new_pos[0], -world_bound, world_bound)
        new_pos[1] = np.clip(new_pos[1], -world_bound, world_bound)
        new_pos[2] = np.clip(new_pos[2], 0.5, 2.5)  # Height bounds
        
        # Update velocity (for observation)
        self.current_vel = (new_pos - self.current_pos) / dt
        self.current_pos = new_pos
        
        # Update MuJoCo simulation
        self.data.qpos[:3] = self.current_pos
        mujoco.mj_forward(self.model, self.data)
        
        # Calculate reward
        reward, info = self._calculate_reward()
        
        # Check for termination
        terminated, truncated = self._check_termination()
        
        # Get observation
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        obs = np.concatenate([
            self.current_pos,     # Current position (3)
            self.current_vel,     # Current velocity (3)
            self.goal_pos        # Goal position (3)
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """Calculate reward for current state"""
        reward = 0.0
        info = {}
        
        # Check for collision
        collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(
            self.current_pos, self.obstacles
        )
        
        if collision:
            reward = -100.0
            info['collision'] = True
            info['obstacle_id'] = obstacle_id
            return reward, info
        
        # Check if reached goal
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        
        if distance_to_goal < 0.5:  # Goal reached
            reward = 100.0
            info['goal_reached'] = True
            return reward, info
        
        # Reward for moving towards goal
        if distance_to_goal < self.previous_distance:
            reward += 0.1  # Moving towards goal
        
        # Small penalty for time to encourage efficiency
        reward -= 0.001
        
        # Update distance tracking
        self.previous_distance = distance_to_goal
        info['distance_to_goal'] = distance_to_goal
        
        return reward, info
    
    def _check_termination(self):
        """Check if episode should terminate"""
        terminated = False
        truncated = False
        
        # Check collision
        collision, _, _ = EnvironmentGenerator.check_collision(
            self.current_pos, self.obstacles
        )
        if collision:
            terminated = True
        
        # Check goal reached
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        if distance_to_goal < 0.5:
            terminated = True
        
        # Check timeout
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Check if UAV is out of bounds (safety)
        world_bound = CONFIG['world_size'] / 2
        if (abs(self.current_pos[0]) > world_bound or 
            abs(self.current_pos[1]) > world_bound or
            self.current_pos[2] < 0.2 or self.current_pos[2] > 3.0):
            terminated = True
        
        return terminated, truncated
    
    def render(self):
        """Render the environment (optional for training)"""
        if self.render_mode == "human":
            # This would open the MuJoCo viewer
            # For training, we typically don't render to save computation
            pass
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Test the environment
if __name__ == "__main__":
    env = UAVNavigationEnv()
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated or truncated}")
        
        if terminated or truncated:
            break
    
    env.close()