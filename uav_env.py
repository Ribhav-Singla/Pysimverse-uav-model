import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import math
import random

# Configuration from uav_render.py, adapted for the environment
CONFIG = {
    'start_pos': np.array([-4.0, -4.0, 1.8]),
    'goal_pos': np.array([4.0, 4.0, 1.8]),
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.8,
    'static_obstacles': 8,
    'min_obstacle_size': 0.2,
    'max_obstacle_size': 0.6,
    'collision_distance': 0.2,
    'control_dt': 0.05,
    'max_steps': 50000,  # Max steps per episode
}

class EnvironmentGenerator:
    @staticmethod
    def generate_obstacles():
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        grid_size = int(math.sqrt(CONFIG['static_obstacles'])) + 1
        cell_size = world_size / grid_size
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = -half_world + (i + 0.5) * cell_size
                y = -half_world + (j + 0.5) * cell_size
                positions.append((x, y))
        
        random.shuffle(positions)
        
        for i in range(CONFIG['static_obstacles']):
            x, y = positions[i]
            
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

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.obstacles = EnvironmentGenerator.generate_obstacles()
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Observation space: [pos(3), vel(3), goal_dist(3), closest_obstacle_dist(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Action space: 4 motor controls
        self.action_space = spaces.Box(low=1.0, high=10.0, shape=(4,), dtype=np.float32)
        
        self.viewer = None
        self.step_count = 0

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Enforce velocity constraints
        vel = self.data.qvel[:3]
        speed = np.linalg.norm(vel)
        if speed < 0.5:
            self.data.qvel[:3] = (vel / speed) * 0.5 if speed > 0 else np.zeros(3)
        elif speed > 1.5:
            self.data.qvel[:3] = (vel / speed) * 1.5
        
        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = self.step_count >= CONFIG['max_steps']
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Reset UAV position and state
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        
        # Regenerate obstacles for each episode
        self.obstacles = EnvironmentGenerator.generate_obstacles()
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
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
        
        # Calculate distance to the closest obstacle
        min_obs_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - np.array(obs['pos']))
            min_obs_dist = min(min_obs_dist, dist)
            
        return np.concatenate([pos, vel, goal_dist, [min_obs_dist]])

    def _get_reward(self, obs):
        pos = obs[:3]
        vel = obs[3:6]
        goal_dist = np.linalg.norm(CONFIG['goal_pos'] - pos)
        
        reward = 0
        # Reward for moving towards the goal
        if np.dot(vel, (CONFIG['goal_pos'] - pos)) > 0:
            reward += 0.1

        # Penalty for collision
        if self._check_collision(pos):
            reward = -100
            
        # Reward for reaching the goal
        if goal_dist < 0.5:
            reward = 100
            
        return reward

    def _is_terminated(self, obs):
        pos = obs[:3]
        goal_dist = np.linalg.norm(CONFIG['goal_pos'] - pos)
        
        return self._check_collision(pos) or goal_dist < 0.5

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
