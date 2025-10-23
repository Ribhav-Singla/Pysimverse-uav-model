# UAV Navigation Simulation - Enhanced Version with Better Path Planning
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
import torch
import os
import sys
from ppo_agent import PPOAgent
from uav_env import UAVEnv, CONFIG as ENV_CONFIG

# Define the path to your XML model
MODEL_PATH = "environment.xml"

# Use the same configuration as the training environment
CONFIG = ENV_CONFIG.copy()
# Add render-specific parameters if not present
if 'path_trail_length' not in CONFIG:
    CONFIG['path_trail_length'] = 500

class EnvironmentGenerator:
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
        
        # Add path trail geometries
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
        """Check if UAV collides with any obstacle"""
        collision_dist = CONFIG['collision_distance']
        
        for obs in obstacles:
            obs_pos = np.array(obs['pos'])
            
            if obs['shape'] == 'box':
                dx = max(obs_pos[0] - obs['size'][0] - uav_pos[0], 
                         uav_pos[0] - (obs_pos[0] + obs['size'][0]), 0)
                dy = max(obs_pos[1] - obs['size'][1] - uav_pos[1], 
                         uav_pos[1] - (obs_pos[1] + obs['size'][1]), 0)
                dz = max(obs_pos[2] - obs['size'][2] - uav_pos[2], 
                         uav_pos[2] - (obs_pos[2] + obs['size'][2]), 0)
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
            elif obs['shape'] == 'cylinder':
                horizontal_dist = math.sqrt((uav_pos[0]-obs_pos[0])**2 + (uav_pos[1]-obs_pos[1])**2)
                vertical_dist = max(0, abs(uav_pos[2]-obs_pos[2]) - obs['size'][1])
                
                if horizontal_dist <= obs['size'][0] and vertical_dist == 0:
                    distance = 0
                else:
                    distance = max(horizontal_dist - obs['size'][0], vertical_dist)
            
            if distance < collision_dist:
                return True, obs['id'], distance
        
        return False, None, float('inf')

def find_waypoint_path(current_pos, goal_pos, obstacles):
    """Simple waypoint-based path planning to avoid getting stuck"""
    # If already close to goal, go directly
    if np.linalg.norm(goal_pos[:2] - current_pos[:2]) < 1.0:
        return goal_pos[:2]
    
    # Check if direct path is clear
    if has_clear_path_to_goal(current_pos, goal_pos, obstacles):
        return goal_pos[:2]
    
    # Generate waypoints around obstacles
    waypoints = []
    
    # Add waypoints around each obstacle
    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        safety_margin = 0.5
        
        if obs['shape'] == 'box':
            radius = max(obs['size'][0], obs['size'][1]) + safety_margin
        elif obs['shape'] == 'cylinder':
            radius = obs['size'][0] + safety_margin
        else:
            continue
            
        # Generate 8 waypoints around the obstacle
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            wp_x = obs_pos[0] + radius * np.cos(angle)
            wp_y = obs_pos[1] + radius * np.sin(angle)
            waypoints.append([wp_x, wp_y])
    
    # Find the best waypoint (closest to line between current pos and goal)
    best_waypoint = None
    best_score = float('inf')
    
    goal_direction = goal_pos[:2] - current_pos[:2]
    goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
    
    for wp in waypoints:
        wp = np.array(wp)
        # Check if waypoint is safe and makes progress toward goal
        wp_dist = np.linalg.norm(wp - current_pos[:2])
        goal_progress = np.dot(wp - current_pos[:2], goal_direction)
        
        if wp_dist > 0.2 and goal_progress > 0:  # Must be some distance away and make progress
            score = wp_dist - goal_progress * 0.5  # Prefer waypoints that make progress
            if score < best_score:
                best_score = score
                best_waypoint = wp
    
    if best_waypoint is not None:
        return best_waypoint
    else:
        # Fall back to direct goal direction
        return current_pos[:2] + goal_direction * 0.5

def has_clear_path_to_goal(current_pos, goal_pos, obstacles):
    """Enhanced clear path detection with better obstacle checking"""
    direction = goal_pos[:2] - current_pos[:2]
    distance = np.linalg.norm(direction)
    
    if distance < 0.1:  # Already at goal
        return True
    
    # Normalize direction
    direction = direction / distance
    
    # Check for obstacle intersections along the path with higher resolution
    num_checks = max(10, int(distance * 20))  # More frequent checks
    for i in range(1, num_checks):
        check_pos = current_pos[:2] + (direction * i * distance / num_checks)
        check_pos_3d = np.array([check_pos[0], check_pos[1], current_pos[2]])
        
        # Check collision with obstacles using a smaller safety margin
        for obs in obstacles:
            obs_pos = np.array(obs['pos'])
            
            if obs['shape'] == 'box':
                if (abs(check_pos_3d[0] - obs_pos[0]) < obs['size'][0] + 0.15 and
                    abs(check_pos_3d[1] - obs_pos[1]) < obs['size'][1] + 0.15):
                    return False
                    
            elif obs['shape'] == 'cylinder':
                horizontal_dist = math.sqrt((check_pos_3d[0]-obs_pos[0])**2 + (check_pos_3d[1]-obs_pos[1])**2)
                if horizontal_dist < obs['size'][0] + 0.15:
                    return False
    
    return True

def update_path_trail(model, data, current_pos, path_history):
    """Update the path trail visualization"""
    path_history.append(current_pos.copy())
    if len(path_history) > CONFIG['path_trail_length']:
        path_history.pop(0)
    
    trail_count = len(path_history)
    total_geoms = model.ngeom
    trail_start_idx = total_geoms - CONFIG['path_trail_length']
    
    for i in range(CONFIG['path_trail_length']):
        geom_idx = trail_start_idx + i
        if 0 <= geom_idx < total_geoms and i < trail_count:
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
    """Makes sure the UAV remains visible"""
    for i in range(model.ngeom):
        try:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        except:
            geom_name = ""
        
        if "uav_body" in geom_name:
            model.geom_rgba[i] = [1.0, 0.0, 0.0, 1.0]  # Red
        elif "prop" in geom_name:
            model.geom_rgba[i] = [0.0, 0.0, 0.0, 1.0]  # Black

def main():
    # Initialize UAV environment
    print("🏗️ Initializing UAV environment for rendering...")
    env = UAVEnv()
    print("🏗️ Using training environment obstacles...")
    obstacles = env.obstacles
    
    # The positions are already in the CONFIG dictionary from uav_env
    # No need to update CONFIG as env uses the same CONFIG
    
    print(f"✅ Start position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
    print(f"✅ Goal position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
    
    # Create environment XML
    EnvironmentGenerator.create_xml_with_obstacles(obstacles)
    
    # Load model and create simulation data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    print(f"🚁 UAV Navigation Environment Loaded!")
    print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
    print(f"🎯 Mission: Navigate from START (green) to GOAL (blue)")
    
    # Initialize PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    action_std = 1.0
    lr_actor = 0.0001
    lr_critic = 0.0004
    gamma = 0.999
    K_epochs = 12
    eps_clip = 0.1
    
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    # Load trained model
    try:
        model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_1_0.pth"
        if os.path.exists(model_path):
            ppo_agent.load(model_path)
            print(f"🤖 Trained PPO agent loaded successfully!")
        else:
            raise FileNotFoundError(f"Weight file not found: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not load trained agent: {str(e)}")
        print("⚠️ Using random actions.")
    
    path_history = []
    stuck_counter = 0
    last_position = None
    current_waypoint = None
    
    # Open viewer and start simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset simulation
        mujoco.mj_resetData(model, data)
        
        # Set initial position
        data.qpos[:3] = CONFIG['start_pos']
        data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation
        
        print("🚀 Mission started! UAV navigating to goal...")
        time.sleep(2)
        
        step_count = 0
        mission_complete = False
        collision_occurred = False
        goal_reached = False
        
        try:
            while viewer.is_running() and not mission_complete and not collision_occurred and step_count < 5000:
                # Get current UAV state
                current_pos = data.qpos[:3].copy()
                current_vel = data.qvel[:3].copy()
                
                # Update environment with current position
                env.uav_pos = current_pos
                
                # Check if stuck (not making progress)
                if last_position is not None:
                    movement = np.linalg.norm(current_pos[:2] - last_position[:2])
                    if movement < 0.05:  # Very little movement
                        stuck_counter += 1
                    else:
                        stuck_counter = 0
                        
                    # If stuck for too long, find a waypoint
                    if stuck_counter > 50:
                        current_waypoint = find_waypoint_path(current_pos, CONFIG['goal_pos'], obstacles)
                        stuck_counter = 0
                        print(f"🔄 Stuck detected! New waypoint: ({current_waypoint[0]:.2f}, {current_waypoint[1]:.2f})")
                
                last_position = current_pos.copy()
                
                # Create observation exactly as in training
                pos = current_pos
                vel = current_vel
                goal_dist = CONFIG['goal_pos'] - pos
                
                # Get LIDAR readings
                lidar_readings = env._get_lidar_readings(pos)
                
                # LIDAR feature engineering (same as training)
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
                
                # Combine LIDAR features
                lidar_features = np.array([
                    min_lidar, mean_lidar,
                    obstacle_direction[0], obstacle_direction[1],
                    danger_level,
                    front_clear, right_clear, back_clear, left_clear,
                    goal_direction_norm[0], goal_direction_norm[1]
                ])
                
                # Create state vector
                state = np.concatenate([pos, vel, goal_dist, lidar_readings, lidar_features])
                state_tensor = torch.FloatTensor(state.reshape(1, -1))
                
                # Agent selects action
                raw_action, _ = ppo_agent.select_action(state_tensor)
                action = np.clip(raw_action.flatten(), -1.0, 1.0)
                
                # Calculate goal distance
                goal_distance = np.linalg.norm(current_pos - CONFIG['goal_pos'])
                
                # Debug output
                if step_count % 100 == 0:
                    print(f"🎮 Step {step_count}: Action=[{action[0]:.3f}, {action[1]:.3f}] | "
                          f"Pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                          f"Goal dist={goal_distance:.2f}m")
                
                # Enhanced movement logic with waypoint navigation
                target_vel = action[:2]
                
                # Check boundaries
                half_world = CONFIG['world_size'] / 2
                distances_to_boundaries = np.array([
                    half_world + current_pos[0],  # West
                    half_world - current_pos[0],  # East  
                    half_world + current_pos[1],  # South
                    half_world - current_pos[1]   # North
                ])
                distance_to_boundary = float(np.min(distances_to_boundaries))
                
                # Priority 1: Smart boundary avoidance (considers goal position)
                if distance_to_boundary < 0.8:
                    boundary_escape_dir = np.zeros(2)
                    goal_dir_to_check = CONFIG['goal_pos'][:2] - current_pos[:2]
                    goal_distance_2d = np.linalg.norm(goal_dir_to_check)
                    
                    # Only avoid boundary if goal is not near that boundary
                    if current_pos[0] > half_world - 0.8:  # Near east boundary
                        # Only avoid if goal is not also near east boundary
                        if CONFIG['goal_pos'][0] < half_world - 1.0:
                            boundary_escape_dir[0] = -1.0
                    elif current_pos[0] < -(half_world - 0.8):  # Near west boundary
                        # Only avoid if goal is not also near west boundary  
                        if CONFIG['goal_pos'][0] > -(half_world - 1.0):
                            boundary_escape_dir[0] = 1.0
                        
                    if current_pos[1] > half_world - 0.8:  # Near north boundary
                        # Only avoid if goal is not also near north boundary
                        if CONFIG['goal_pos'][1] < half_world - 1.0:
                            boundary_escape_dir[1] = -1.0
                    elif current_pos[1] < -(half_world - 0.8):  # Near south boundary
                        # Only avoid if goal is not also near south boundary
                        if CONFIG['goal_pos'][1] > -(half_world - 1.0):
                            boundary_escape_dir[1] = 1.0
                    
                    # If we're very close to goal and near boundary, allow approach
                    if goal_distance < 1.5 and np.linalg.norm(boundary_escape_dir) > 0:
                        # Mix boundary avoidance with goal approach
                        goal_dir_norm = goal_dir_to_check / (goal_distance_2d + 1e-8)
                        boundary_escape_dir = boundary_escape_dir / np.linalg.norm(boundary_escape_dir)
                        
                        # When very close to goal, prioritize goal approach over boundary avoidance
                        target_vel = 0.3 * boundary_escape_dir + 0.7 * goal_dir_norm
                        
                        if step_count % 50 == 0:
                            print(f"🎯 Near-goal boundary navigation | Goal dist: {goal_distance:.2f}m")
                    elif np.linalg.norm(boundary_escape_dir) > 0:
                        # Normal boundary avoidance
                        boundary_escape_dir = boundary_escape_dir / np.linalg.norm(boundary_escape_dir)
                        target_vel = boundary_escape_dir * 0.8
                        
                        if step_count % 50 == 0:
                            print(f"⚠️ Boundary avoidance active | Distance: {distance_to_boundary:.2f}m")
                
                # Priority 2: Waypoint navigation if we have one
                elif current_waypoint is not None:
                    waypoint_dir = current_waypoint - current_pos[:2]
                    waypoint_dist = np.linalg.norm(waypoint_dir)
                    
                    if waypoint_dist < 0.25:  # Close to waypoint, clear it
                        current_waypoint = None
                        if step_count % 50 == 0:
                            print(f"✅ Waypoint reached! Resuming normal navigation.")
                    else:
                        # Move toward waypoint with stronger control when close to goal
                        waypoint_dir = waypoint_dir / waypoint_dist
                        
                        # Stronger waypoint influence when close to goal
                        if goal_distance < 2.0:
                            # Very close to goal - use strong waypoint control
                            waypoint_strength = 0.9
                            action_strength = 0.1
                        elif goal_distance < 3.0:
                            # Moderately close - balanced control
                            waypoint_strength = 0.7
                            action_strength = 0.3
                        else:
                            # Far from goal - moderate waypoint influence
                            waypoint_strength = 0.6
                            action_strength = 0.4
                        
                        target_vel = action_strength * target_vel + waypoint_strength * waypoint_dir
                        
                        if step_count % 50 == 0:
                            print(f"🎯 Waypoint nav | Dist: {waypoint_dist:.2f}m | Strength: {waypoint_strength:.1f}")
                
                # Priority 3: Clear path boost (lowest priority)
                else:
                    has_clear_path = has_clear_path_to_goal(current_pos, CONFIG['goal_pos'], obstacles)
                    
                    if has_clear_path and goal_distance > 0.5:
                        # Strong goal direction boost
                        goal_direction = (CONFIG['goal_pos'][:2] - current_pos[:2])
                        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
                        
                        # Strong boost toward goal
                        target_vel = target_vel * 0.3 + goal_direction * 0.8
                        
                        if step_count % 50 == 0:
                            print(f"🚀 Clear path boost active | Goal direction applied")
                
                # Ensure velocity bounds
                vel_magnitude = np.linalg.norm(target_vel)
                if vel_magnitude > 1.0:
                    target_vel = target_vel / vel_magnitude
                
                # Apply velocity to simulation
                data.qvel[0] = float(target_vel[0])
                data.qvel[1] = float(target_vel[1])
                data.qvel[2] = 0.0  # No vertical movement
                
                # Maintain constant height
                data.qpos[2] = CONFIG['uav_flight_height']
                
                # Check if goal reached
                if goal_distance < 0.6 and not goal_reached:
                    print(f"\n🎯 GOAL REACHED! Distance: {goal_distance:.3f}m")
                    goal_reached = True
                    mission_complete = True
                    mission_complete = True
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Enforce velocity limits post-step
                current_vel_magnitude = np.linalg.norm(data.qvel[:2])
                if current_vel_magnitude > 1.0:
                    data.qvel[0] = np.clip(data.qvel[0], -1.0, 1.0)
                    data.qvel[1] = np.clip(data.qvel[1], -1.0, 1.0)
                
                # Update visualization
                if step_count % 5 == 0:
                    update_path_trail(model, data, current_pos, path_history)
                
                ensure_uav_visibility(model)
                viewer.sync()
                
                # Hard boundary enforcement
                boundary_margin = 0.1
                if (abs(current_pos[0]) > half_world - boundary_margin or 
                    abs(current_pos[1]) > half_world - boundary_margin):
                    
                    data.qvel[0] = 0.0
                    data.qvel[1] = 0.0
                    
                    # Pull back to safe zone
                    if current_pos[0] > half_world - boundary_margin:
                        data.qpos[0] = half_world - boundary_margin
                    elif current_pos[0] < -(half_world - boundary_margin):
                        data.qpos[0] = -(half_world - boundary_margin)
                        
                    if current_pos[1] > half_world - boundary_margin:
                        data.qpos[1] = half_world - boundary_margin
                    elif current_pos[1] < -(half_world - boundary_margin):
                        data.qpos[1] = -(half_world - boundary_margin)
                    
                    print(f"\n🛑 Boundary hit - UAV repositioned!")
                
                # Check for complete boundary violation
                if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                    current_pos[2] < 0.1 or current_pos[2] > 5.0):
                    collision_occurred = True
                    print(f"\n🚨 BOUNDARY VIOLATION! Mission failed.")
                    break
                
                # Check for obstacle collision
                has_collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(data.qpos[:3], obstacles)
                if has_collision:
                    collision_occurred = True
                    print(f"\n💥 COLLISION with {obstacle_id}! Distance: {collision_dist:.3f}m")
                    break
                
                step_count += 1
                time.sleep(CONFIG['control_dt'])
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        # Final status
        if mission_complete and not collision_occurred:
            print("\n" + "="*50)
            print("🎉 MISSION SUCCESS!")
            print("🏆 UAV successfully reached the goal!")
            print(f"📊 Steps taken: {step_count}")
            print("="*50)
            
            # Victory celebration
            for _ in range(200):
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
                
        elif collision_occurred:
            print("\n" + "="*50)
            print("💥 MISSION FAILED - COLLISION!")
            print("="*50)
            
        elif step_count >= 5000:
            print("\n" + "="*50)
            print("⏰ MISSION TIMEOUT!")
            print("🔄 UAV did not reach goal within time limit")
            print("="*50)
            
        else:
            print("\n" + "="*50)
            print("⏹️ SIMULATION TERMINATED")
            print("="*50)

if __name__ == "__main__":
    main()