# UAV Navigation Simulation - Headless Parameterized Version
# This version runs without the MuJoCo viewer for compatibility and faster execution

import numpy as np
import mujoco
import time
import math
import random
import torch
import os
import sys
from ppo_agent import PPOAgent
from uav_env import UAVEnv, CONFIG as ENV_CONFIG

def run_uav_simulation_headless(
    start_pos,
    goal_pos,
    obstacle_count,
    model_path,
    obstacle_positions,
    world_size=None,
    uav_flight_height=None,
    control_dt=None,
    collision_distance=None,
    lidar_num_rays=None,
    lidar_max_range=None,
    max_steps=5000
):
    """
    Run UAV simulation headless (without viewer) with custom parameters.
    
    Parameters:
    -----------
    start_pos : list or np.array
        Starting position [x, y, z] of the UAV
    goal_pos : list or np.array
        Goal position [x, y, z] for the UAV
    obstacle_count : int
        Number of obstacles to generate/use
    model_path : str
        Path to the trained PPO model weights file
    obstacle_positions : list of dict
        Array of obstacle dictionaries with keys:
        - 'id': str, unique identifier
        - 'shape': str, 'box' or 'cylinder'
        - 'pos': list [x, y, z], position
        - 'size': list [size params based on shape]
        - 'color': list [r, g, b, a], RGBA color
    world_size : float, optional
        Size of the world (default: from ENV_CONFIG)
    uav_flight_height : float, optional
        Flight height of UAV (default: from ENV_CONFIG)
    control_dt : float, optional
        Control timestep (default: from ENV_CONFIG)
    collision_distance : float, optional
        Collision detection distance (default: from ENV_CONFIG)
    lidar_num_rays : int, optional
        Number of LIDAR rays (default: from ENV_CONFIG)
    lidar_max_range : float, optional
        Maximum LIDAR range (default: from ENV_CONFIG)
    max_steps : int, optional
        Maximum steps before timeout (default: 5000)
    
    Returns:
    --------
    dict : Mission results including success status, steps taken, etc.
    """
    
    # Create a custom CONFIG based on parameters
    CUSTOM_CONFIG = ENV_CONFIG.copy()
    CUSTOM_CONFIG['start_pos'] = np.array(start_pos)
    CUSTOM_CONFIG['goal_pos'] = np.array(goal_pos)
    
    # Override optional parameters if provided
    if world_size is not None:
        CUSTOM_CONFIG['world_size'] = world_size
    if uav_flight_height is not None:
        CUSTOM_CONFIG['uav_flight_height'] = uav_flight_height
    if control_dt is not None:
        CUSTOM_CONFIG['control_dt'] = control_dt
    if collision_distance is not None:
        CUSTOM_CONFIG['collision_distance'] = collision_distance
    if lidar_num_rays is not None:
        CUSTOM_CONFIG['lidar_num_rays'] = lidar_num_rays
    if lidar_max_range is not None:
        CUSTOM_CONFIG['lidar_max_range'] = lidar_max_range
    
    # Use provided obstacles
    obstacles = obstacle_positions[:obstacle_count] if len(obstacle_positions) > obstacle_count else obstacle_positions
    
    # Define the path to the XML model
    MODEL_XML_PATH = "environment_headless.xml"
    
    def create_xml_with_obstacles(obstacles):
        """Create XML model with dynamically generated obstacles"""
        xml_template = f'''<mujoco model="headless_uav_env">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="{CUSTOM_CONFIG['world_size']/2} {CUSTOM_CONFIG['world_size']/2} 0.1"/>
    
    <!-- UAV starting position -->
    <body name="chassis" pos="{CUSTOM_CONFIG['start_pos'][0]} {CUSTOM_CONFIG['start_pos'][1]} {CUSTOM_CONFIG['start_pos'][2]}">
      <joint type="free" name="root"/>
      <geom name="uav_body" type="box" size="0.12 0.12 0.03" rgba="1.0 0.0 0.0 1.0" mass="0.8"/>
      
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
        with open(MODEL_XML_PATH, 'w') as f:
            f.write(xml_template)
        
        return obstacles
    
    def check_collision(uav_pos, obstacles):
        """Check if UAV collides with any obstacle"""
        collision_dist = CUSTOM_CONFIG['collision_distance']
        
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

    # Initialize UAV environment
    print("🏗️ Initializing UAV environment for headless simulation...")
    print(f"✅ Start position: [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
    print(f"✅ Goal position: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f}]")
    print(f"✅ Obstacles: {obstacle_count}")
    
    # Create environment XML
    create_xml_with_obstacles(obstacles)
    
    # Load model and create simulation data
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    
    print(f"🚁 UAV Navigation Environment Loaded!")
    print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
    print(f"🎯 Mission: Navigate from START to GOAL")
    
    # Initialize PPO agent
    # Create a temporary environment to get state/action dimensions
    temp_env = UAVEnv()
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    
    action_std = 1.0
    lr_actor = 0.0001
    lr_critic = 0.0004
    gamma = 0.999
    K_epochs = 12
    eps_clip = 0.1
    
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    # Load trained model
    try:
        if os.path.exists(model_path):
            ppo_agent.load(model_path)
            print(f"🤖 Trained PPO agent loaded successfully from {model_path}!")
        else:
            raise FileNotFoundError(f"Weight file not found: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not load trained agent: {str(e)}")
        print("⚠️ Using random actions.")
    
    stuck_counter = 0
    last_position = None
    current_waypoint = None
    
    # Results dictionary
    results = {
        'success': False,
        'collision': False,
        'timeout': False,
        'steps': 0,
        'final_distance': 0.0,
        'path': []
    }
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position
    data.qpos[:3] = CUSTOM_CONFIG['start_pos']
    data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation
    
    print("🚀 Mission started! UAV navigating to goal...")
    
    step_count = 0
    mission_complete = False
    collision_occurred = False
    goal_reached = False
    
    try:
        while not mission_complete and not collision_occurred and step_count < max_steps:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Store position for path tracking
            results['path'].append(current_pos.tolist())
            
            # Update environment with current position (for LIDAR)
            temp_env.uav_pos = current_pos
            
            # Check if stuck (not making progress)
            if last_position is not None:
                movement = np.linalg.norm(current_pos[:2] - last_position[:2])
                if movement < 0.05:  # Very little movement
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                    
                # If stuck for too long, find a waypoint
                if stuck_counter > 50:
                    current_waypoint = find_waypoint_path(current_pos, CUSTOM_CONFIG['goal_pos'], obstacles)
                    stuck_counter = 0
            
            last_position = current_pos.copy()
            
            # Create observation exactly as in training
            pos = current_pos
            vel = current_vel
            goal_dist = CUSTOM_CONFIG['goal_pos'] - pos
            
            # Get LIDAR readings
            lidar_readings = temp_env._get_lidar_readings(pos)
            
            # LIDAR feature engineering (same as training)
            min_lidar = np.min(lidar_readings)
            mean_lidar = np.mean(lidar_readings)
            
            closest_idx = np.argmin(lidar_readings)
            closest_angle = (2 * np.pi * closest_idx) / CUSTOM_CONFIG['lidar_num_rays']
            obstacle_direction = np.array([np.cos(closest_angle), np.sin(closest_angle)])
            
            danger_threshold = 0.36
            num_close_obstacles = np.sum(lidar_readings < danger_threshold)
            danger_level = num_close_obstacles / CUSTOM_CONFIG['lidar_num_rays']
            
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
            goal_distance = np.linalg.norm(current_pos - CUSTOM_CONFIG['goal_pos'])
            
            # Debug output (less frequent for headless mode)
            if step_count % 500 == 0:
                print(f"🎮 Step {step_count}: Pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Goal dist={goal_distance:.2f}m")
            
            # Enhanced movement logic with waypoint navigation
            target_vel = action[:2]
            
            # Check boundaries
            half_world = CUSTOM_CONFIG['world_size'] / 2
            distances_to_boundaries = np.array([
                half_world + current_pos[0],  # West
                half_world - current_pos[0],  # East  
                half_world + current_pos[1],  # South
                half_world - current_pos[1]   # North
            ])
            distance_to_boundary = float(np.min(distances_to_boundaries))
            
            # Priority 1: Smart boundary avoidance
            if distance_to_boundary < 0.8:
                boundary_escape_dir = np.zeros(2)
                goal_dir_to_check = CUSTOM_CONFIG['goal_pos'][:2] - current_pos[:2]
                goal_distance_2d = np.linalg.norm(goal_dir_to_check)
                
                if current_pos[0] > half_world - 0.8:
                    if CUSTOM_CONFIG['goal_pos'][0] < half_world - 1.0:
                        boundary_escape_dir[0] = -1.0
                elif current_pos[0] < -(half_world - 0.8):
                    if CUSTOM_CONFIG['goal_pos'][0] > -(half_world - 1.0):
                        boundary_escape_dir[0] = 1.0
                        
                if current_pos[1] > half_world - 0.8:
                    if CUSTOM_CONFIG['goal_pos'][1] < half_world - 1.0:
                        boundary_escape_dir[1] = -1.0
                elif current_pos[1] < -(half_world - 0.8):
                    if CUSTOM_CONFIG['goal_pos'][1] > -(half_world - 1.0):
                        boundary_escape_dir[1] = 1.0
                
                if goal_distance < 1.5 and np.linalg.norm(boundary_escape_dir) > 0:
                    goal_dir_norm = goal_dir_to_check / (goal_distance_2d + 1e-8)
                    boundary_escape_dir = boundary_escape_dir / np.linalg.norm(boundary_escape_dir)
                    target_vel = 0.3 * boundary_escape_dir + 0.7 * goal_dir_norm
                elif np.linalg.norm(boundary_escape_dir) > 0:
                    boundary_escape_dir = boundary_escape_dir / np.linalg.norm(boundary_escape_dir)
                    target_vel = boundary_escape_dir * 0.8
            
            # Priority 2: Waypoint navigation
            elif current_waypoint is not None:
                waypoint_dir = current_waypoint - current_pos[:2]
                waypoint_dist = np.linalg.norm(waypoint_dir)
                
                if waypoint_dist < 0.25:
                    current_waypoint = None
                else:
                    waypoint_dir = waypoint_dir / waypoint_dist
                    
                    if goal_distance < 2.0:
                        waypoint_strength = 0.9
                        action_strength = 0.1
                    elif goal_distance < 3.0:
                        waypoint_strength = 0.7
                        action_strength = 0.3
                    else:
                        waypoint_strength = 0.6
                        action_strength = 0.4
                    
                    target_vel = action_strength * target_vel + waypoint_strength * waypoint_dir
            
            # Priority 3: Clear path boost
            else:
                has_clear_path = has_clear_path_to_goal(current_pos, CUSTOM_CONFIG['goal_pos'], obstacles)
                
                if has_clear_path and goal_distance > 0.5:
                    goal_direction = (CUSTOM_CONFIG['goal_pos'][:2] - current_pos[:2])
                    goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
                    target_vel = target_vel * 0.3 + goal_direction * 0.8
            
            # Ensure velocity bounds
            vel_magnitude = np.linalg.norm(target_vel)
            if vel_magnitude > 1.0:
                target_vel = target_vel / vel_magnitude
            
            # Apply velocity to simulation
            data.qvel[0] = float(target_vel[0])
            data.qvel[1] = float(target_vel[1])
            data.qvel[2] = 0.0  # No vertical movement
            
            # Maintain constant height
            data.qpos[2] = CUSTOM_CONFIG['uav_flight_height']
            
            # Check if goal reached
            if goal_distance < 0.1 and not goal_reached:
                goal_reached = True
                mission_complete = True
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Enforce velocity limits post-step
            current_vel_magnitude = np.linalg.norm(data.qvel[:2])
            if current_vel_magnitude > 1.0:
                data.qvel[0] = np.clip(data.qvel[0], -1.0, 1.0)
                data.qvel[1] = np.clip(data.qvel[1], -1.0, 1.0)
            
            # Hard boundary enforcement
            boundary_margin = 0.1
            if (abs(current_pos[0]) > half_world - boundary_margin or 
                abs(current_pos[1]) > half_world - boundary_margin):
                
                data.qvel[0] = 0.0
                data.qvel[1] = 0.0
                
                if current_pos[0] > half_world - boundary_margin:
                    data.qpos[0] = half_world - boundary_margin
                elif current_pos[0] < -(half_world - boundary_margin):
                    data.qpos[0] = -(half_world - boundary_margin)
                    
                if current_pos[1] > half_world - boundary_margin:
                    data.qpos[1] = half_world - boundary_margin
                elif current_pos[1] < -(half_world - boundary_margin):
                    data.qpos[1] = -(half_world - boundary_margin)
            
            # Check for complete boundary violation
            if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                current_pos[2] < 0.1 or current_pos[2] > 5.0):
                collision_occurred = True
                break
            
            # Check for obstacle collision
            has_collision, obstacle_id, collision_dist = check_collision(data.qpos[:3], obstacles)
            if has_collision:
                collision_occurred = True
                break
            
            step_count += 1
            time.sleep(CUSTOM_CONFIG['control_dt'] * 0.1)  # Faster execution for headless mode
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Update results
    results['steps'] = step_count
    results['final_distance'] = np.linalg.norm(current_pos - CUSTOM_CONFIG['goal_pos'])
    
    # Final status
    if mission_complete and not collision_occurred:
        results['success'] = True
        print(f"✅ GOAL REACHED! Steps: {step_count}, Final dist: {results['final_distance']:.3f}m")
    elif collision_occurred:
        results['collision'] = True
        print(f"💥 COLLISION! Steps: {step_count}")
    elif step_count >= max_steps:
        results['timeout'] = True
        print(f"⏰ TIMEOUT! Steps: {step_count}")
    else:
        print(f"⏹️ TERMINATED! Steps: {step_count}")
    
    return results


if __name__ == "__main__":
    # Example usage
    start_position = [-3.5, -3.5, 0.5]
    goal_position = [3.5, 3.5, 0.5]
    num_obstacles = 10
    weights_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth"
    
    # Example obstacle array
    example_obstacles = [
        {
            'id': 'obstacle_0',
            'shape': 'cylinder',
            'pos': [0.5, 0.5, 0.5],
            'size': [0.3, 0.5],
            'color': [0.7, 0.3, 0.3, 1.0]
        },
    ]
    
    # Run simulation
    results = run_uav_simulation_headless(
        start_pos=start_position,
        goal_pos=goal_position,
        obstacle_count=num_obstacles,
        model_path=weights_path,
        obstacle_positions=example_obstacles
    )
    
    print("\n📊 Final Results:")
    print(f"Success: {results['success']}")
    print(f"Collision: {results['collision']}")
    print(f"Timeout: {results['timeout']}")
    print(f"Steps: {results['steps']}")
    print(f"Final Distance: {results['final_distance']:.3f}m")