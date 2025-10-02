# Neurosymbolic UAV Navigation Rendering
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
import torch
import os
import sys
from neurosymbolic_ppo_agent import NeuroSymbolicPPOAgent
from neurosymbolic_rdr import RDRKnowledgeBase, NeuroSymbolicIntegrator
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
    'lidar_range': 2.9,
    'lidar_num_rays': 16,
    'step_reward': -0.01,
    
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
    def get_opposite_goal_position(start_pos):
        """Get goal position in the opposite corner"""
        half_world = CONFIG['world_size'] / 2 - 1.0
        # Choose opposite corner based on start position
        if start_pos[0] < 0 and start_pos[1] < 0:  # Bottom-left start
            return np.array([half_world, half_world, CONFIG['uav_flight_height']])  # Top-right goal
        elif start_pos[0] > 0 and start_pos[1] < 0:  # Bottom-right start
            return np.array([-half_world, half_world, CONFIG['uav_flight_height']])  # Top-left goal
        elif start_pos[0] < 0 and start_pos[1] > 0:  # Top-left start
            return np.array([half_world, -half_world, CONFIG['uav_flight_height']])  # Bottom-right goal
        else:  # Top-right start
            return np.array([-half_world, -half_world, CONFIG['uav_flight_height']])  # Bottom-left goal

    @staticmethod
    def generate_obstacles(num_obstacles=None):
        """Generate obstacles with specified count"""
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
        
        # Randomly select positions for obstacles
        selected_positions = random.sample(positions, min(num_obstacles, len(positions)))
        
        for idx, (x, y) in enumerate(selected_positions):
            # Randomize size within bounds
            size_scale = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
            
            # Determine shape
            shapes = ['box', 'cylinder', 'sphere']
            shape = random.choice(shapes)
            
            if shape == 'box':
                size = [size_scale, size_scale, CONFIG['obstacle_height']/2]
            elif shape == 'cylinder':
                size = [size_scale, CONFIG['obstacle_height']/2]
            else:  # sphere
                size = [size_scale * 1.5]  # Slightly larger for spheres
            
            # Random colors for variety
            colors = [
                [0.6, 0.3, 0.1, 1.0],  # Brown
                [0.4, 0.4, 0.4, 1.0],  # Gray
                [0.2, 0.5, 0.2, 1.0],  # Dark green
                [0.5, 0.2, 0.2, 1.0],  # Dark red
            ]
            
            obstacle = {
                'id': f'obstacle_{idx}',
                'shape': shape,
                'pos': [x, y, CONFIG['obstacle_height']/2],
                'size': size,
                'color': random.choice(colors)
            }
            obstacles.append(obstacle)
        
        return obstacles

def generate_dynamic_xml(obstacles, start_pos=None, goal_pos=None):
    """Generate MuJoCo XML with dynamic obstacles and positions"""
    # Use provided positions or defaults
    if start_pos is None:
        start_pos = CONFIG['start_pos']
    if goal_pos is None:
        goal_pos = CONFIG['goal_pos']
    
    # Update CONFIG for this run
    CONFIG['start_pos'] = start_pos
    CONFIG['goal_pos'] = goal_pos
    
    xml_template = f'''<mujoco model="uav_environment">
  <compiler angle="radian" coordinate="local"/>
  
  <option timestep="0.01" gravity="0 0 -9.81" density="1.225"/>
  
  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="800" offheight="600"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="50"/>
  </visual>
  
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  
  <worldbody>
    <!-- Lighting -->
    <light directional="true" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light directional="true" pos="5 5 10" dir="-1 -1 -1" diffuse="0.4 0.4 0.4"/>
    
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="{CONFIG['world_size']/2} {CONFIG['world_size']/2} 0.1" material="grid" rgba="0.2 0.3 0.4 1"/>
    
    <!-- Boundaries (invisible collision walls) -->
    <geom name="wall_north" type="box" size="{CONFIG['world_size']/2} 0.1 2" pos="0 {CONFIG['world_size']/2} 1" rgba="0.8 0.2 0.2 0.3"/>
    <geom name="wall_south" type="box" size="{CONFIG['world_size']/2} 0.1 2" pos="0 {-CONFIG['world_size']/2} 1" rgba="0.8 0.2 0.2 0.3"/>
    <geom name="wall_east" type="box" size="0.1 {CONFIG['world_size']/2} 2" pos="{CONFIG['world_size']/2} 0 1" rgba="0.8 0.2 0.2 0.3"/>
    <geom name="wall_west" type="box" size="0.1 {CONFIG['world_size']/2} 2" pos="{-CONFIG['world_size']/2} 0 1" rgba="0.8 0.2 0.2 0.3"/>
    
    <!-- Goal marker -->
    <geom name="goal" type="sphere" size="0.2" pos="{goal_pos[0]} {goal_pos[1]} {goal_pos[2]}" rgba="0.0 1.0 0.0 0.8" contype="0" conaffinity="0"/>

    <!-- UAV starting position -->
    <body name="chassis" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">
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
    
    # Add symbolic advice visualization markers
    xml_template += '''
    <!-- Symbolic advice visualization -->
    <geom name="advice_arrow" type="capsule" size="0.02 0.3" pos="0 0 -10" rgba="1 1 0 0.8" contype="0" conaffinity="0"/>
    <geom name="rule_indicator" type="sphere" size="0.1" pos="0 0 -10" rgba="0 0 1 0.7" contype="0" conaffinity="0"/>'''
    
    xml_template += '''
  </worldbody>

  <actuator>
    <motor name="m1" gear="0 0 1 0 0 0" site="motor1"/>
    <motor name="m2" gear="0 0 1 0 0 0" site="motor2"/>
    <motor name="m3" gear="0 0 1 0 0 0" site="motor3"/>
    <motor name="m4" gear="0 0 1 0 0 0" site="motor4"/>
  </actuator>

  <sensor>
    <accelerometer name="accel" site="motor1"/>
    <gyro name="gyro" site="motor1"/>
    <magnetometer name="mag" site="motor1"/>
  </sensor>
</mujoco>'''
    
    return xml_template

def render_neurosymbolic_uav(num_episodes=5, num_obstacles=9, load_checkpoint=True):
    """Render neurosymbolic UAV navigation with rule visualization"""
    
    # Initialize environment
    env = UAVEnv()
    base_state_dim = env.observation_space.shape[0]  # Base observation space
    state_dim = base_state_dim + 10  # +10 for symbolic features
    action_dim = env.action_space.shape[0]
    
    # Initialize neurosymbolic PPO agent
    agent = NeuroSymbolicPPOAgent(
        state_dim=base_state_dim,  # Base state dimension (36)
        action_dim=action_dim,
        lr_actor=0.0001,
        lr_critic=0.0004,
        gamma=0.999,
        K_epochs=12,
        eps_clip=0.1,
        integration_weight=0.15,
        knowledge_file="uav_navigation_rules.json"
    )
    
    # Get the integrator from the agent
    ns_integrator = agent.ns_integrator
    rdr_kb = agent.rdr_kb
    
    # Load neurosymbolic checkpoint
    if load_checkpoint:
        checkpoint_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_neurosymbolic.pth"
        
        if os.path.exists(checkpoint_path):
            try:
                agent.load(checkpoint_path)
                print(f"✅ Loaded neurosymbolic checkpoint: {checkpoint_path}")
                print(f"🧠 Neurosymbolic integration enabled with {len(rdr_kb.rules)} rules")
            except Exception as e:
                print(f"⚠️ Error loading neurosymbolic checkpoint: {e}")
                print("Using randomly initialized agent")
        else:
            print(f"⚠️ Neurosymbolic checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized agent")
    
    for episode in range(num_episodes):
        print(f"\n🎬 Episode {episode + 1}/{num_episodes}")
        
        # Use the existing environment model (from environment.xml)
        model = env.model
        data = env.data
        
        # Initialize viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 12
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 45
        viewer.cam.lookat[:] = [0, 0, 1]
        
        # Reset simulation data and set initial position
        mujoco.mj_resetData(model, data)
        
        # Reset environment state (use default environment reset)
        state = env.reset()[0]
        
        # Set UAV initial position in MuJoCo data (ensure it's within map bounds)
        from uav_env import CONFIG as ENV_CONFIG
        start_pos = ENV_CONFIG['start_pos']
        
        # Ensure start position is within visible bounds
        world_size = ENV_CONFIG.get('world_size', 8.0)
        half_world = world_size / 2 - 0.5  # Leave some margin
        start_pos[0] = max(-half_world, min(half_world, start_pos[0]))
        start_pos[1] = max(-half_world, min(half_world, start_pos[1]))
        start_pos[2] = ENV_CONFIG.get('uav_flight_height', 1.0)
        
        data.qpos[:3] = start_pos
        data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation (quaternion)
        data.qvel[:] = 0  # Zero initial velocity
        
        # Update goal position to be within bounds too
        goal_pos = ENV_CONFIG['goal_pos'].copy()
        goal_pos[0] = max(-half_world, min(half_world, goal_pos[0]))
        goal_pos[1] = max(-half_world, min(half_world, goal_pos[1]))
        goal_pos[2] = ENV_CONFIG.get('uav_flight_height', 1.0)
        
        # Get actual positions from environment after reset
        uav_pos = data.qpos[:3].copy()
        print(f"🎯 Start: [{uav_pos[0]:.1f}, {uav_pos[1]:.1f}, {uav_pos[2]:.1f}]")
        print(f"� Goal: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f}]")
        print(f"🧱 Using default environment obstacles")
        print(f"🗺️  Map bounds: ±{world_size/2:.1f}m")
        
        # Path tracking
        path_history = []
        trail_index = 0
        last_advice = "no_advice"
        rule_confidence = 0.0
        
        step_count = 0
        episode_reward = 0
        
        # Start message
        print("🚁 Mission started! Watch the neurosymbolic agent control the UAV!")
        print("🧠 Symbolic rules will provide guidance to the RL agent")
        print(f"🌍 UAV will fly from START to GOAL within ±{world_size/2:.1f}m bounds")
        distance_to_goal = np.linalg.norm(np.array(uav_pos[:2]) - np.array(goal_pos[:2]))
        print(f"📏 Distance to goal: {distance_to_goal:.2f}m")
        
        while True:
            # Get action from neurosymbolic agent
            action, _ = agent.select_action(state)
            
            # Get symbolic advice for visualization
            features = ns_integrator.extract_observation_features(state)
            rule_advice = rdr_kb.get_rule_advice(features)
            
            if rule_advice:
                last_advice = rule_advice[0].value
                rule_confidence = rule_advice[1]
            else:
                last_advice = "no_advice"
                rule_confidence = 0.0
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Get current UAV position from MuJoCo data (updated by env.step)
            uav_pos = data.qpos[:3].copy()
            
            # Apply action to MuJoCo simulation for visualization
            # Height control (maintain constant flight height)
            current_height = data.qpos[2]
            desired_height = ENV_CONFIG['uav_flight_height']
            height_error = desired_height - current_height
            vz_correction = 1.5 * height_error  # P-controller for height
            
            # Don't override X,Y velocities - let env.step() handle the action properly
            # The environment has already processed the action and updated the UAV position
            # Only control height for visualization stability
            data.qvel[2] = vz_correction  # Height controller only
            
            # Update path trail (only every few steps to avoid clutter)
            if step_count % 5 == 0:
                path_history.append(uav_pos.copy())
                if len(path_history) > 50:  # Keep last 50 positions
                    path_history.pop(0)
            
            # Update symbolic advice visualization
            advice_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'advice_arrow')
            rule_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'rule_indicator')
            
            if advice_geom_id >= 0 and rule_geom_id >= 0:
                if last_advice != "no_advice" and rule_confidence > 0.1:
                    # Show advice arrow above UAV
                    arrow_pos = uav_pos + np.array([0, 0, 0.5])
                    model.geom_pos[advice_geom_id] = arrow_pos
                    model.geom_rgba[advice_geom_id][3] = rule_confidence  # Alpha based on confidence
                    
                    # Show rule indicator
                    rule_pos = uav_pos + np.array([0.3, 0, 0.3])
                    model.geom_pos[rule_geom_id] = rule_pos
                    model.geom_rgba[rule_geom_id][3] = rule_confidence
                    
                    # Color based on advice type
                    if "avoid" in last_advice:
                        model.geom_rgba[advice_geom_id][:3] = [1, 0, 0]  # Red for avoidance
                    elif "goal" in last_advice:
                        model.geom_rgba[advice_geom_id][:3] = [0, 1, 0]  # Green for goal-seeking
                    else:
                        model.geom_rgba[advice_geom_id][:3] = [1, 1, 0]  # Yellow for other advice
                else:
                    # Hide indicators when no advice
                    model.geom_pos[advice_geom_id] = [0, 0, -10]
                    model.geom_pos[rule_geom_id] = [0, 0, -10]
            
            # Step simulation and ensure UAV visibility
            mujoco.mj_step(model, data)
            
            # Ensure UAV body remains visible (force red color)
            uav_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'uav_body')
            if uav_body_id >= 0:
                model.geom_rgba[uav_body_id] = [1.0, 0.0, 0.0, 1.0]  # Solid red
            
            viewer.sync()
            
            # Display info and trail
            if step_count % 20 == 0:  # Update display every 20 steps
                distance_to_goal = np.linalg.norm(np.array(uav_pos[:2]) - np.array(goal_pos[:2]))
                trail_length = len(path_history)
                print(f"Step {step_count:3d} | Pos: [{uav_pos[0]:5.2f}, {uav_pos[1]:5.2f}, {uav_pos[2]:4.2f}] | "
                      f"Goal dist: {distance_to_goal:.2f} | Rule: {last_advice:15s} | "
                      f"Confidence: {rule_confidence:.2f} | Trail pts: {trail_length:2d} | Reward: {episode_reward:6.1f}")
                
                # Print recent trail positions for visibility
                if len(path_history) >= 3:
                    recent_positions = path_history[-3:]
                    trail_str = " → ".join([f"[{p[0]:.1f},{p[1]:.1f}]" for p in recent_positions])
                    print(f"     🟢 Recent trail: {trail_str}")
                
                # Debug: Show goal direction and action
                goal_direction = np.array(goal_pos[:2]) - np.array(uav_pos[:2])
                goal_direction_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
                print(f"     🎯 Goal dir: [{goal_direction_norm[0]:5.2f}, {goal_direction_norm[1]:5.2f}] | Action: [{action[0]:.3f}, {action[1]:.3f}]")
            
            state = next_state
            step_count += 1
            
            # Control frame rate
            time.sleep(0.05)  # Slower for better visibility
            
            if done or truncated:
                if info.get('success', False):
                    print(f"🎉 SUCCESS! Reached goal in {step_count} steps")
                elif info.get('collision', False):
                    print(f"💥 COLLISION after {step_count} steps")
                elif info.get('out_of_bounds', False):
                    print(f"🚫 OUT OF BOUNDS after {step_count} steps")
                else:
                    print(f"⏰ TIMEOUT after {step_count} steps")
                
                print(f"📊 Final reward: {episode_reward:.2f}")
                print(f"🧠 Symbolic advice used: {agent.symbolic_advice_count} times")
                break
        
        # Wait before next episode
        print("Press SPACE to continue to next episode, or close viewer to exit...")
        
        # Wait for user input or viewer close
        while viewer.is_running():
            time.sleep(0.1)
            # You can add keyboard handling here if needed
        
        viewer.close()
        
        if not viewer.is_running():
            break
    
    print(f"\n🎬 Completed {episode + 1} episodes of neurosymbolic UAV navigation!")

if __name__ == "__main__":
    print("🚁 Neurosymbolic UAV Navigation Renderer")
    print("=" * 50)
    print("This renderer demonstrates the neurosymbolic UAV with rule visualization")
    print("🔵 Blue sphere: Active rule indicator")
    print("🟡 Yellow/Red/Green arrow: Symbolic advice direction")
    print("🟢 Green trail: UAV path history")
    print("=" * 50)
    
    # Parse command line arguments
    import sys
    
    num_episodes = 5
    num_obstacles = 9
    
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            print("Invalid number of episodes, using default: 5")
    
    if len(sys.argv) > 2:
        try:
            num_obstacles = int(sys.argv[2])
        except ValueError:
            print("Invalid number of obstacles, using default: 9")
    
    print(f"🎯 Running {num_episodes} episodes")
    print("🧠 Loading neurosymbolic model: PPO_UAV_Weights_neurosymbolic.pth")
    
    try:
        render_neurosymbolic_uav(num_episodes=num_episodes, num_obstacles=num_obstacles)
    except KeyboardInterrupt:
        print("\n⏹️ Rendering interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()