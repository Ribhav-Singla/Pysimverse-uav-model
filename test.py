# UAV Navigation Simulation
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
import random
from xml.etree import ElementTree as ET

# Define the path to your XML model
MODEL_PATH = "uav_model.xml"

# Configuration parameters for easy tweaking
CONFIG = {
    'start_pos': np.array([-4.0, -4.0, 1.8]),   # UAV start position (x, y, z)
    'goal_pos': np.array([4.0, 4.0, 1.8]),       # UAV goal position
    'step_dist': 0.01,                            # Movement per control loop (m)
    'takeoff_thrust': 3.0,                        # Thrust for initial takeoff
    'kp_pos': 1.5,                               # Position gain
    'kd_pos': 1.0,                               # Velocity damping gain
    'hover_thrust': 7.0,                         # Base thrust to hover
    'velocity': 0.15,                            # Desired UAV speed in m/s
    'control_dt': 0.05,                          # Control loop timestep in seconds (20Hz)
    'waypoint_threshold': 0.5,                   # Distance threshold to consider waypoint reached (m)
    'takeoff_altitude': 1.5,                     # Altitude to reach before navigation (m)
    
    # Environment parameters
    'world_size': 8.0,                           # World boundary size (8x8 grid)
    'obstacle_height': 2.0,                      # Fixed height for all obstacles
    'uav_flight_height': 1.8,                   # UAV flies below obstacle tops
    'static_obstacles': 8,                       # Number of static obstacles
    'dynamic_obstacles': 0,                      # Number of dynamic obstacles
    'min_obstacle_size': 0.2,                    # Minimum obstacle dimension
    'max_obstacle_size': 0.6,                    # Maximum obstacle dimension
    'dynamic_speed': 0.1,                        # Speed of dynamic obstacles (m/s)
    'collision_distance': 0.2,                   # Collision threshold (m)
    'path_trail_length': 200,                    # Number of points to keep in the path trail
}

class EnvironmentGenerator:
    @staticmethod
    def generate_obstacles():
        """Generate both static and dynamic obstacles with random non-overlapping positions"""
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        # Minimum separation distance between obstacle centers
        min_separation = 1.2  # meters
        
        def check_collision_with_existing(new_pos, new_size, existing_obstacles):
            """Check if a new obstacle would overlap with existing ones"""
            for obs in existing_obstacles:
                obs_pos = np.array(obs['pos'][:2])  # Only x,y coordinates
                distance = np.linalg.norm(np.array(new_pos[:2]) - obs_pos)
                
                # Calculate required separation based on obstacle sizes
                if obs['shape'] == 'box':
                    obs_radius = max(obs['size'][:2])  # Use max dimension as radius
                elif obs['shape'] == 'cylinder':
                    obs_radius = obs['size'][0]  # Cylinder radius
                else:
                    obs_radius = obs['size'][0]  # Sphere radius
                
                if isinstance(new_size, list) and len(new_size) >= 2:
                    new_radius = max(new_size[:2])  # For box or cylinder
                else:
                    new_radius = new_size  # For sphere
                
                required_separation = obs_radius + new_radius + min_separation
                
                if distance < required_separation:
                    return True
            return False
        
        def is_near_start_or_goal(pos):
            """Check if position is too close to start or goal"""
            start_dist = np.linalg.norm(np.array(pos[:2]) - np.array(CONFIG['start_pos'][:2]))
            goal_dist = np.linalg.norm(np.array(pos[:2]) - np.array(CONFIG['goal_pos'][:2]))
            return start_dist < 2.0 or goal_dist < 2.0
        
        # Generate static obstacles with random placement
        for i in range(CONFIG['static_obstacles']):
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts:
                # Random position within world bounds
                x = random.uniform(-half_world + 1, half_world - 1)
                y = random.uniform(-half_world + 1, half_world - 1)
                pos = [x, y, CONFIG['obstacle_height']/2]
                
                # Skip if too close to start/goal
                if is_near_start_or_goal(pos):
                    attempts += 1
                    continue
                
                # Random size and shape
                size_x = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
                size_y = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
                height = CONFIG['obstacle_height']
                
                obstacle_type = random.choice(['box', 'cylinder'])
                size = [size_x, size_y, height/2] if obstacle_type == 'box' else [min(size_x, size_y), height/2]
                
                # Check for collisions with existing obstacles
                if not check_collision_with_existing(pos, size, obstacles):
                    color = [random.uniform(0.1, 0.9) for _ in range(3)] + [1.0]
                    
                    obstacles.append({
                        'type': 'static',
                        'shape': obstacle_type,
                        'pos': pos,
                        'size': size,
                        'color': color,
                        'id': f'static_obs_{i}'
                    })
                    break
                
                attempts += 1
            
            # If we couldn't place after max attempts, place it anyway but warn
            if attempts >= max_attempts:
                print(f"Warning: Could not find non-overlapping position for obstacle {i}")
        
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
      <geom type="box" size="0.12 0.12 0.02" rgba="0.8 0.2 0.2 1" mass="0.8"/>
      
      <!-- Propeller arms and motors -->
      <geom type="box" size="0.08 0.01 0.005" pos="0 0 0.01" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" size="0.01 0.08 0.005" pos="0 0 0.01" rgba="0.3 0.3 0.3 1"/>
      
      <!-- Motor visual geometry -->
      <geom type="cylinder" size="0.015 0.02" pos="0.08 0.08 0.015" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.015 0.02" pos="-0.08 0.08 0.015" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.015 0.02" pos="0.08 -0.08 0.015" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.015 0.02" pos="-0.08 -0.08 0.015" rgba="0.2 0.2 0.2 1"/>
      
      <!-- Propellers -->
      <geom type="cylinder" size="0.04 0.002" pos="0.08 0.08 0.035" rgba="0.7 0.7 0.7 0.6"/>
      <geom type="cylinder" size="0.04 0.002" pos="-0.08 0.08 0.035" rgba="0.7 0.7 0.7 0.6"/>
      <geom type="cylinder" size="0.04 0.002" pos="0.08 -0.08 0.035" rgba="0.7 0.7 0.7 0.6"/>
      <geom type="cylinder" size="0.04 0.002" pos="-0.08 -0.08 0.035" rgba="0.7 0.7 0.7 0.6"/>
      
      <!-- Sites for motor force application -->
      <site name="motor1" pos="0.08 0.08 0" size="0.01"/>
      <site name="motor2" pos="-0.08 0.08 0" size="0.01"/>
      <site name="motor3" pos="0.08 -0.08 0" size="0.01"/>
      <site name="motor4" pos="-0.08 -0.08 0" size="0.01"/>
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
        
        # Add path trail sites (initially invisible)
        for i in range(CONFIG['path_trail_length']):
            xml_template += f'''
    <site name="path_{i}" type="sphere" size="0.03" pos="0 0 0" rgba="0 0 0 0"/>'''
        
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

class UAVController:
    def __init__(self):
        self.start_pos = CONFIG['start_pos']
        self.goal_pos = CONFIG['goal_pos']
        self.current_waypoint = 0
        self.waypoints = self.generate_waypoints()
        self.dynamic_obstacles = []
        self.all_obstacles = []
        self.path_history = []  # Store UAV position history for path visualization
        self.path_index = 0     # Current index in the path trail
        
    def generate_waypoints(self):
        """Generate adaptive waypoints from start to goal"""
        flight_height = CONFIG['uav_flight_height']
        waypoints = [
            self.start_pos.copy(),
            np.array([-2.0, -2.0, flight_height]),  # Initial climb
            np.array([-1.0, -1.0, flight_height]),  # Navigate around obstacles
            np.array([0.0, 0.0, flight_height]),    # Center waypoint
            np.array([1.0, 1.0, flight_height]),    # Continue navigation
            np.array([2.0, 2.0, flight_height]),    # Approach goal area
            self.goal_pos.copy()
        ]
        return waypoints
    
    def update_dynamic_obstacles(self, obstacles, dt):
        """Update positions of dynamic obstacles"""
        world_bound = CONFIG['world_size'] / 2
        
        for obs in obstacles:
            if obs['type'] == 'dynamic':
                # Update position
                obs['pos'][0] += obs['velocity'][0] * dt
                obs['pos'][1] += obs['velocity'][1] * dt
                
                # Bounce off boundaries
                if obs['pos'][0] <= -world_bound + 0.5 or obs['pos'][0] >= world_bound - 0.5:
                    obs['velocity'][0] *= -1
                if obs['pos'][1] <= -world_bound + 0.5 or obs['pos'][1] >= world_bound - 0.5:
                    obs['velocity'][1] *= -1
                
                # Keep within bounds
                obs['pos'][0] = np.clip(obs['pos'][0], -world_bound + 0.5, world_bound - 0.5)
                obs['pos'][1] = np.clip(obs['pos'][1], -world_bound + 0.5, world_bound - 0.5)
    
    def update_path_trail(self, model, data, current_pos):
        """Update the path trail visualization"""
        # Add current position to history
        self.path_history.append(current_pos.copy())
        
        # Keep only the most recent positions
        if len(self.path_history) > CONFIG['path_trail_length']:
            self.path_history.pop(0)
        
        # Update the path sites in the model
        for i, pos in enumerate(self.path_history):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"path_{i}")
            if site_id != -1:
                # Calculate color based on position in trail (fade from yellow to red)
                alpha = 0.7 * (i / len(self.path_history))
                color = [1.0, 1.0 - alpha, 0.0, 0.7]  # Yellow to red gradient
                
                # Update site position and color
                data.site_xpos[site_id] = pos
                model.site_rgba[site_id] = color
    
    def get_target_position(self, current_pos):
        """Get the current target waypoint"""
        if self.current_waypoint < len(self.waypoints):
            target = self.waypoints[self.current_waypoint]
            
            # Check if we've reached the current waypoint
            distance = np.linalg.norm(current_pos - target)
            if distance < CONFIG['waypoint_threshold']:  # Within waypoint threshold
                self.current_waypoint += 1
                print(f"Reached waypoint {self.current_waypoint}!")
                if self.current_waypoint < len(self.waypoints):
                    target = self.waypoints[self.current_waypoint]
                    
            return target
        else:
            return self.goal_pos
    
    def calculate_control(self, current_pos, current_vel, target_pos):
        """Calculate motor controls to move towards target"""
        # PD controller parameters from CONFIG
        kp_pos = CONFIG['kp_pos']
        kd_pos = CONFIG['kd_pos']
        hover_thrust = CONFIG['hover_thrust']
        
        # Calculate position error
        pos_error = target_pos - current_pos
        
        # Calculate desired forces
        force_x = kp_pos * pos_error[0] - kd_pos * current_vel[0]
        force_y = kp_pos * pos_error[1] - kd_pos * current_vel[1] 
        force_z = kp_pos * pos_error[2] - kd_pos * current_vel[2] + hover_thrust
        
        # Convert forces to motor thrusts (simplified quadcopter mixing)
        base_thrust = np.clip(force_z / 4, 2.0, 8.0)  # Higher minimum thrust
        
        # Add control for x, y movement (stronger control)
        motor_controls = np.array([
            base_thrust + 0.3 * force_x + 0.3 * force_y,   # Front-right
            base_thrust - 0.3 * force_x + 0.3 * force_y,   # Front-left  
            base_thrust + 0.3 * force_x - 0.3 * force_y,   # Back-right
            base_thrust - 0.3 * force_x - 0.3 * force_y    # Back-left
        ])
        
        return np.clip(motor_controls, 1.0, 10.0)  # Higher thrust range

# Generate complex environment
print("🏗️ Generating complex environment with dynamic obstacles...")
obstacles = EnvironmentGenerator.create_xml_with_obstacles()

# Load the model and create the simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"🚁 Complex UAV Navigation Environment Loaded!")
print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
print(f"🎯 Mission: Navigate from START (green) to GOAL (blue)")
print(f"🚧 Static obstacles: {CONFIG['static_obstacles']}")
print(f"🔴 Dynamic obstacles: {CONFIG['dynamic_obstacles']}")
print(f"📏 Obstacle height: {CONFIG['obstacle_height']}m")
print(f"✈️ UAV flight height: {CONFIG['uav_flight_height']}m")
print(f"🛣️ Path trail: {CONFIG['path_trail_length']} points")

# Initialize controller
controller = UAVController()
controller.dynamic_obstacles = [obs for obs in obstacles if obs['type'] == 'dynamic']
controller.all_obstacles = obstacles

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position
    data.qpos[:3] = controller.start_pos
    data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation (quaternion)
    
    print("Mission started! Watch the red UAV navigate to the blue goal!")
    time.sleep(2)
    
    step_count = 0
    mission_complete = False
    takeoff_complete = False
    collision_occurred = False
    
    try:
        while viewer.is_running() and not mission_complete and not collision_occurred:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
            # Update path trail visualization
            if step_count % 5 == 0:  # Update trail every 5 steps
                controller.update_path_trail(model, data, current_pos)
            
            # Takeoff phase - apply strong upward thrust first
            if not takeoff_complete and current_pos[2] < CONFIG['takeoff_altitude']:
                motor_controls = np.array([CONFIG['takeoff_thrust']] * model.nu)
                if step_count % 100 == 0:
                    print(f"Taking off... Height: {current_pos[2]:.2f}m")
            else:
                if not takeoff_complete:
                    takeoff_complete = True
                    print("✈️ Takeoff complete! Starting navigation...")
                
                # Get target position from controller
                target_pos = controller.get_target_position(current_pos)
                
                # Calculate motor controls
                motor_controls = controller.calculate_control(current_pos, current_vel, target_pos)
            
            # Update dynamic obstacles
            controller.update_dynamic_obstacles(controller.dynamic_obstacles, CONFIG['control_dt'])
            
            # Update dynamic obstacle positions in simulation
            for i, obs in enumerate(controller.dynamic_obstacles):
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obs['id'])
                if geom_id >= 0:
                    model.geom_pos[geom_id][:3] = obs['pos']
            
            # Manually update UAV position for visualization
            target_pos = controller.get_target_position(current_pos)
            # Compute direction and step towards target
            pos_error = target_pos - current_pos
            dist = np.linalg.norm(pos_error)
            if dist > 0.01:
                step_dist = CONFIG['velocity'] * CONFIG['control_dt']
                direction = pos_error / dist
                new_pos = current_pos + direction * min(step_dist, dist)
                data.qpos[:3] = new_pos
            
            # Forward model to update rendering
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            # Check for collision
            has_collision, obstacle_id, collision_dist = EnvironmentGenerator.check_collision(data.qpos[:3], controller.all_obstacles)
            if has_collision:
                collision_occurred = True
                print(f"\n💥 COLLISION DETECTED!")
                print(f"💀 UAV crashed into obstacle: {obstacle_id}")
                print(f"📏 Collision distance: {collision_dist:.3f}m")
                print(f"📍 UAV position at crash: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
                print(f"⚠️ MISSION FAILED!")
                break
            
            # Check if mission is complete
            goal_distance = np.linalg.norm(current_pos - controller.goal_pos)
            if goal_distance < 0.5:
                mission_complete = True
                print(f"\n🎉 MISSION COMPLETE! UAV reached the goal!")
                print(f"Final position: {current_pos}")
                print(f"Distance to goal: {goal_distance:.2f}m")
                print(f"🏆 SUCCESS! No collisions occurred!")
            
            # Status updates
            if step_count % 500 == 0:
                distance_to_target = np.linalg.norm(current_pos - target_pos)
                # Find closest obstacle distance
                min_obstacle_dist = float('inf')
                for obs in controller.all_obstacles:
                    dist = np.linalg.norm(current_pos - np.array(obs['pos']))
                    min_obstacle_dist = min(min_obstacle_dist, dist)
                
                print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                      f"Target=({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f}) "
                      f"Dist={distance_to_target:.2f} | Closest obs: {min_obstacle_dist:.2f}m")
            
            step_count += 1
            time.sleep(CONFIG['control_dt'])  # Control loop dt
            
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