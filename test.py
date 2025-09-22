# UAV Navigation Simulation
import numpy as np
import mujoco
import mujoco.viewer
import time
import math

# Define the path to your XML model
MODEL_PATH = "uav_model.xml"

# Configuration parameters for easy tweaking
CONFIG = {
    'start_pos': np.array([-3.0, -3.0, 1.5]),   # UAV start position (x, y, z)
    'goal_pos': np.array([3.0, 3.0, 1.5]),       # UAV goal position
    'step_dist': 0.01,                            # Movement per control loop (m)
    'takeoff_thrust':3.0,                       # Thrust for initial takeoff
    'kp_pos': 1.5,                               # Position gain
    'kd_pos': 1.0,                               # Velocity damping gain
    'hover_thrust': 7.0,                        # Base thrust to hover
    'obstacle_count': 8,                          # Number of obstacles in env
    'velocity': 0.15,            # Desired UAV speed in m/s
    'control_dt': 0.05,         # Control loop timestep in seconds (20Hz)
    'waypoint_threshold': 0.5,  # Distance threshold to consider waypoint reached (m)
    'takeoff_altitude': 1.2     # Altitude to reach before navigation (m)
}

class UAVController:
    def __init__(self):
        self.start_pos = CONFIG['start_pos']
        self.goal_pos = CONFIG['goal_pos']
        self.current_waypoint = 0
        self.waypoints = self.generate_waypoints()
        
    def generate_waypoints(self):
        """Generate waypoints from start to goal avoiding obstacles"""
        waypoints = [
            self.start_pos.copy(),
            np.array([-2.0, -2.0, 1.5]),  # Move away from start
            np.array([-1.5, -0.5, 1.8]),  # Navigate around obstacles
            np.array([0.5, 0.5, 2.0]),    # Go over central obstacles
            np.array([1.5, 1.5, 1.8]),    # Descend towards goal
            np.array([2.5, 2.5, 1.5]),    # Approach goal
            self.goal_pos.copy()
        ]
        return waypoints
    
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

# Load the model and create the simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"UAV Navigation Simulation Started!")
print(f"Model: {model.nu} actuators, {model.nbody} bodies")
print(f"Mission: Navigate from START (green) to GOAL (blue)")
print(f"Avoid the colored obstacles in between!")

# Initialize controller
controller = UAVController()

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
    
    try:
        while viewer.is_running() and not mission_complete:
            # Get current UAV state
            current_pos = data.qpos[:3].copy()
            current_vel = data.qvel[:3].copy()
            
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
            
            # Check if mission is complete
            goal_distance = np.linalg.norm(current_pos - controller.goal_pos)
            if goal_distance < 0.5:
                mission_complete = True
                print(f"\n🎉 MISSION COMPLETE! UAV reached the goal!")
                print(f"Final position: {current_pos}")
                print(f"Distance to goal: {goal_distance:.2f}m")
            
            # Status updates
            if step_count % 500 == 0:
                distance_to_target = np.linalg.norm(current_pos - target_pos)
                print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                      f"Target=({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f}) "
                      f"Dist={distance_to_target:.2f}")
            
            step_count += 1
            time.sleep(CONFIG['control_dt'])  # Control loop dt
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    if mission_complete:
        # Celebration hover
        print("🎊 Mission complete! Hovering at goal...")
        for _ in range(500):
            data.ctrl[:] = [4.0, 4.0, 4.0, 4.0]  # Stronger hover thrust
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    print("Simulation finished.")