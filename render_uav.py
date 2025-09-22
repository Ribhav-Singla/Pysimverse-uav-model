#!/usr/bin/env python3
"""
UAV Environment Renderer
Visualizes the UAV navigation environment with MuJoCo viewer
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from uav_config import EnvironmentGenerator, CONFIG

def main():
    """Render the UAV environment with MuJoCo viewer"""
    print("🏗️ Generating UAV environment...")
    
    # Generate environment with obstacles
    obstacles = EnvironmentGenerator.create_xml_with_obstacles()
    
    # Load the model
    model = mujoco.MjModel.from_xml_path("uav_model.xml")
    data = mujoco.MjData(model)
    
    print(f"🚁 UAV Environment Loaded!")
    print(f"📊 Model: {model.nu} actuators, {model.nbody} bodies")
    print(f"🎯 Mission: Navigate from START (green) to GOAL (blue)")
    print(f"🚧 Static obstacles: {CONFIG['static_obstacles']}")
    print(f"📏 Obstacle height: {CONFIG['obstacle_height']}m")
    print(f"✈️ UAV flight height: {CONFIG['uav_flight_height']}m")
    print("\n🎮 Controls:")
    print("- Close viewer window to exit")
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom")
    print("- Right click + drag: Pan view")
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset simulation to initial state
        mujoco.mj_resetData(model, data)
        
        # Set UAV initial position
        data.qpos[:3] = CONFIG['start_pos']
        data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation (quaternion)
        data.qvel[:] = 0  # Zero initial velocities
        
        print("\n🌟 Environment rendered! UAV is positioned at the start (green marker).")
        print("💡 The goal is the blue marker. Red obstacles are scattered throughout.")
        
        # Keep the viewer running
        step_count = 0
        while viewer.is_running():
            # Apply small hover thrust to keep UAV at current position
            data.ctrl[:] = [2.0, 2.0, 2.0, 2.0]  # Light hover thrust
            
            # Step physics simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Small delay
            time.sleep(0.01)
            step_count += 1
            
            # Print status every few seconds
            if step_count % 500 == 0:
                current_pos = data.qpos[:3]
                print(f"UAV Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
    
    print("👋 Viewer closed. Rendering finished.")

if __name__ == "__main__":
    main()