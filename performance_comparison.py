#!/usr/bin/env python3
"""
UAV Performance Comparison Script
Compares performance across three approaches:
1. Human Expert (Manual Control)
2. Neural Only (PPO with lambda=0.0)
3. Neurosymbolic (PPO with lambda=1.0)

Tests across curriculum levels 1-10 with increasing obstacle counts.
"""

import numpy as np
import torch
import time
import threading
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict
import mujoco
import mujoco.viewer

# Try to import keyboard for manual control
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: 'keyboard' module not available. Install with: pip install keyboard")
    print("Manual control trials will be disabled.")
    KEYBOARD_AVAILABLE = False

# Import custom modules
from uav_env import UAVEnv, CONFIG, EnvironmentGenerator
from ppo_agent import PPOAgent
from uav_agent_runner import UAVAgentRunner

class PerformanceTracker:
    """Track performance metrics for each approach"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new trial"""
        self.start_time = None
        self.end_time = None
        self.path_length = 0.0
        self.step_count = 0
        self.success = False
        self.collision = False
        self.out_of_bounds = False
        self.timeout = False
        self.final_distance = None
        self.path_positions = []
        self.max_steps = CONFIG['max_steps']
    
    def start_trial(self, start_pos):
        """Start tracking a new trial"""
        self.reset()
        self.start_time = time.time()
        self.path_positions = [start_pos.copy()]
    
    def update_step(self, position, goal_pos):
        """Update metrics for each step"""
        if len(self.path_positions) > 0:
            step_distance = np.linalg.norm(position - self.path_positions[-1])
            self.path_length += step_distance
        
        self.path_positions.append(position.copy())
        self.step_count += 1
        self.final_distance = np.linalg.norm(position - goal_pos)
    
    def end_trial(self, success=False, collision=False, out_of_bounds=False, timeout=False):
        """End the trial and calculate final metrics"""
        self.end_time = time.time()
        self.success = success
        self.collision = collision
        self.out_of_bounds = out_of_bounds
        self.timeout = timeout
    
    def get_metrics(self):
        """Get performance metrics dictionary"""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            'path_length': self.path_length,
            'step_count': self.step_count,
            'success': self.success,
            'collision': self.collision,
            'out_of_bounds': self.out_of_bounds,
            'timeout': self.timeout,
            'final_distance': self.final_distance,
            'duration': duration,
            'path_efficiency': self.calculate_path_efficiency()
        }
    
    def calculate_path_efficiency(self):
        """Calculate path efficiency (direct distance / actual path length)"""
        if len(self.path_positions) < 2 or self.path_length == 0:
            return 0.0
        
        direct_distance = np.linalg.norm(self.path_positions[-1] - self.path_positions[0])
        return direct_distance / self.path_length if self.path_length > 0 else 0.0

class IntegratedManualController:
    """Integrated manual control system for human expert trials within the comparison environment"""
    
    def __init__(self):
        self.target_velocity = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.active = False
        self.manual_speed = 0.8
        self.manual_acceleration = 0.1
        self.reset_requested = False
        self.exit_requested = False
        self.path_history = []
        self.trial_complete = False
        self.collision_occurred = False
        self.goal_reached = False
    
    def start_control(self):
        """Start manual control session"""
        self.active = True
        self.reset_requested = False
        self.exit_requested = False
        self.trial_complete = False
        self.collision_occurred = False
        self.goal_reached = False
        self.path_history = []
        self.print_controls()
    
    def stop_control(self):
        """Stop manual control session"""
        self.active = False
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print("🎮 INTEGRATED MANUAL CONTROL ACTIVE")
        print("="*60)
        print("MOVEMENT: Arrow Keys (↑↓←→)")
        print("STOP: SPACE")
        print("RESET: R")
        print("EXIT TRIAL: ESC")
        print("Navigate from START (green) to GOAL (blue)")
        print("="*60)
    
    def update_controls(self):
        """Update target velocity based on keyboard input with smooth acceleration"""
        if not self.active or not KEYBOARD_AVAILABLE:
            return np.array([0.0, 0.0])
        
        target_vel = np.array([0.0, 0.0])
        
        try:
            if keyboard.is_pressed('up'):
                target_vel[1] += self.manual_speed  # Forward
            if keyboard.is_pressed('down'):
                target_vel[1] -= self.manual_speed  # Backward
            if keyboard.is_pressed('left'):
                target_vel[0] -= self.manual_speed  # Left
            if keyboard.is_pressed('right'):
                target_vel[0] += self.manual_speed  # Right
            if keyboard.is_pressed('space'):
                target_vel = np.array([0.0, 0.0])  # Stop
            if keyboard.is_pressed('r'):
                self.reset_requested = True
            if keyboard.is_pressed('esc'):
                self.exit_requested = True
        except:
            # Handle potential keyboard module issues
            pass
        
        self.target_velocity = target_vel
        
        # Smooth velocity transition for realistic control
        vel_diff = self.target_velocity - self.current_velocity
        self.current_velocity += vel_diff * self.manual_acceleration
        
        return self.current_velocity.copy()
    
    def update_path_trail(self, model, current_pos):
        """Update the path trail visualization"""
        self.path_history.append(current_pos.copy())
        if len(self.path_history) > 500:  # Limit trail length
            self.path_history.pop(0)
        
        # Update trail geometries if they exist in the model
        trail_count = len(self.path_history)
        total_geoms = model.ngeom
        
        # Find trail geometries and update them
        for i in range(min(trail_count, 500)):
            # Look for trail geometries (they should be at the end of the geom list)
            geom_idx = total_geoms - 500 + i
            if 0 <= geom_idx < total_geoms:
                try:
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx) or ""
                    if "trail_" in geom_name and i < trail_count:
                        model.geom_pos[geom_idx] = self.path_history[i]
                        alpha = 0.3 + 0.5 * (i / max(1, trail_count))
                        model.geom_rgba[geom_idx] = [0.0, 1.0, 0.0, alpha]
                except:
                    pass  # Skip if geometry access fails
    
    def ensure_uav_visibility(self, model):
        """Ensure UAV remains visible by protecting its colors"""
        for i in range(model.ngeom):
            try:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if "uav_body" in geom_name:
                    model.geom_rgba[i] = [1.0, 0.0, 0.0, 1.0]  # Red UAV body
                elif "prop" in geom_name:
                    model.geom_rgba[i] = [0.0, 0.0, 0.0, 1.0]  # Black propellers
            except:
                pass
    
    def ensure_markers_visibility(self, model):
        """Ensure start/goal markers remain visible"""
        for i in range(model.ngeom):
            try:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if "start_marker" in geom_name:
                    model.geom_rgba[i] = [0.0, 1.0, 0.0, 0.8]  # Green start marker
                elif "start_pole" in geom_name:
                    model.geom_rgba[i] = [0.0, 1.0, 0.0, 1.0]  # Green start pole
                elif "goal_marker" in geom_name:
                    model.geom_rgba[i] = [0.0, 0.0, 1.0, 0.8]  # Blue goal marker
                elif "goal_pole" in geom_name:
                    model.geom_rgba[i] = [0.0, 0.0, 1.0, 1.0]  # Blue goal pole
            except:
                pass

# Helper functions for LIDAR and collision detection
def get_lidar_readings(pos, obstacles):
    """Generate LIDAR readings in 360 degrees around the UAV"""
    lidar_readings = []
    lidar_range = CONFIG.get('lidar_range', 2.9)
    lidar_num_rays = CONFIG.get('lidar_num_rays', 16)
    
    for i in range(lidar_num_rays):
        angle = (2 * np.pi * i) / lidar_num_rays
        ray_dir = np.array([np.cos(angle), np.sin(angle), 0])
        
        min_distance = lidar_range
        
        # Check boundary intersection
        boundary_dist = ray_boundary_intersection(pos, ray_dir)
        if boundary_dist < min_distance:
            min_distance = boundary_dist
        
        # Check obstacle intersections
        for obs in obstacles:
            obs_dist = ray_obstacle_intersection(pos, ray_dir, obs)
            if obs_dist < min_distance:
                min_distance = obs_dist
        
        # Normalize LIDAR reading to [0, 1] range
        normalized_distance = min_distance / lidar_range
        lidar_readings.append(normalized_distance)
    
    return np.array(lidar_readings)

def ray_boundary_intersection(pos, ray_dir):
    """Calculate intersection of ray with world boundaries"""
    half_world = CONFIG['world_size'] / 2
    lidar_range = CONFIG.get('lidar_range', 2.9)
    min_dist = lidar_range
    
    boundaries = [
        (half_world, np.array([1, 0, 0])),
        (-half_world, np.array([-1, 0, 0])),
        (half_world, np.array([0, 1, 0])),
        (-half_world, np.array([0, -1, 0]))
    ]
    
    for boundary_pos, boundary_normal in boundaries:
        denominator = np.dot(ray_dir, boundary_normal)
        if abs(denominator) > 1e-6:
            if boundary_normal[0] != 0:
                t = (boundary_pos - pos[0]) / ray_dir[0]
            else:
                t = (boundary_pos - pos[1]) / ray_dir[1]
            
            if t > 0:
                intersection = pos + t * ray_dir
                if (-half_world <= intersection[0] <= half_world and 
                    -half_world <= intersection[1] <= half_world):
                    min_dist = min(min_dist, t)
    
    return min_dist

def ray_obstacle_intersection(pos, ray_dir, obstacle):
    """Calculate intersection of ray with obstacle"""
    obs_pos = np.array(obstacle['pos'])
    lidar_range = CONFIG.get('lidar_range', 2.9)
    min_dist = lidar_range
    
    if obstacle['shape'] == 'box':
        to_obs = obs_pos - pos
        proj_length = np.dot(to_obs, ray_dir)
        
        if proj_length > 0:
            closest_point = pos + proj_length * ray_dir
            lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])
            
            if lateral_dist < max(obstacle['size'][0], obstacle['size'][1]):
                surface_dist = max(0, proj_length - max(obstacle['size'][0], obstacle['size'][1]))
                min_dist = min(min_dist, surface_dist)
    
    elif obstacle['shape'] == 'cylinder':
        to_obs = obs_pos - pos
        proj_length = np.dot(to_obs, ray_dir)
        
        if proj_length > 0:
            closest_point = pos + proj_length * ray_dir
            lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])
            
            if lateral_dist < obstacle['size'][0]:
                surface_dist = max(0, proj_length - obstacle['size'][0])
                min_dist = min(min_dist, surface_dist)
    
    return min_dist

def check_collision_precise(uav_pos, obstacles):
    """Check if UAV collides with any obstacle with precise threshold"""
    collision_distance = CONFIG.get('collision_distance', 0.1)
    
    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        
        if obs['shape'] == 'box':
            # Calculate minimum distance to box faces
            dx = max(obs_pos[0] - obs['size'][0] - uav_pos[0], 
                     uav_pos[0] - (obs_pos[0] + obs['size'][0]), 0)
            dy = max(obs_pos[1] - obs['size'][1] - uav_pos[1], 
                     uav_pos[1] - (obs_pos[1] + obs['size'][1]), 0)
            dz = max(obs_pos[2] - obs['size'][2] - uav_pos[2], 
                     uav_pos[2] - (obs_pos[2] + obs['size'][2]), 0)
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
        elif obs['shape'] == 'cylinder':
            # Horizontal and vertical distances
            horizontal_dist = np.sqrt((uav_pos[0]-obs_pos[0])**2 + (uav_pos[1]-obs_pos[1])**2)
            vertical_dist = max(0, abs(uav_pos[2]-obs_pos[2]) - obs['size'][1])
            
            if horizontal_dist <= obs['size'][0] and vertical_dist == 0:
                distance = 0  # Inside cylinder
            else:
                distance = max(horizontal_dist - obs['size'][0], vertical_dist)
        
        if distance < collision_distance:
            return True, obs['id'], distance
    
    return False, None, float('inf')

class ComparisonEnvironment:
    """Environment wrapper for running comparison trials"""
    
    def __init__(self, level=1):
        self.level = level
        self.obstacles = []
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment for the current level with deterministic obstacle generation"""
        # FIXED: Use level-based seed for reproducible obstacle generation
        # This ensures the same level always generates identical obstacles
        import random
        random.seed(1000 + self.level)  # Deterministic seed based on level
        np.random.seed(1000 + self.level)  # Also set numpy seed
        
        # Generate obstacles for current level (now deterministic)
        self.obstacles = EnvironmentGenerator.generate_obstacles(self.level)
        
        # Generate safe start and goal positions (also deterministic)
        self.generate_safe_positions()
        
        # Store positions for reuse across trials
        self.start_pos = CONFIG['start_pos'].copy()
        self.goal_pos = CONFIG['goal_pos'].copy()
        
        # Create XML file
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
    
    def generate_safe_positions(self):
        """Generate safe start and goal positions for the current level (deterministic)"""
        # Use similar logic as in UAVEnv but make it deterministic per level
        half_world = CONFIG['world_size'] / 2 - 1.0
        
        # Deterministic corner selection based on level
        corners = [
            np.array([-half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([-half_world, half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, half_world, CONFIG['uav_flight_height']])
        ]
        
        # Use deterministic start corner (level-based)
        start_corner_idx = self.level % len(corners)
        CONFIG['start_pos'] = corners[start_corner_idx].copy()
        
        # Use deterministic goal corner (different from start, level-based)
        available_goal_corners = [c for i, c in enumerate(corners) if i != start_corner_idx]
        goal_corner_idx = (self.level + 1) % len(available_goal_corners)
        CONFIG['goal_pos'] = available_goal_corners[goal_corner_idx].copy()
        
        print(f"📍 Level {self.level}: Start {CONFIG['start_pos'][:2]}, Goal {CONFIG['goal_pos'][:2]} (deterministic)")
    
    def reset_uav(self):
        """Reset UAV to start position"""
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']
        self.data.qvel[:] = 0
    
    def check_collision(self, pos):
        """Check if UAV position collides with obstacles"""
        for obs in self.obstacles:
            obs_pos = np.array(obs['pos'])
            
            if obs['shape'] == 'box':
                # Simple box collision check
                if (abs(pos[0] - obs_pos[0]) < obs['size'][0] + 0.1 and
                    abs(pos[1] - obs_pos[1]) < obs['size'][1] + 0.1 and
                    abs(pos[2] - obs_pos[2]) < obs['size'][2] + 0.1):
                    return True
            elif obs['shape'] == 'cylinder':
                # Cylinder collision check
                horizontal_dist = np.linalg.norm(pos[:2] - obs_pos[:2])
                if (horizontal_dist < obs['size'][0] + 0.1 and
                    abs(pos[2] - obs_pos[2]) < obs['size'][1] + 0.1):
                    return True
        
        return False
    
    def check_bounds(self, pos):
        """Check if UAV is within world boundaries"""
        half_world = CONFIG['world_size'] / 2
        return (abs(pos[0]) > half_world or abs(pos[1]) > half_world)
    
    def check_goal_reached(self, pos):
        """Check if UAV reached the goal"""
        distance_to_goal = np.linalg.norm(pos - CONFIG['goal_pos'])
        return distance_to_goal < 0.3  # 30cm tolerance

class PerformanceComparison:
    """Main comparison class"""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.check_model_files()
    
    def check_model_files(self):
        """Check which model files are available"""
        print("🤖 Checking available PPO models...")
        
        self.neural_model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth"
        self.neurosymbolic_model_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_1_0.pth"
        
        self.neural_available = os.path.exists(self.neural_model_path)
        self.neurosymbolic_available = os.path.exists(self.neurosymbolic_model_path)
        
        print(f"✅ Neural model: {'Available' if self.neural_available else 'Missing'}")
        print(f"✅ Neurosymbolic model: {'Available' if self.neurosymbolic_available else 'Missing'}")
        
        if not (self.neural_available or self.neurosymbolic_available):
            print("❌ No trained models found! Please ensure model files exist.")
    
    def run_integrated_manual_trial(self, env, max_steps=5000):
        """Run an integrated manual control trial using the same environment as other approaches"""
        if not KEYBOARD_AVAILABLE:
            print("❌ Manual control not available - keyboard module not installed")
            return None
        
        # FIXED: Ensure CONFIG uses the same start/goal positions as other approaches
        CONFIG['start_pos'] = env.start_pos.copy()
        CONFIG['goal_pos'] = env.goal_pos.copy()
        
        print("👤 Starting Human Expert trial with INTEGRATED manual control...")
        print(f"🗺️  Using SAME obstacle configuration as other approaches")
        print(f"📍 Navigate from START {env.start_pos[:2]} to GOAL {env.goal_pos[:2]}")
        print(f"🚧 Level {env.level} with {len(env.obstacles)} obstacles")
        print(f"✅ Synchronized start/goal positions with other approaches")
        
        # FIXED: Regenerate XML with correct start/goal markers for manual control
        print("🔄 Regenerating environment XML with correct start/goal markers...")
        EnvironmentGenerator.create_xml_with_obstacles(env.obstacles)
        
        # Reload MuJoCo model to get updated start/goal markers
        env.model = mujoco.MjModel.from_xml_path("environment.xml")
        env.data = mujoco.MjData(env.model)
        
        # Create integrated manual controller
        controller = IntegratedManualController()
        controller.start_control()
        
        # Initialize performance tracker
        tracker = PerformanceTracker()
        tracker.start_trial(env.start_pos)
        
        # Reset UAV to start position
        env.reset_uav()
        
        # Open viewer for manual control
        print("🚀 Starting integrated manual control simulation...")
        print(f"🟢 GREEN markers show START position: {env.start_pos[:2]}")
        print(f"🔵 BLUE markers show GOAL position: {env.goal_pos[:2]}")
        print(f"🔴 RED UAV starts at: {env.start_pos[:2]}")
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            step_count = 0
            control_dt = CONFIG.get('control_dt', 0.05)
            kp_pos = CONFIG.get('kp_pos', 1.5)
            
            start_time = time.time()
            
            try:
                while (viewer.is_running() and controller.active and 
                       not controller.trial_complete and step_count < max_steps):
                    
                    # Get current UAV state
                    current_pos = env.data.qpos[:3].copy()
                    current_vel = env.data.qvel[:3].copy()
                    
                    # Update manual controls
                    manual_action = controller.update_controls()
                    
                    # Handle special keys
                    if controller.reset_requested:
                        env.reset_uav()
                        controller.reset_requested = False
                        controller.current_velocity = np.array([0.0, 0.0])
                        print("🔄 UAV reset to start position")
                        time.sleep(0.2)
                        continue
                    
                    if controller.exit_requested:
                        print("⏹️ Manual control exited by user")
                        break
                    
                    # Height control (maintain flight altitude)
                    desired_height = CONFIG['uav_flight_height']
                    height_error = desired_height - current_pos[2]
                    vz_correction = kp_pos * height_error
                    
                    # Apply manual control actions
                    env.data.qvel[0] = manual_action[0]  # X-velocity
                    env.data.qvel[1] = manual_action[1]  # Y-velocity  
                    env.data.qvel[2] = vz_correction    # Z-velocity (height control)
                    
                    # Check goal reached
                    goal_distance = np.linalg.norm(current_pos - env.goal_pos)
                    if goal_distance < 0.2 and not controller.goal_reached:
                        controller.goal_reached = True
                        controller.trial_complete = True
                        print(f"\n🎉 GOAL REACHED!")
                        print(f"📍 Final position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                        print(f"📏 Distance to goal: {goal_distance:.3f}m")
                        break
                    
                    # Check for collision
                    has_collision, obstacle_id, collision_dist = check_collision_precise(current_pos, env.obstacles)
                    if has_collision:
                        controller.collision_occurred = True
                        controller.trial_complete = True
                        print(f"\n💥 COLLISION DETECTED!")
                        print(f"💀 UAV crashed into obstacle: {obstacle_id}")
                        print(f"📏 Collision distance: {collision_dist:.3f}m")
                        break
                    
                    # Check boundaries
                    half_world = CONFIG['world_size'] / 2
                    if (abs(current_pos[0]) > half_world or abs(current_pos[1]) > half_world or 
                        current_pos[2] < 0.1 or current_pos[2] > 5.0):
                        controller.collision_occurred = True  # Treat as collision
                        controller.trial_complete = True
                        print(f"\n🚨 BOUNDARY VIOLATION!")
                        print(f"📍 UAV position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
                        break
                    
                    # Step the simulation
                    mujoco.mj_step(env.model, env.data)
                    
                    # Update path trail visualization
                    if step_count % 5 == 0:
                        controller.update_path_trail(env.model, current_pos)
                    
                    # Ensure UAV and marker visibility
                    controller.ensure_uav_visibility(env.model)
                    controller.ensure_markers_visibility(env.model)
                    
                    # Update performance tracking
                    tracker.update_step(current_pos, env.goal_pos)
                    
                    # Forward to viewer
                    viewer.sync()
                    
                    # Status updates
                    if step_count % 100 == 0:
                        print(f"Step {step_count:4d}: Pos=({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) | "
                              f"Goal dist: {goal_distance:.2f}m")
                    
                    step_count += 1
                    time.sleep(control_dt)
                    
            except KeyboardInterrupt:
                print("🛑 Manual control interrupted by user")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Finalize tracking
            tracker.end_trial(
                success=controller.goal_reached,
                collision=controller.collision_occurred,
                out_of_bounds=False,  # Handled as collision
                timeout=(step_count >= max_steps)
            )
            
            # Get final metrics
            metrics = tracker.get_metrics()
            metrics['duration'] = duration  # Override with actual duration
            
            # Print result summary
            if controller.goal_reached:
                print("\n🎯 Manual control completed successfully!")
            elif controller.collision_occurred:
                print("\n💥 Manual control ended with collision!")
            elif step_count >= max_steps:
                print("\n⏰ Manual control timed out!")
            else:
                print("\n⏹️ Manual control session ended.")
            
            return metrics
    

    
    def run_agent_trial_using_runner(self, env, model_path, agent_type="Neural", max_steps=5000):
        """Run an agent trial using the UAVAgentRunner"""
        print(f"🤖 Starting {agent_type} agent trial...")
        
        # FIXED: Ensure CONFIG uses the same start/goal positions as environment
        CONFIG['start_pos'] = env.start_pos.copy()
        CONFIG['goal_pos'] = env.goal_pos.copy()
        print(f"✅ Using SAME start/goal positions: START {env.start_pos[:2]} → GOAL {env.goal_pos[:2]}")
        
        # Setup neurosymbolic configuration based on agent type
        if agent_type == "Neurosymbolic":
            ns_cfg = {'use_neurosymbolic': True, 'lambda': 1.0}
        else:
            ns_cfg = {'use_neurosymbolic': False, 'lambda': 0.0}
        
        # Create agent runner
        runner = UAVAgentRunner(
            model_path=model_path,
            ns_cfg=ns_cfg,
            max_steps=max_steps,
            show_viewer=True,  # Show viewer for comparison
            verbose=False  # Reduce verbosity for cleaner output
        )
        
        try:
            # Setup environment with current obstacles
            runner.setup_environment(obstacles=env.obstacles)
            
            # Run trial
            results = runner.run_trial()
            
            # Convert results to our format
            direct_distance = np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos'])
            metrics = {
                'path_length': results['path_length'],
                'step_count': results['step_count'],
                'success': results['success'],
                'collision': results['collision'],
                'out_of_bounds': results['out_of_bounds'],
                'timeout': results['timeout'],
                'final_distance': results['final_distance'],
                'duration': 0.0,  # Not tracked in runner
                'path_efficiency': direct_distance / results['path_length'] if results['path_length'] > 0 else 0.0
            }
            
            # Print result
            if results['success']:
                print("🎯 Goal reached!")
            elif results['collision']:
                print("💥 Collision detected!")
            elif results['out_of_bounds']:
                print("🚫 Out of bounds!")
            elif results['timeout']:
                print("⏰ Trial timed out!")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Agent trial failed: {e}")
            return None
    
    def run_level_comparison(self, level):
        """Run comparison for a specific level with consistent environment across all approaches"""
        print(f"\n🎯 LEVEL {level} - {level} OBSTACLES")
        print("="*50)
        
        # FIXED: Generate obstacles once per level and reuse for all approaches
        # This ensures fair comparison with identical obstacle configurations
        env = ComparisonEnvironment(level)
        level_results = {}
        
        print(f"🗺️  Generated level {level} environment with {len(env.obstacles)} obstacles")
        print(f"📍 Start: [{env.start_pos[0]:.1f}, {env.start_pos[1]:.1f}] → Goal: [{env.goal_pos[0]:.1f}, {env.goal_pos[1]:.1f}]")
        print(f"⚖️  ALL APPROACHES will use IDENTICAL obstacle configuration for fair comparison")
        print(f"🔧 Environment stored positions: Start={env.start_pos}, Goal={env.goal_pos}")
        
        # Run trials for each approach in sequence - ALL USE SAME OBSTACLES & ENVIRONMENT
        
        # 1. Human Expert (first) - Uses SAME obstacles as others
        if KEYBOARD_AVAILABLE:
            print(f"\n--- Human Expert Trial (Same Environment) ---")
            try:
                metrics = self.run_integrated_manual_trial(env)
                level_results["Human Expert"] = metrics
                self.print_trial_results("Human Expert", metrics)
            except KeyboardInterrupt:
                print(f"❌ Human Expert trial cancelled by user")
                level_results["Human Expert"] = None
            except Exception as e:
                print(f"❌ Human Expert trial failed: {e}")
                level_results["Human Expert"] = None
        else:
            print("⚠️  Skipping Human Expert trial - keyboard module not available")
        
        # 2. Neural Only (second) - REUSES SAME obstacles & environment
        if self.neural_available:
            print(f"\n--- Neural Only Trial (Same Environment) ---")
            try:
                metrics = self.run_agent_trial_using_runner(env, self.neural_model_path, "Neural")
                level_results["Neural Only"] = metrics
                self.print_trial_results("Neural Only", metrics)
            except KeyboardInterrupt:
                print(f"❌ Neural Only trial cancelled by user")
                level_results["Neural Only"] = None
            except Exception as e:
                print(f"❌ Neural Only trial failed: {e}")
                level_results["Neural Only"] = None
        
        # 3. Neurosymbolic (third) - REUSES SAME obstacles & environment  
        if self.neurosymbolic_available:
            print(f"\n--- Neurosymbolic Trial (Same Environment) ---")
            try:
                metrics = self.run_agent_trial_using_runner(env, self.neurosymbolic_model_path, "Neurosymbolic")
                level_results["Neurosymbolic"] = metrics
                self.print_trial_results("Neurosymbolic", metrics)
            except KeyboardInterrupt:
                print(f"❌ Neurosymbolic trial cancelled by user")
                level_results["Neurosymbolic"] = None
            except Exception as e:
                print(f"❌ Neurosymbolic trial failed: {e}")
                level_results["Neurosymbolic"] = None
        
        # Store results
        self.results[level] = level_results
        
        # Print level summary
        self.print_level_summary(level, level_results)
        
        return level_results
    
    def print_trial_results(self, approach, metrics):
        """Print results for a single trial"""
        if metrics is None:
            print(f"❌ {approach}: Failed")
            return
        
        status = "✅ SUCCESS" if metrics['success'] else "❌ FAILED"
        failure_reason = ""
        if not metrics['success']:
            if metrics['collision']:
                failure_reason = " (Collision)"
            elif metrics['out_of_bounds']:
                failure_reason = " (Out of Bounds)"
            elif metrics['timeout']:
                failure_reason = " (Timeout)"
        
        print(f"{status}{failure_reason}")
        print(f"  Path Length: {metrics['path_length']:.2f}m")
        print(f"  Steps: {metrics['step_count']}")
        print(f"  Final Distance: {metrics['final_distance']:.2f}m")
        print(f"  Duration: {metrics['duration']:.1f}s")
        print(f"  Path Efficiency: {metrics['path_efficiency']:.2f}")
    
    def print_level_summary(self, level, results):
        """Print summary for a level"""
        print(f"\n📊 LEVEL {level} SUMMARY")
        print("-" * 30)
        
        for approach, metrics in results.items():
            if metrics:
                status = "SUCCESS" if metrics['success'] else "FAILED"
                print(f"{approach:15}: {status:7} | {metrics['path_length']:6.2f}m | {metrics['step_count']:4d} steps")
            else:
                print(f"{approach:15}: ERROR")
    
    def save_results_to_csv(self):
        """Save all results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['level', 'approach', 'success', 'path_length', 'step_count', 
                         'final_distance', 'duration', 'path_efficiency', 'collision', 
                         'out_of_bounds', 'timeout']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for level, level_results in self.results.items():
                for approach, metrics in level_results.items():
                    if metrics:
                        row = {
                            'level': level,
                            'approach': approach,
                            'success': metrics['success'],
                            'path_length': metrics['path_length'],
                            'step_count': metrics['step_count'],
                            'final_distance': metrics['final_distance'],
                            'duration': metrics['duration'],
                            'path_efficiency': metrics['path_efficiency'],
                            'collision': metrics['collision'],
                            'out_of_bounds': metrics['out_of_bounds'],
                            'timeout': metrics['timeout']
                        }
                        writer.writerow(row)
        
        print(f"📄 Results saved to {filename}")
    
    def print_final_summary(self):
        """Print final comparison summary"""
        print("\n" + "="*80)
        print("🏆 FINAL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Calculate aggregate statistics
        approaches = set()
        for level_results in self.results.values():
            approaches.update(level_results.keys())
        
        for approach in approaches:
            print(f"\n🤖 {approach}")
            print("-" * 50)
            
            successes = 0
            total_trials = 0
            avg_path_length = 0
            avg_steps = 0
            
            for level, level_results in self.results.items():
                if approach in level_results and level_results[approach]:
                    metrics = level_results[approach]
                    total_trials += 1
                    if metrics['success']:
                        successes += 1
                        avg_path_length += metrics['path_length']
                        avg_steps += metrics['step_count']
            
            if total_trials > 0:
                success_rate = successes / total_trials * 100
                if successes > 0:
                    avg_path_length /= successes
                    avg_steps /= successes
                
                print(f"Success Rate: {success_rate:.1f}% ({successes}/{total_trials})")
                if successes > 0:
                    print(f"Avg Path Length: {avg_path_length:.2f}m")
                    print(f"Avg Steps: {avg_steps:.1f}")
    
    def run_full_comparison(self):
        """Run the complete comparison across all levels"""
        print("🚀 UAV PERFORMANCE COMPARISON")
        print("Comparing Human Expert vs Neural Only vs Neurosymbolic")
        print("Levels 1-10 with increasing obstacle counts")
        print("="*80)
        
        for level in range(1, 11):
            try:
                self.run_level_comparison(level)
                
                # Ask user to continue
                if level < 10:
                    print(f"\n🔄 Level {level} completed!")
                    input("Press Enter to continue to next level, or Ctrl+C to stop...")
            
            except KeyboardInterrupt:
                print(f"\n⏹️  Comparison stopped at level {level}")
                break
            except Exception as e:
                print(f"❌ Error in level {level}: {e}")
                continue
        
        # Print final summary and save results
        self.print_final_summary()
        self.save_results_to_csv()

def main():
    """Main function"""
    print("🎮 UAV Performance Comparison Tool")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth",
        "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_1_0.pth"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nNote: Comparison will proceed with available files.")
    
    try:
        comparison = PerformanceComparison()
        comparison.run_full_comparison()
    except KeyboardInterrupt:
        print("\n👋 Comparison cancelled by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()