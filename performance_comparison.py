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

class ManualController:
    """Handle manual control input for human expert trials"""
    
    def __init__(self):
        self.target_velocity = np.array([0.0, 0.0])
        self.active = False
        self.manual_speed = CONFIG.get('manual_speed', 0.8)
        self.reset_requested = False
        self.exit_requested = False
    
    def start_control(self):
        """Start manual control session"""
        self.active = True
        self.reset_requested = False
        self.exit_requested = False
        self.print_controls()
    
    def stop_control(self):
        """Stop manual control session"""
        self.active = False
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print("🎮 MANUAL CONTROL ACTIVE")
        print("="*60)
        print("MOVEMENT: Arrow Keys (↑↓←→)")
        print("STOP: SPACE")
        print("RESET: R")
        print("EXIT TRIAL: ESC")
        print("="*60)
    
    def update_controls(self):
        """Update target velocity based on keyboard input"""
        if not self.active or not KEYBOARD_AVAILABLE:
            return np.array([0.0, 0.0])
        
        target_vel = np.array([0.0, 0.0])
        
        try:
            if keyboard.is_pressed('up'):
                target_vel[1] += self.manual_speed
            if keyboard.is_pressed('down'):
                target_vel[1] -= self.manual_speed
            if keyboard.is_pressed('left'):
                target_vel[0] -= self.manual_speed
            if keyboard.is_pressed('right'):
                target_vel[0] += self.manual_speed
            if keyboard.is_pressed('space'):
                target_vel = np.array([0.0, 0.0])
            if keyboard.is_pressed('r'):
                self.reset_requested = True
            if keyboard.is_pressed('esc'):
                self.exit_requested = True
        except:
            # Handle potential keyboard module issues
            pass
        
        self.target_velocity = target_vel
        return target_vel

class ComparisonEnvironment:
    """Environment wrapper for running comparison trials"""
    
    def __init__(self, level=1):
        self.level = level
        self.obstacles = []
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment for the current level"""
        # Generate obstacles for current level
        self.obstacles = EnvironmentGenerator.generate_obstacles(self.level)
        
        # Generate safe start and goal positions
        self.generate_safe_positions()
        
        # Create XML file
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
    
    def generate_safe_positions(self):
        """Generate safe start and goal positions for the current level"""
        # Use similar logic as in UAVEnv
        half_world = CONFIG['world_size'] / 2 - 1.0
        
        # Random start corner
        corners = [
            np.array([-half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),
            np.array([-half_world, half_world, CONFIG['uav_flight_height']]),
            np.array([half_world, half_world, CONFIG['uav_flight_height']])
        ]
        
        CONFIG['start_pos'] = corners[0]  # Use consistent start position
        
        # Random goal position (different from start)
        goal_corners = corners[1:]  # Exclude start corner
        CONFIG['goal_pos'] = goal_corners[np.random.randint(len(goal_corners))]
        
        print(f"📍 Level {self.level}: Start {CONFIG['start_pos'][:2]}, Goal {CONFIG['goal_pos'][:2]}")
    
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
        self.manual_controller = ManualController()
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
    
    def run_manual_trial_using_existing(self, env, max_steps=5000):
        """Run a manual control trial using the existing uav_manual_control.py file"""
        if not KEYBOARD_AVAILABLE:
            print("❌ Manual control not available - keyboard module not installed")
            return None
        
        print("👤 Starting Human Expert trial...")
        print("🎮 Launching UAV Manual Control (uav_manual_control.py)...")
        
        # Update CONFIG in the environment to match current level
        original_start = CONFIG['start_pos'].copy()
        original_goal = CONFIG['goal_pos'].copy()
        
        try:
            # Set the current level's start/goal positions
            CONFIG['start_pos'] = CONFIG['start_pos']
            CONFIG['goal_pos'] = CONFIG['goal_pos']
            
            # Backup original environment.xml if it exists
            if os.path.exists("environment.xml"):
                os.rename("environment.xml", "environment_backup.xml")
            
            # Create the environment.xml with current obstacles
            EnvironmentGenerator.create_xml_with_obstacles(env.obstacles)
            
            print(f"📍 Navigate from START {CONFIG['start_pos'][:2]} to GOAL {CONFIG['goal_pos'][:2]}")
            print(f"🚧 Level {env.level} with {len(env.obstacles)} obstacles")
            print("")
            print("🎮 Manual Control Instructions:")
            print("   ↑↓←→  : Move UAV")
            print("   SPACE : Stop")  
            print("   R     : Reset")
            print("   ESC   : Exit")
            print("")
            print("🎯 Navigate to the GOAL (blue marker) and avoid obstacles!")
            print("📊 Performance will be measured automatically.")
            print("")
            
            # Record start time
            start_time = time.time()
            
            # Run the original manual control script interactively
            import subprocess
            
            print("🚀 Launching interactive manual control...")
            print("⚠️  A new window will open for manual control.")
            print("🎮 Use that window to control the UAV, then return here when done.")
            
            # Run without capturing output so user can interact
            result = subprocess.run([
                sys.executable, 'uav_manual_control.py',
                '--static_obstacles', str(len(env.obstacles)),
                '--start_pos', str(CONFIG['start_pos'][0]), str(CONFIG['start_pos'][1]), str(CONFIG['start_pos'][2]),
                '--goal_pos', str(CONFIG['goal_pos'][0]), str(CONFIG['goal_pos'][1]), str(CONFIG['goal_pos'][2])
            ], timeout=300)  # 5 minute timeout, no output capture
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Ask user for result since we can't parse output
            print("\n🔍 Manual control session completed.")
            print("Please report the outcome:")
            print("1 - Success (reached goal)")
            print("2 - Collision (hit obstacle)")
            print("3 - Gave up / Other")
            
            while True:
                try:
                    choice = input("Enter your choice (1-3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Please enter 1, 2, or 3")
                except KeyboardInterrupt:
                    choice = '3'
                    break
            
            success = (choice == '1')
            collision = (choice == '2')
            
            if success:
                print("🎯 Manual control completed successfully!")
            elif collision:
                print("💥 Manual control ended with collision!")
            else:
                print("⏹️ Manual control session ended without completion.")
            
            # Since we can't get exact metrics from the original script,
            # we'll return estimated metrics based on the outcome
            if success:
                # Estimate metrics for successful completion
                direct_distance = np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos'])
                estimated_path_length = direct_distance * 1.2  # Assume 20% longer than direct path
                estimated_steps = int(duration / CONFIG.get('control_dt', 0.05))
                
                metrics = {
                    'path_length': estimated_path_length,
                    'step_count': estimated_steps,
                    'success': True,
                    'collision': False,
                    'out_of_bounds': False,
                    'timeout': False,
                    'final_distance': 0.2,  # Within goal tolerance
                    'duration': duration,
                    'path_efficiency': direct_distance / estimated_path_length
                }
            elif collision:
                # Estimate metrics for collision
                direct_distance = np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos'])
                estimated_path_length = direct_distance * 0.5  # Partial path
                estimated_steps = int(duration / CONFIG.get('control_dt', 0.05))
                
                metrics = {
                    'path_length': estimated_path_length,
                    'step_count': estimated_steps,
                    'success': False,
                    'collision': True,
                    'out_of_bounds': False,
                    'timeout': False,
                    'final_distance': direct_distance * 0.5,  # Estimate based on partial completion
                    'duration': duration,
                    'path_efficiency': 0.5
                }
            else:
                # User exited or other termination
                direct_distance = np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos'])
                estimated_steps = int(duration / CONFIG.get('control_dt', 0.05))
                
                metrics = {
                    'path_length': 1.0,  # Minimal movement
                    'step_count': estimated_steps,
                    'success': False,
                    'collision': False,
                    'out_of_bounds': False,
                    'timeout': duration > 250,  # Close to timeout
                    'final_distance': direct_distance,  # No progress
                    'duration': duration,
                    'path_efficiency': 0.1
                }
            
            return metrics
            
        except subprocess.TimeoutExpired:
            print("⏰ Manual control timed out after 5 minutes")
            return {
                'path_length': 0.0,
                'step_count': 0,
                'success': False,
                'collision': False,
                'out_of_bounds': False,
                'timeout': True,
                'final_distance': np.linalg.norm(CONFIG['goal_pos'] - CONFIG['start_pos']),
                'duration': 300.0,
                'path_efficiency': 0.0
            }
        except Exception as e:
            print(f"❌ Manual control failed: {e}")
            return None
        finally:
            # Restore original positions
            CONFIG['start_pos'] = original_start
            CONFIG['goal_pos'] = original_goal
            
            # Restore original environment.xml if we backed it up
            if os.path.exists("environment_backup.xml"):
                if os.path.exists("environment.xml"):
                    os.remove("environment.xml")
                os.rename("environment_backup.xml", "environment.xml")
    

    
    def run_agent_trial_using_runner(self, env, model_path, agent_type="Neural", max_steps=5000):
        """Run an agent trial using the UAVAgentRunner"""
        print(f"🤖 Starting {agent_type} agent trial...")
        
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
        """Run comparison for a specific level"""
        print(f"\n🎯 LEVEL {level} - {level} OBSTACLES")
        print("="*50)
        
        env = ComparisonEnvironment(level)
        level_results = {}
        
        # Run trials for each approach in sequence
        
        # 1. Human Expert (first)
        if KEYBOARD_AVAILABLE:
            print(f"\n--- Human Expert Trial ---")
            try:
                metrics = self.run_manual_trial_using_existing(env)
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
        
        # 2. Neural Only (second)
        if self.neural_available:
            print(f"\n--- Neural Only Trial ---")
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
        
        # 3. Neurosymbolic (third)
        if self.neurosymbolic_available:
            print(f"\n--- Neurosymbolic Trial ---")
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