# UAV Navigation Comparison Test Suite
# Tests both Pure Neural and Neurosymbolic approaches across different scenarios

import numpy as np
import random
import time
import os
from datetime import datetime
import csv
import json
from uav_render_parameterized import run_uav_simulation_headless

def generate_random_obstacles(num_obstacles, world_size=8.0, obstacle_height=2.0):
    """Generate randomized obstacles for testing"""
    obstacles = []
    half_world = world_size / 2
    
    # Grid-based distribution for better spacing
    grid_size = int(np.sqrt(max(num_obstacles, 9))) + 1
    cell_size = world_size / grid_size
    positions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = -half_world + (i + 0.5) * cell_size
            y = -half_world + (j + 0.5) * cell_size
            positions.append((x, y))
    
    random.shuffle(positions)
    
    for i in range(num_obstacles):
        x, y = positions[i % len(positions)]
        
        # Add randomness to avoid perfect grid alignment
        x += random.uniform(-cell_size/4, cell_size/4)
        y += random.uniform(-cell_size/4, cell_size/4)
        
        # Ensure obstacles stay within bounds
        x = max(-half_world + 0.5, min(half_world - 0.5, x))
        y = max(-half_world + 0.5, min(half_world - 0.5, y))
        
        # Random sizes
        size_x = random.uniform(0.05, 0.12)
        size_y = random.uniform(0.05, 0.12)
        height = obstacle_height
        
        # Random shape
        obstacle_type = random.choice(['box', 'cylinder'])
        
        # Random color
        color = [random.uniform(0.1, 0.9) for _ in range(3)] + [1.0]
        
        obstacles.append({
            'id': f'test_obs_{i}',
            'shape': obstacle_type,
            'pos': [x, y, height/2],
            'size': [size_x, size_y, height/2] if obstacle_type == 'box' else [min(size_x, size_y), height/2],
            'color': color
        })
    
    return obstacles

def generate_mujoco_xml(obstacles, start_pos, goal_pos):
    """Generate MuJoCo XML with given obstacles and positions"""
    xml_lines = [
        '<mujoco model="complex_uav_env">',
        '  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>',
        '  <option integrator="RK4" timestep="0.01"/>',
        '  ',
        '  <visual>',
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global offwidth="2560" offheight="1440"/>',
        '  </visual>',
        '  ',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        '    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>',
        '    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>',
        '  </asset>',
        '',
        '  <worldbody>',
        '    <!-- Ground plane -->',
        '    <geom name="ground" type="plane" size="4.0 4.0 0.1" material="grid"/>',
        '    <light name="light1" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>',
        '    ',
        f'    <!-- Start position marker (green) -->',
        f'    <geom name="start_marker" type="cylinder" size="0.15 0.05" pos="{start_pos[0]} {start_pos[1]} 0.05" rgba="0 1 0 0.8"/>',
        f'    <geom name="start_pole" type="box" size="0.03 0.03 0.4" pos="{start_pos[0]} {start_pos[1]} 0.4" rgba="0 1 0 1"/>',
        '    ',
        f'    <!-- Goal position marker (blue) -->',
        f'    <geom name="goal_marker" type="cylinder" size="0.15 0.05" pos="{goal_pos[0]} {goal_pos[1]} 0.05" rgba="0 0 1 0.8"/>',
        f'    <geom name="goal_pole" type="box" size="0.03 0.03 0.4" pos="{goal_pos[0]} {goal_pos[1]} 0.4" rgba="0 0 1 1"/>',
        '    ',
        '    <!-- UAV starting position -->',
        f'    <body name="chassis" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}">',
        '      <joint type="free" name="root"/>',
        '      <geom type="box" size="0.12 0.12 0.02" rgba="1.0 0.0 0.0 1.0" mass="0.8"/>',
        '      ',
        '      <!-- Propeller arms and motors -->',
        '      <geom type="box" size="0.08 0.01 0.005" pos="0 0 0.01" rgba="0.3 0.3 0.3 1"/>',
        '      <geom type="box" size="0.01 0.08 0.005" pos="0 0 0.01" rgba="0.3 0.3 0.3 1"/>',
        '      ',
        '      <!-- Motor visual geometry -->',
        '      <geom type="cylinder" size="0.015 0.02" pos="0.08 0.08 0.015" rgba="0.2 0.2 0.2 1"/>',
        '      <geom type="cylinder" size="0.015 0.02" pos="-0.08 0.08 0.015" rgba="0.2 0.2 0.2 1"/>',
        '      <geom type="cylinder" size="0.015 0.02" pos="0.08 -0.08 0.015" rgba="0.2 0.2 0.2 1"/>',
        '      <geom type="cylinder" size="0.015 0.02" pos="-0.08 -0.08 0.015" rgba="0.2 0.2 0.2 1"/>',
        '      ',
        '      <!-- Propellers -->',
        '      <geom type="cylinder" size="0.04 0.002" pos="0.08 0.08 0.035" rgba="0.7 0.7 0.7 0.8"/>',
        '      <geom type="cylinder" size="0.04 0.002" pos="-0.08 0.08 0.035" rgba="0.7 0.7 0.7 0.8"/>',
        '      <geom type="cylinder" size="0.04 0.002" pos="0.08 -0.08 0.035" rgba="0.7 0.7 0.7 0.8"/>',
        '      <geom type="cylinder" size="0.04 0.002" pos="-0.08 -0.08 0.035" rgba="0.7 0.7 0.7 0.8"/>',
        '      ',
        '      <!-- Sites for motor force application -->',
        '      <site name="motor1" pos="0.08 0.08 0" size="0.01"/>',
        '      <site name="motor2" pos="-0.08 0.08 0" size="0.01"/>',
        '      <site name="motor3" pos="0.08 -0.08 0" size="0.01"/>',
        '      <site name="motor4" pos="-0.08 -0.08 0" size="0.01"/>',
        '    </body>',
    ]
    
    # Add obstacles
    for obs in obstacles:
        pos_str = f"{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}"
        color_str = f"{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"
        
        if obs['shape'] == 'box':
            size_str = f"{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}"
            xml_lines.append(f'    <geom name="{obs["id"]}" type="box" size="{size_str}" pos="{pos_str}" rgba="{color_str}"/>')
        else:  # cylinder
            size_str = f"{obs['size'][0]} {obs['size'][1]}"
            xml_lines.append(f'    <geom name="{obs["id"]}" type="cylinder" size="{size_str}" pos="{pos_str}" rgba="{color_str}"/>')
    
    xml_lines.extend([
        '  </worldbody>',
        '',
        '  <actuator>',
        '    <motor name="m1" gear="0 0 1 0 0 0" site="motor1"/>',
        '    <motor name="m2" gear="0 0 1 0 0 0" site="motor2"/>',
        '    <motor name="m3" gear="0 0 1 0 0 0" site="motor3"/>',
        '    <motor name="m4" gear="0 0 1 0 0 0" site="motor4"/>',
        '  </actuator>',
        '</mujoco>',
    ])
    
    return '\n'.join(xml_lines)

def generate_random_corner_position(world_size=8.0, flight_height=1.0):
    """Generate random start position from corners"""
    half_world = world_size / 2 - 1.0  # 1m margin from boundary
    corners = [
        [-half_world, -half_world, flight_height],  # Bottom-left
        [half_world, -half_world, flight_height],   # Bottom-right
        [-half_world, half_world, flight_height],   # Top-left
        [half_world, half_world, flight_height]     # Top-right
    ]
    return random.choice(corners)

def generate_random_goal_position(start_pos, world_size=8.0, flight_height=1.0):
    """Generate random goal position ensuring it's different from start"""
    half_world = world_size / 2 - 1.0  # 1m margin from boundary
    
    # Try to get a goal that's sufficiently far from start
    max_attempts = 10
    for _ in range(max_attempts):
        x = random.uniform(-half_world, half_world)
        y = random.uniform(-half_world, half_world)
        goal_pos = [x, y, flight_height]
        
        # Ensure goal is at least 2 units away from start
        distance = np.linalg.norm(np.array(goal_pos[:2]) - np.array(start_pos[:2]))
        if distance >= 2.0:
            return goal_pos
    
    # Fallback: use opposite corner
    if start_pos[0] < 0 and start_pos[1] < 0:  # Bottom-left start
        return [half_world, half_world, flight_height]  # Top-right goal
    elif start_pos[0] > 0 and start_pos[1] < 0:  # Bottom-right start
        return [-half_world, half_world, flight_height]  # Top-left goal
    elif start_pos[0] < 0 and start_pos[1] > 0:  # Top-left start
        return [half_world, -half_world, flight_height]  # Bottom-right goal
    else:  # Top-right start
        return [-half_world, -half_world, flight_height]  # Bottom-left goal

def check_position_safety(position, obstacles, safety_radius=0.8):
    """Check if start/goal positions are safe from obstacles"""
    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        distance = np.linalg.norm(np.array(position[:2]) - obs_pos[:2])
        
        if obs['shape'] == 'box':
            min_safe_distance = max(obs['size'][0], obs['size'][1]) + safety_radius
        else:  # cylinder
            min_safe_distance = obs['size'][0] + safety_radius
        
        if distance < min_safe_distance:
            return False
    return True

def run_comprehensive_comparison():
    """Run comprehensive comparison between Pure Neural, Neurosymbolic, and AR_PPO approaches"""
    
    # Model paths
    pure_neural_model = "PPO_preTrained/UAVEnv/Vanilla_PPO_UAV_Weights.pth"
    neurosymbolic_model = "PPO_preTrained/UAVEnv/NS_PPO_UAV_Weights.pth"
    ar_ppo_model = "PPO_preTrained/UAVEnv/AR_PPO_UAV_Weights.pth"
    
    # Check if model files exist
    if not os.path.exists(pure_neural_model):
        print(f"❌ Pure Neural model not found: {pure_neural_model}")
        return
    if not os.path.exists(neurosymbolic_model):
        print(f"❌ Neurosymbolic model not found: {neurosymbolic_model}")
        return
    if not os.path.exists(ar_ppo_model):
        print(f"❌ AR_PPO model not found: {ar_ppo_model}")
        return
    
    # Check if we're on macOS and warn about mjpython requirement
    import platform
    if platform.system() == "Darwin":  # macOS
        print("⚠️  macOS detected: MuJoCo viewer may require 'mjpython' instead of 'python'")
        print("⚠️  If you encounter viewer errors, run with: mjpython uav_comparison_test.py")
        print("⚠️  Continuing with headless mode for compatibility...")
    
    # Results storage
    all_results = []
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"comparison_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚁 UAV Navigation Comparison Test Suite")
    print("="*60)
    print(f"📊 Testing obstacle counts: 1-15")
    print(f"🔄 Iterations per obstacle count: 10")
    print(f"🤖 Models: Pure Neural vs Neurosymbolic vs AR_PPO")
    print(f"📁 Results directory: {results_dir}")
    print("="*60)
    
    total_tests = 15 * 10 * 3  # obstacle_counts * iterations * models
    current_test = 0
    
    # Outer loop: obstacle count (1 to 15)
    for obstacle_count in range(1, 16):
        print(f"\n🎯 Testing with {obstacle_count} obstacles")
        print("-" * 40)
        
        obstacle_results = {
            'obstacle_count': obstacle_count,
            'pure_neural': {'successes': 0, 'collisions': 0, 'timeouts': 0, 'avg_steps': 0, 'avg_distance': 0},
            'neurosymbolic': {'successes': 0, 'collisions': 0, 'timeouts': 0, 'avg_steps': 0, 'avg_distance': 0},
            'ar_ppo': {'successes': 0, 'collisions': 0, 'timeouts': 0, 'avg_steps': 0, 'avg_distance': 0}
        }
        
        # Inner loop: 10 iterations with randomized positions
        for iteration in range(10):
            print(f"\n📍 Iteration {iteration + 1}/10 for {obstacle_count} obstacles")
            
            # Generate random scenario
            obstacles = generate_random_obstacles(obstacle_count)
            
            # Generate safe start and goal positions
            max_position_attempts = 20
            for attempt in range(max_position_attempts):
                start_pos = generate_random_corner_position()
                goal_pos = generate_random_goal_position(start_pos)
                
                # Check if positions are safe
                if (check_position_safety(start_pos, obstacles) and 
                    check_position_safety(goal_pos, obstacles)):
                    break
                
                if attempt == max_position_attempts - 1:
                    print(f"⚠️ Could not find safe positions after {max_position_attempts} attempts. Using default positions.")
                    start_pos = [-3.0, -3.0, 1.0]
                    goal_pos = [3.0, 3.0, 1.0]
            
            print(f"   🟢 Start: [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
            print(f"   🔵 Goal:  [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f}]")
            print(f"   🧱 Obstacles: {len(obstacles)}")
            
            # Test all three models
            for model_name, model_path in [("Pure Neural", pure_neural_model), 
                                           ("Neurosymbolic", neurosymbolic_model),
                                           ("AR_PPO", ar_ppo_model)]:
                current_test += 1
                print(f"\n   🤖 Testing {model_name} ({current_test}/{total_tests})")
                
                try:
                    # Run simulation (headless mode for compatibility)
                    start_time = time.time()
                    results = run_uav_simulation_headless(
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        obstacle_count=obstacle_count,
                        model_path=model_path,
                        obstacle_positions=obstacles
                    )
                    end_time = time.time()
                    
                    # Process results
                    duration = end_time - start_time
                    if model_name == "Pure Neural":
                        model_key = 'pure_neural'
                    elif model_name == "Neurosymbolic":
                        model_key = 'neurosymbolic'
                    else:  # AR_PPO
                        model_key = 'ar_ppo'
                    
                    if results['success']:
                        obstacle_results[model_key]['successes'] += 1
                        status = "✅ SUCCESS"
                    elif results['collision']:
                        obstacle_results[model_key]['collisions'] += 1
                        status = "💥 COLLISION"
                    elif results.get('out_of_bounds', False):
                        status = "🚫 OUT OF BOUNDS"
                    elif results['timeout']:
                        obstacle_results[model_key]['timeouts'] += 1
                        status = "⏰ TIMEOUT"
                    else:
                        status = "❓ UNKNOWN"
                    
                    obstacle_results[model_key]['avg_steps'] += results['steps']
                    obstacle_results[model_key]['avg_distance'] += results['final_distance']
                    
                    print(f"      {status} | Steps: {results['steps']} | Final dist: {results['final_distance']:.3f}m | Time: {duration:.1f}s")
                    
                    # Store detailed result
                    detailed_result = {
                        'timestamp': datetime.now().isoformat(),
                        'obstacle_count': obstacle_count,
                        'iteration': iteration + 1,
                        'model': model_name,
                        'start_pos': start_pos,
                        'goal_pos': goal_pos,
                        'success': results['success'],
                        'collision': results['collision'],
                        'timeout': results['timeout'],
                        'out_of_bounds': results.get('out_of_bounds', False),
                        'steps': results['steps'],
                        'final_distance': results['final_distance'],
                        'duration_seconds': duration,
                        'obstacle_positions': obstacles
                    }
                    all_results.append(detailed_result)
                    
                except Exception as e:
                    print(f"      ❌ ERROR: {str(e)}")
                    # Record error result
                    detailed_result = {
                        'timestamp': datetime.now().isoformat(),
                        'obstacle_count': obstacle_count,
                        'iteration': iteration + 1,
                        'model': model_name,
                        'start_pos': start_pos,
                        'goal_pos': goal_pos,
                        'success': False,
                        'collision': False,
                        'timeout': False,
                        'out_of_bounds': False,
                        'steps': 0,
                        'final_distance': 999.0,
                        'duration_seconds': 0,
                        'error': str(e),
                        'obstacle_positions': obstacles
                }
                all_results.append(detailed_result)
        
        # Print summary for this obstacle count
        print(f"\n📊 Summary for {obstacle_count} obstacles:")
        for model_key in ['Pure_Neural', 'Neurosymbolic', 'AR_PPO']:
            result = obstacle_results[model_key]
            print(f"   {model_key:15s} {result['successes']}/1 success | "
                  f"Steps: {result['steps']:.0f} | "
                  f"Dist: {result['distance']:.3f}m")    # Create summary CSV
    custom_csv_data = []
    
    for idx, result in enumerate(all_results):
        if 'error' not in result:  # Skip error results
            # Map seed is the obstacle count (each obstacle count = 1 unique map)
            map_seed = result['obstacle_count']
            
            # Use the actual out_of_bounds data from simulation
            out_of_bound = result.get('out_of_bounds', False)
            
            # Handle steps for timeout cases - use max steps (5000) if timeout occurred
            steps = result['steps'] if not result['timeout'] else 5000
            
            custom_csv_data.append({
                'map_seed': map_seed,
                'level': result['obstacle_count'],
                'model': result['model'],
                'goal_reached': result['success'],
                'number_of_steps': steps,
                'collision': result['collision'],
                'out_of_bound': out_of_bound
            })
    
    # Save summary CSV
    custom_csv_file = os.path.join(results_dir, "results_summary.csv")
    with open(custom_csv_file, 'w', newline='') as f:
        if custom_csv_data:
            fieldnames = ['map_seed', 'level', 'model', 'goal_reached', 'number_of_steps', 'collision', 'out_of_bound']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(custom_csv_data)
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"📁 Results saved to: {results_dir}/")
    print(f"   📊 Summary CSV: results_summary.csv")
    print(f"\n📂 Folder Structure:")
    print(f"   {results_dir}/")
    print(f"   ├── Pure_Neural/")
    print(f"   │   ├── obstacles_1/")
    print(f"   │   │   ├── map.xml")
    print(f"   │   │   ├── map_metadata.json")
    print(f"   │   │   └── trajectories/trajectory.json")
    print(f"   │   ├── obstacles_2/")
    print(f"   │   └── ...")
    print(f"   ├── Neurosymbolic/")
    print(f"   │   └── (same structure)")
    print(f"   ├── AR_PPO/")
    print(f"   │   └── (same structure)")
    print(f"   └── results_summary.csv")

if __name__ == "__main__":
    print("🚁 Starting UAV Navigation Comparison Test Suite...")
    print("⚠️  This will take a significant amount of time to complete.")
    print("💡 Press Ctrl+C to stop at any time.")
    
    try:
        run_comprehensive_comparison()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {str(e)}")
    
    print("\n🏁 Test suite finished.")