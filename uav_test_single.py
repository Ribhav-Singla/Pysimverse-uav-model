# UAV Single Model Test Script
# Tests a single PPO variant (vanilla, ar, or ns) with trajectory tracking

import numpy as np
import random
import time
import os
import argparse
from datetime import datetime
import csv
import json
from uav_render_parameterized import run_uav_simulation_headless

# PPO Model Configuration
PPO_MODEL_MAP = {
    'vanilla': {'name': 'Pure_Neural', 'path': 'PPO_preTrained/UAVEnv/Vanilla_PPO_UAV_Weights.pth'},
    'ar': {'name': 'AR_PPO', 'path': 'PPO_preTrained/UAVEnv/AR_PPO_UAV_Weights.pth'},
    'ns': {'name': 'Neurosymbolic', 'path': 'PPO_preTrained/UAVEnv/NS_PPO_UAV_Weights.pth'}
}

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

def generate_random_goal_position(start_pos, obstacles, world_size=8.0, flight_height=1.0):
    """Generate random goal position ensuring it's different from start and away from obstacles"""
    half_world = world_size / 2 - 1.0  # 1m margin from boundary
    
    # Try to get a goal that's sufficiently far from start and obstacles
    max_attempts = 50
    for _ in range(max_attempts):
        x = random.uniform(-half_world, half_world)
        y = random.uniform(-half_world, half_world)
        goal_pos = [x, y, flight_height]
        
        # Ensure goal is at least 2 units away from start
        distance_to_start = np.linalg.norm(np.array(goal_pos[:2]) - np.array(start_pos[:2]))
        if distance_to_start < 2.0:
            continue
        
        # Ensure goal is at least 0.8 units away from all obstacles
        safe_from_obstacles = True
        for obs in obstacles:
            obs_pos = np.array(obs['pos'][:2])
            distance_to_obs = np.linalg.norm(np.array(goal_pos[:2]) - obs_pos)
            
            if obs['shape'] == 'box':
                min_safe_distance = max(obs['size'][0], obs['size'][1]) + 0.8
            else:  # cylinder
                min_safe_distance = obs['size'][0] + 0.8
            
            if distance_to_obs < min_safe_distance:
                safe_from_obstacles = False
                break
        
        if safe_from_obstacles:
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

def run_single_model_test(ppo_type):
    """Run test for a single PPO model with trajectory tracking"""
    
    # Get model configuration
    model_config = PPO_MODEL_MAP[ppo_type]
    model = {"name": model_config['name'], "path": model_config['path']}
    
    # Check if model file exists
    if not os.path.exists(model["path"]):
        print(f"❌ {model['name']} model not found: {model['path']}")
        return
    
    # Results storage
    all_results = []
    
    # Create results directory
    results_dir = "Agents"
    agent_name = model["name"]
    
    # Remove existing agent folder if it exists to replace it
    import shutil
    agent_dir = os.path.join(results_dir, agent_name)
    if os.path.exists(agent_dir):
        shutil.rmtree(agent_dir)
    os.makedirs(agent_dir, exist_ok=True)
    
    print(f"🚁 UAV Navigation Test - {ppo_type.upper()} PPO ({agent_name})")
    print("="*60)
    print(f"📊 Testing obstacle counts: 1-25")
    print(f"🗺️  5 unique maps per obstacle count")
    print(f"🤖 Model: {agent_name}")
    print(f"📁 Results directory: {agent_dir}")
    print("="*60)
    
    maps_per_level = 5
    total_tests = 25 * maps_per_level  # obstacle_counts * maps
    current_test = 0
    
    # Loop through obstacle counts (1 to 25)
    for obstacle_count in range(1, 26):
        print(f"\n🎯 Testing with {obstacle_count} obstacles")
        print("=" * 60)
        
        # Generate 5 unique maps for this obstacle count
        for map_id in range(1, maps_per_level + 1):
            current_test += 1
            print(f"\n📍 Map {map_id}/{maps_per_level} for obstacle level {obstacle_count} ({current_test}/{total_tests})")
            print("-" * 40)
            
            obstacles = generate_random_obstacles(obstacle_count)
            
            # Generate safe start and goal positions
            max_position_attempts = 20
            for attempt in range(max_position_attempts):
                start_pos = generate_random_corner_position()
                goal_pos = generate_random_goal_position(start_pos, obstacles)
                
                if check_position_safety(start_pos, obstacles):
                    break
                
                if attempt == max_position_attempts - 1:
                    print(f"⚠️ Using default positions")
                    start_pos = [-3.0, -3.0, 1.0]
                    goal_pos = [3.0, 3.0, 1.0]
            
            print(f"   🟢 Start: [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
            print(f"   🔵 Goal:  [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f}]")
            print(f"   🧱 Obstacles: {len(obstacles)}")
            
            # Generate MuJoCo XML for this map
            map_xml = generate_mujoco_xml(obstacles, start_pos, goal_pos)
            
            # Create obstacle folder name with map ID
            obstacle_folder_name = f"obstacles_{obstacle_count}"
            
            print(f"\n   🤖 Testing {model['name']}")
            
            # Create folder structure: agent_name/obstacles_X/map_Y/trajectories/
            obstacle_dir = os.path.join(agent_dir, obstacle_folder_name)
            map_dir = os.path.join(obstacle_dir, f"map_{map_id}")
            trajectory_dir = os.path.join(map_dir, "trajectories")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Save map XML in map folder
            map_xml_file = os.path.join(map_dir, "map.xml")
            with open(map_xml_file, 'w') as f:
                f.write(map_xml)
            
            # Save map metadata
            map_metadata = {
                'obstacle_count': obstacle_count,
                'map_id': map_id,
                'start_position': start_pos,
                'goal_position': goal_pos,
                'obstacles': obstacles
            }
            map_metadata_file = os.path.join(map_dir, "map_metadata.json")
            with open(map_metadata_file, 'w') as f:
                json.dump(map_metadata, f, indent=2)
            
            try:
                # Run simulation
                start_time = time.time()
                results = run_uav_simulation_headless(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    obstacle_count=obstacle_count,
                    model_path=model["path"],
                    obstacle_positions=obstacles,
                    return_trajectory=True
                )
                end_time = time.time()
                
                duration = end_time - start_time
                
                if results['success']:
                    status = "✅ SUCCESS"
                elif results['collision']:
                    status = "💥 COLLISION"
                elif results.get('out_of_bounds', False):
                    status = "🚫 OUT OF BOUNDS"
                elif results['timeout']:
                    status = "⏰ TIMEOUT"
                else:
                    status = "❓ UNKNOWN"
                
                print(f"      {status} | Steps: {results['steps']} | Dist: {results['final_distance']:.3f}m | Time: {duration:.1f}s")
                
                # Save trajectory data
                if 'trajectory' in results and results['trajectory']:
                    trajectory_file = os.path.join(trajectory_dir, "trajectory.json")
                    with open(trajectory_file, 'w') as f:
                        json.dump(results['trajectory'], f, indent=2)
                    print(f"      💾 Saved {len(results['trajectory'])} trajectory points")
                
                # Store result
                detailed_result = {
                    'timestamp': datetime.now().isoformat(),
                    'obstacle_count': obstacle_count,
                    'map_id': map_id,
                    'model': model["name"],
                    'success': results['success'],
                    'collision': results['collision'],
                    'timeout': results['timeout'],
                    'out_of_bounds': results.get('out_of_bounds', False),
                    'steps': results['steps'],
                    'final_distance': results['final_distance'],
                    'duration_seconds': duration
                }
                all_results.append(detailed_result)
                
            except Exception as e:
                print(f"      ❌ ERROR: {str(e)}")
                detailed_result = {
                    'timestamp': datetime.now().isoformat(),
                    'obstacle_count': obstacle_count,
                    'map_id': map_id,
                    'model': model["name"],
                    'success': False,
                    'collision': False,
                    'timeout': False,
                    'out_of_bounds': False,
                    'steps': 0,
                    'final_distance': 999.0,
                    'duration_seconds': 0,
                    'error': str(e)
                }
                all_results.append(detailed_result)
    
    # Create summary CSV
    custom_csv_data = []
    
    for result in all_results:
        if 'error' not in result:
            steps = result['steps'] if not result['timeout'] else 5000
            
            custom_csv_data.append({
                'map_seed': f"{result['obstacle_count']}_{result['map_id']}",
                'level': result['obstacle_count'],
                'map_id': result['map_id'],
                'model': result['model'],
                'goal_reached': result['success'],
                'number_of_steps': steps,
                'collision': result['collision'],
                'out_of_bound': result.get('out_of_bounds', False)
            })
    
    # Save summary CSV in Agents folder
    custom_csv_file = os.path.join(results_dir, "results_summary.csv")
    with open(custom_csv_file, 'w', newline='') as f:
        if custom_csv_data:
            fieldnames = ['map_seed', 'level', 'map_id', 'model', 'goal_reached', 'number_of_steps', 'collision', 'out_of_bound']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(custom_csv_data)
    
    print(f"\n🎉 {ppo_type.upper()} PPO TEST COMPLETE!")
    print(f"📁 Results saved to: {agent_dir}/")
    print(f"   📊 Summary CSV: {custom_csv_file}")
    print(f"   📈 Total tests run: {len(all_results)}")
    print(f"   🗺️  Total maps tested: {25 * 5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a single UAV PPO model')
    parser.add_argument('--ppo_type', type=str, choices=['vanilla', 'ar', 'ns'], default='ns',
                        help='PPO variant to test: vanilla (Pure_Neural), ar (AR_PPO), ns (Neurosymbolic) (default: ns)')
    args = parser.parse_args()
    
    print(f"🚁 Starting UAV Navigation Test for {args.ppo_type.upper()} PPO...")
    print("⚠️  This will take time to complete.")
    print("💡 Press Ctrl+C to stop at any time.\n")
    
    try:
        run_single_model_test(args.ppo_type)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}")
    
    print("\n🏁 Test finished.")
