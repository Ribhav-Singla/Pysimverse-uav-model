"""
Debug script to analyze UAV navigation issues
"""
import numpy as np
import torch
from uav_env import UAVEnv, CONFIG
from ppo_agent import PPOAgent
import os

def test_trained_model():
    """Test the trained model and analyze its behavior"""
    print("🔍 Loading trained model for analysis...")
    
    # Load environment
    env = UAVEnv(render_mode=None, curriculum_learning=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load PPO agent
    ppo_agent = PPOAgent(state_dim, action_dim, 0.0003, 0.0009, 0.99, 4, 0.2, 0.6)
    
    # Load pre-trained weights
    checkpoint_path = "PPO_preTrained/UAVEnv/PPO_UAV_Weights.pth"
    if os.path.exists(checkpoint_path):
        ppo_agent.load(checkpoint_path)
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print("❌ No pre-trained weights found!")
        return
    
    # Run multiple test episodes
    num_test_episodes = 5
    results = []
    
    for episode in range(num_test_episodes):
        print(f"\n📊 Testing Episode {episode + 1}/{num_test_episodes}")
        
        obs, _ = env.reset()
        episode_data = {
            'start_pos': CONFIG['start_pos'].copy(),
            'goal_pos': CONFIG['goal_pos'].copy(),
            'positions': [obs[:3].copy()],
            'actions': [],
            'velocities': [],
            'goal_distances': [],
            'rewards': [],
            'lidar_readings': [],
            'episode_reward': 0,
            'steps': 0,
            'terminated': False,
            'termination_reason': 'unknown'
        }
        
        print(f"Start: {CONFIG['start_pos']}")
        print(f"Goal: {CONFIG['goal_pos']}")
        
        max_steps = 1000  # Limit for analysis
        for step in range(max_steps):
            # Get action from trained model
            action, _ = ppo_agent.select_action(obs)
            
            # Store data
            episode_data['actions'].append(np.array(action).flatten())
            episode_data['velocities'].append(obs[3:6].copy())
            episode_data['goal_distances'].append(np.linalg.norm(CONFIG['goal_pos'] - obs[:3]))
            episode_data['lidar_readings'].append(obs[9:25].copy())
            
            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_data['positions'].append(obs[:3].copy())
            episode_data['rewards'].append(reward)
            episode_data['episode_reward'] += reward
            episode_data['steps'] += 1
            
            if terminated or truncated:
                episode_data['terminated'] = True
                if hasattr(env, 'last_termination_info'):
                    episode_data['termination_reason'] = env.last_termination_info.get('termination_reason', 'unknown')
                break
        
        # Analysis
        final_pos = episode_data['positions'][-1]
        goal_distance = np.linalg.norm(CONFIG['goal_pos'] - final_pos)
        
        print(f"Final position: {final_pos}")
        print(f"Final distance to goal: {goal_distance:.3f}m")
        print(f"Total reward: {episode_data['episode_reward']:.2f}")
        print(f"Steps taken: {episode_data['steps']}")
        print(f"Termination reason: {episode_data['termination_reason']}")
        
        # Check if UAV moved toward goal
        initial_distance = episode_data['goal_distances'][0]
        final_distance = episode_data['goal_distances'][-1]
        progress = initial_distance - final_distance
        print(f"Progress toward goal: {progress:.3f}m ({'✅' if progress > 0 else '❌'})")
        
        results.append(episode_data)
    
    return results

def analyze_results(results):
    """Analyze the test results to identify issues"""
    print("\n📈 ANALYSIS SUMMARY")
    print("=" * 50)
    
    for i, episode in enumerate(results):
        print(f"\nEpisode {i+1}:")
        
        # Calculate movement statistics
        positions = np.array(episode['positions'])
        actions = np.array(episode['actions'])
        
        # Movement analysis
        total_distance_moved = 0
        for j in range(1, len(positions)):
            total_distance_moved += np.linalg.norm(positions[j] - positions[j-1])
        
        # Goal direction analysis
        start_pos = positions[0]
        goal_pos = episode['goal_pos']
        goal_direction = goal_pos - start_pos
        goal_direction_norm = goal_direction / np.linalg.norm(goal_direction)
        
        # Check if actions are generally toward goal
        total_action_alignment = 0
        for action in actions:
            # Ensure action is 1D array and get X,Y components
            action_flat = np.array(action).flatten()
            action_2d = action_flat[:2]  # Only X,Y components
            goal_dir_2d = goal_direction_norm[:2]
            if np.linalg.norm(action_2d) > 0.01:  # Avoid division by zero
                action_norm = action_2d / np.linalg.norm(action_2d)
                alignment = np.dot(action_norm, goal_dir_2d)
                total_action_alignment += alignment
        
        avg_action_alignment = total_action_alignment / len(actions) if len(actions) > 0 else 0
        
        print(f"  - Total distance moved: {total_distance_moved:.3f}m")
        print(f"  - Average action alignment with goal: {avg_action_alignment:.3f}")
        print(f"  - Goal distance reduction: {episode['goal_distances'][0] - episode['goal_distances'][-1]:.3f}m")
        print(f"  - Final termination: {episode['termination_reason']}")
        
        # Check for potential issues
        if avg_action_alignment < 0.1:
            print(f"  ⚠️  ISSUE: Actions not aligned with goal direction!")
        
        if total_distance_moved < 1.0:
            print(f"  ⚠️  ISSUE: UAV barely moved!")
        
        if episode['goal_distances'][-1] > episode['goal_distances'][0] * 0.9:
            print(f"  ⚠️  ISSUE: UAV didn't make significant progress toward goal!")
            
        # Check action magnitudes
        action_magnitudes = [np.linalg.norm(np.array(a).flatten()[:2]) for a in actions]
        avg_action_magnitude = np.mean(action_magnitudes)
        print(f"  - Average action magnitude: {avg_action_magnitude:.3f}")
        
        if avg_action_magnitude < 0.5:
            print(f"  ⚠️  ISSUE: Actions too small - UAV might be moving very slowly!")

def plot_trajectories(results):
    """Print trajectory analysis (plotting disabled)"""
    print("\n📊 TRAJECTORY ANALYSIS")
    print("=" * 50)
    
    for i, episode in enumerate(results):
        positions = np.array(episode['positions'])
        print(f"\nEpisode {i+1} Trajectory:")
        print(f"  Start: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f})")
        print(f"  End: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")
        print(f"  Goal: ({episode['goal_pos'][0]:.2f}, {episode['goal_pos'][1]:.2f})")
        
        # Calculate trajectory statistics
        distances_moved = []
        for j in range(1, len(positions)):
            dist = np.linalg.norm(positions[j] - positions[j-1])
            distances_moved.append(dist)
        
        total_movement = sum(distances_moved)
        avg_step_size = np.mean(distances_moved) if distances_moved else 0
        
        print(f"  Total movement: {total_movement:.3f}m")
        print(f"  Average step size: {avg_step_size:.4f}m")
        print(f"  Steps taken: {len(positions)-1}")
        
        # Check if trajectory is reasonable
        if total_movement < 0.5:
            print(f"  ⚠️  Very little movement detected!")
        if avg_step_size < 0.001:
            print(f"  ⚠️  Extremely small steps - UAV might be stuck!")

if __name__ == "__main__":
    print("🔍 UAV Navigation Debug Analysis")
    print("=" * 40)
    
    # Test the trained model
    results = test_trained_model()
    
    if results:
        # Analyze results
        analyze_results(results)
        
        # Plot trajectories
        plot_trajectories(results)
    else:
        print("❌ Could not load model for analysis")