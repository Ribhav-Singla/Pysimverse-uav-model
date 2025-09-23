#!/usr/bin/env python3
"""
Test script to verify LIDAR functionality and boundary checking
"""

import numpy as np
from uav_env import UAVEnv

def test_lidar_and_boundaries():
    print("🧪 Testing LIDAR and Boundary Detection System")
    print("=" * 50)
    
    # Create environment
    env = UAVEnv()
    obs, _ = env.reset()
    
    print(f"✅ Environment initialized")
    print(f"📊 Observation space shape: {env.observation_space.shape}")
    print(f"📊 Observation breakdown:")
    print(f"   - Position: 3 values")
    print(f"   - Velocity: 3 values") 
    print(f"   - Goal distance: 3 values")
    print(f"   - LIDAR readings: 16 values")
    print(f"   - Total: {3+3+3+16} = {env.observation_space.shape[0]}")
    
    # Test initial observation
    print(f"\n📡 Initial LIDAR readings (first 8 of 16):")
    lidar_data = obs[9:17]  # LIDAR starts at index 9
    for i in range(8):
        angle = (360 * i) // 16
        print(f"   {angle:3d}°: {lidar_data[i]:.2f}m")
    
    # Test boundary detection
    print(f"\n🚨 Testing boundary detection...")
    
    # Simulate UAV at different positions
    test_positions = [
        np.array([0, 0, 1.8]),      # Center (safe)
        np.array([3.9, 0, 1.8]),    # Near +X boundary 
        np.array([-4.1, 0, 1.8]),   # Outside -X boundary
        np.array([0, 4.1, 1.8]),    # Outside +Y boundary
        np.array([0, 0, -0.1]),     # Below ground
        np.array([0, 0, 6.0]),      # Too high
    ]
    
    position_names = [
        "Center (safe)",
        "Near +X boundary", 
        "Outside -X boundary",
        "Outside +Y boundary", 
        "Below ground",
        "Too high"
    ]
    
    for pos, name in zip(test_positions, position_names):
        is_out = env._check_out_of_bounds(pos)
        status = "❌ OUT OF BOUNDS" if is_out else "✅ SAFE"
        print(f"   {name:20} {pos} -> {status}")
    
    # Test action space
    print(f"\n🎮 Testing action space...")
    action = env.action_space.sample()
    print(f"   Sample action: {action}")
    print(f"   Action range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    
    # Take a few steps
    print(f"\n🚁 Taking test steps...")
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos = obs[:3]
        lidar_min = np.min(obs[9:25])  # Min LIDAR reading
        
        print(f"   Step {step+1}: Pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
              f"Reward={reward:.1f} Min_LIDAR={lidar_min:.2f}m Terminated={terminated}")
        
        if terminated:
            print(f"     Episode terminated!")
            break
    
    env.close()
    print(f"\n✅ LIDAR and boundary testing completed!")

if __name__ == "__main__":
    test_lidar_and_boundaries()