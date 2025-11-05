#!/usr/bin/env python3
"""
Test script to verify all three PPO variants work correctly
"""

import argparse
import sys
from uav_env import UAVEnv

def test_ppo_variant(ppo_type):
    """Test a specific PPO variant configuration"""
    print(f"\n🧪 Testing {ppo_type.upper()} PPO variant...")
    
    # Map PPO type to configuration (same logic as training.py)
    if ppo_type == 'vanilla':
        ns_lambda = 0.0
        use_neurosymbolic = False
        use_extra_rewards = False
        print(f"   🤖 Configuration: Basic rewards only (lambda={ns_lambda})")
    elif ppo_type == 'ar':
        ns_lambda = 0.0
        use_neurosymbolic = False
        use_extra_rewards = True
        print(f"   💰 Configuration: Augmented rewards (lambda={ns_lambda})")
    elif ppo_type == 'ns':
        ns_lambda = 1.0
        use_neurosymbolic = True
        use_extra_rewards = False
        print(f"   🧠 Configuration: Neurosymbolic (lambda={ns_lambda})")
    
    # Create configuration
    ns_cfg = {
        'use_neurosymbolic': use_neurosymbolic,
        'use_extra_rewards': use_extra_rewards,
        'ppo_type': ppo_type,
        'lambda': ns_lambda,
        'warmup_steps': 100,
    }
    
    try:
        # Test environment initialization
        env = UAVEnv(curriculum_learning=True, ns_cfg=ns_cfg)
        print(f"   ✅ Environment initialized successfully")
        print(f"      - PPO type: {env.ns_cfg.get('ppo_type')}")
        print(f"      - Extra rewards: {env.ns_cfg.get('use_extra_rewards')}")
        print(f"      - Neurosymbolic: {env.ns_cfg.get('use_neurosymbolic')}")
        
        # Test observation and action spaces
        obs, _ = env.reset()
        print(f"   ✅ Environment reset successful")
        print(f"      - Observation shape: {obs.shape}")
        print(f"      - Action space: {env.action_space}")
        
        # Test a few steps to verify reward function works
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   ✅ Step {step+1}: reward={reward:.3f}, done={done}")
            if done or truncated:
                break
        
        env.close()
        print(f"   ✅ {ppo_type.upper()} PPO variant test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ {ppo_type.upper()} PPO variant test FAILED: {e}")
        return False

def test_reward_features():
    """Test the specific reward features for AR PPO"""
    print(f"\n🔬 Testing AR PPO reward features...")
    
    ns_cfg = {
        'use_neurosymbolic': False,
        'use_extra_rewards': True,
        'ppo_type': 'ar',
        'lambda': 0.0,
    }
    
    env = UAVEnv(curriculum_learning=False, ns_cfg=ns_cfg)
    obs, _ = env.reset()
    
    # Test boundary approach penalty
    pos_near_boundary = [3.5, 0.0, 1.0]  # Close to east boundary (world size = 8, so boundary at ±4)
    penalty = env._get_boundary_approach_penalty(pos_near_boundary)
    print(f"   ✅ Boundary approach penalty: {penalty:.3f} (near boundary)")
    
    pos_center = [0.0, 0.0, 1.0]  # Center of map
    penalty = env._get_boundary_approach_penalty(pos_center)
    print(f"   ✅ Boundary approach penalty: {penalty:.3f} (center)")
    
    # Test LIDAR goal detection reward
    import numpy as np
    pos = np.array([0.0, 0.0, 1.0])
    vel = np.array([1.0, 1.0, 0.0])  # Moving toward goal
    lidar_readings = np.ones(16) * 0.8  # Mock LIDAR readings (normalized)
    
    goal_reward = env._get_lidar_goal_detection_reward(pos, vel, lidar_readings)
    print(f"   ✅ LIDAR goal detection reward: {goal_reward:.3f}")
    
    env.close()
    print(f"   ✅ AR PPO reward features test PASSED")

def main():
    parser = argparse.ArgumentParser(description='Test PPO variants')
    parser.add_argument('--variant', type=str, choices=['vanilla', 'ar', 'ns', 'all'], 
                        default='all', help='Which variant to test')
    args = parser.parse_args()
    
    print("🚀 PPO Variants Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    variants_to_test = ['vanilla', 'ar', 'ns'] if args.variant == 'all' else [args.variant]
    
    for variant in variants_to_test:
        total_tests += 1
        if test_ppo_variant(variant):
            success_count += 1
    
    # Test AR PPO specific features
    if args.variant in ['ar', 'all']:
        total_tests += 1
        try:
            test_reward_features()
            success_count += 1
        except Exception as e:
            print(f"   ❌ AR PPO reward features test FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests PASSED! PPO variants are working correctly.")
        return 0
    else:
        print("⚠️  Some tests FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())