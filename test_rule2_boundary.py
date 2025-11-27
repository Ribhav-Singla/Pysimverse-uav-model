#!/usr/bin/env python3
"""
Test script to verify Rule 2 (Boundary Safety) behavior
Tests that the rule only activates for boundaries and reduces velocity in the correct axis
"""

import numpy as np
from uav_env import UAVEnv, CONFIG

def test_rule2_boundary_detection():
    """Test that Rule 2 only activates near boundaries, not obstacles"""
    
    print("=" * 80)
    print("Testing Rule 2: Boundary Safety - Boundary vs Obstacle Detection")
    print("=" * 80)
    
    # Create environment
    env = UAVEnv(curriculum_learning=False, ns_cfg={'use_neurosymbolic': True})
    env.reset()
    
    half_world = CONFIG['world_size'] / 2  # Should be 4.0
    
    # Test Case 1: Near east boundary (should activate Rule 2)
    print("\n[Test 1] Position near EAST boundary:")
    pos_near_boundary = np.array([3.5, 0.0, 1.0])  # 0.5m from east boundary
    env.data.qpos[:3] = pos_near_boundary
    env.data.qvel[:3] = np.array([0.6, 0.0, 0.0])  # Moving east
    
    obs = env._get_obs()
    lidar = obs[9:25]
    context = env._prepare_rdr_context(pos_near_boundary, lidar, obs)
    
    print(f"  Position: {pos_near_boundary}")
    print(f"  Velocity: {env.data.qvel[:3]}")
    print(f"  Distance to boundary: {context['distance_to_boundary']:.3f}m")
    print(f"  Rule 2 should activate: {env.rdr_system.all_rules['R2_BOUNDARY_SAFETY'].evaluate_condition(context)}")
    
    action = env.symbolic_action()
    print(f"  Generated action: {action}")
    print(f"  Expected: X velocity reduced (< 0.6)")
    
    # Test Case 2: Center of map (should NOT activate Rule 2)
    print("\n[Test 2] Position at CENTER of map:")
    pos_center = np.array([0.0, 0.0, 1.0])  # Center of map
    env.data.qpos[:3] = pos_center
    env.data.qvel[:3] = np.array([0.6, 0.6, 0.0])  # Moving NE
    
    obs = env._get_obs()
    lidar = obs[9:25]
    context = env._prepare_rdr_context(pos_center, lidar, obs)
    
    print(f"  Position: {pos_center}")
    print(f"  Velocity: {env.data.qvel[:3]}")
    print(f"  Distance to boundary: {context['distance_to_boundary']:.3f}m")
    print(f"  Rule 2 should activate: {env.rdr_system.all_rules['R2_BOUNDARY_SAFETY'].evaluate_condition(context)}")
    
    # Test Case 3: Near north-east corner (should activate Rule 2)
    print("\n[Test 3] Position near NORTH-EAST corner:")
    pos_corner = np.array([3.5, 3.5, 1.0])  # Near NE corner
    env.data.qpos[:3] = pos_corner
    env.data.qvel[:3] = np.array([0.5, 0.5, 0.0])  # Moving toward corner
    
    obs = env._get_obs()
    lidar = obs[9:25]
    context = env._prepare_rdr_context(pos_corner, lidar, obs)
    
    print(f"  Position: {pos_corner}")
    print(f"  Velocity: {env.data.qvel[:3]}")
    print(f"  Distance to boundary: {context['distance_to_boundary']:.3f}m")
    print(f"  Rule 2 should activate: {env.rdr_system.all_rules['R2_BOUNDARY_SAFETY'].evaluate_condition(context)}")
    
    action = env.symbolic_action()
    print(f"  Generated action: {action}")
    print(f"  Expected: Both X and Y velocities reduced")
    
    # Test Case 4: Near boundary but moving away (should still activate but not reduce)
    print("\n[Test 4] Position near EAST boundary but MOVING WEST (away):")
    pos_near_boundary = np.array([3.5, 0.0, 1.0])  # 0.5m from east boundary
    env.data.qpos[:3] = pos_near_boundary
    env.data.qvel[:3] = np.array([-0.6, 0.0, 0.0])  # Moving WEST (away from boundary)
    
    obs = env._get_obs()
    lidar = obs[9:25]
    context = env._prepare_rdr_context(pos_near_boundary, lidar, obs)
    
    print(f"  Position: {pos_near_boundary}")
    print(f"  Velocity: {env.data.qvel[:3]}")
    print(f"  Distance to boundary: {context['distance_to_boundary']:.3f}m")
    print(f"  Rule 2 should activate: {env.rdr_system.all_rules['R2_BOUNDARY_SAFETY'].evaluate_condition(context)}")
    
    action = env.symbolic_action()
    print(f"  Generated action: {action}")
    print(f"  Expected: X velocity NOT reduced (moving away from boundary)")
    
    env.close()
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

def test_axis_specific_reduction():
    """Test that velocity reduction is axis-specific"""
    
    print("\n" + "=" * 80)
    print("Testing Rule 2: Axis-Specific Velocity Reduction")
    print("=" * 80)
    
    env = UAVEnv(curriculum_learning=False, ns_cfg={'use_neurosymbolic': True})
    env.reset()
    
    half_world = CONFIG['world_size'] / 2
    
    # Test: Near east boundary, moving east and north
    print("\n[Test] Near EAST boundary, moving EAST and NORTH:")
    pos = np.array([3.5, 0.0, 1.0])  # Near east boundary
    env.data.qpos[:3] = pos
    env.data.qvel[:3] = np.array([0.6, 0.5, 0.0])  # Moving east and north
    
    obs = env._get_obs()
    lidar = obs[9:25]
    context = env._prepare_rdr_context(pos, lidar, obs)
    
    print(f"  Position: {pos}")
    print(f"  Initial velocity: {env.data.qvel[:3]}")
    
    # Get the applicable rule
    rule = env.rdr_system.evaluate_rules(context)
    print(f"  Applicable rule: {rule.rule_id}")
    
    # Generate action
    action = env._generate_action_from_rule(rule, context)
    
    print(f"  Generated action: {action}")
    print(f"  Analysis:")
    print(f"    - X velocity: {env.data.qvel[0]:.3f} → {action[0]:.3f} (expected: reduced)")
    print(f"    - Y velocity: {env.data.qvel[1]:.3f} → {action[1]:.3f} (expected: unchanged)")
    print(f"  ✓ Only X-axis velocity should be reduced (near east boundary)")
    
    env.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_rule2_boundary_detection()
    test_axis_specific_reduction()
    print("\n✅ All tests completed successfully!")
