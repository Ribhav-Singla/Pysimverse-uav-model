#!/usr/bin/env python3
"""
Test script for RDR (Ripple Down Rules) integration in UAV environment
"""

import numpy as np
from uav_env import UAVEnv

def test_rdr_system():
    """Test the RDR system integration"""
    print("🧪 Testing RDR (Ripple Down Rules) System Integration")
    print("=" * 60)
    
    # Create environment with RDR-enabled neurosymbolic configuration
    ns_cfg = {
        'use_neurosymbolic': True,
        'lambda': 1.0,  # Full symbolic control
        'warmup_steps': 0,
        'debug_rdr': True  # Enable debug output
    }
    
    env = UAVEnv(curriculum_learning=False, ns_cfg=ns_cfg)
    print(f"✅ Environment created with RDR system")
    
    # Display initial RDR hierarchy
    print("\n🌳 Initial RDR Rule Hierarchy:")
    env.print_rdr_hierarchy()
    
    # Run a few test episodes
    for episode in range(3):
        print(f"\n🎮 Episode {episode + 1}")
        print("-" * 30)
        
        obs, _ = env.reset()
        env.current_episode = episode
        done = False
        step_count = 0
        rules_used = set()
        
        while not done and step_count < 100:  # Limit steps for testing
            # Get RDR-based action
            action = env.symbolic_action()
            
            # Track which rule was used
            if env.current_rule:
                rules_used.add(env.current_rule.rule_id)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Print first few steps for debugging
            if step_count <= 5:
                pos = obs[:3]
                from uav_env import CONFIG  # Import CONFIG
                goal_dist = np.linalg.norm(env.data.qpos[:3] - CONFIG['goal_pos'])
                rule_id = env.current_rule.rule_id if env.current_rule else "None"
                print(f"  Step {step_count}: Rule={rule_id}, Pos=({pos[0]:.2f},{pos[1]:.2f}), Goal_Dist={goal_dist:.2f}")
        
        # Episode summary
        final_pos = obs[:3]
        from uav_env import CONFIG  # Import CONFIG
        final_goal_dist = np.linalg.norm(final_pos - CONFIG['goal_pos'])
        termination_reason = env.last_termination_info.get('termination_reason', 'unknown')
        
        print(f"  Episode completed: {step_count} steps")
        print(f"  Final goal distance: {final_goal_dist:.2f}")
        print(f"  Termination reason: {termination_reason}")
        print(f"  Rules used: {', '.join(sorted(rules_used))}")
    
    # Display RDR statistics
    print(f"\n📊 RDR Rule Performance Statistics:")
    stats = env.get_rdr_statistics()
    for rule_id, rule_stats in stats.items():
        usage = rule_stats['usage_count']
        success_rate = rule_stats['success_rate']
        if usage > 0:
            print(f"  {rule_id}: {usage} uses, {success_rate:.2f} success rate")
    
    # Display final hierarchy with performance
    print(f"\n🌳 Final RDR Rule Hierarchy (with performance):")
    env.print_rdr_hierarchy()
    
    env.close()
    print(f"\n✅ RDR system test completed!")

def test_rdr_conditions():
    """Test individual RDR conditions"""
    print(f"\n🔍 Testing RDR Condition Functions:")
    print("-" * 40)
    
    env = UAVEnv(curriculum_learning=False)
    rdr = env.rdr_system
    
    # Test different context scenarios
    test_contexts = [
        {
            'name': 'Clear Path Scenario',
            'context': {
                'has_los_to_goal': True,
                'min_obstacle_dist': 2.0,
                'distance_to_boundary': 3.0,
                'goal_distance': 4.0,
                'num_blocked_directions': 0
            }
        },
        {
            'name': 'Near Obstacle Scenario',
            'context': {
                'has_los_to_goal': True,
                'min_obstacle_dist': 0.6,  # Close to obstacle
                'distance_to_boundary': 3.0,
                'goal_distance': 4.0,
                'num_blocked_directions': 1
            }
        },
        {
            'name': 'Blocked Path Scenario',
            'context': {
                'has_los_to_goal': False,
                'min_obstacle_dist': 1.5,
                'distance_to_boundary': 3.0,
                'goal_distance': 4.0,
                'num_blocked_directions': 2
            }
        },
        {
            'name': 'Emergency Scenario',
            'context': {
                'has_los_to_goal': False,
                'min_obstacle_dist': 0.2,  # Very close!
                'distance_to_boundary': 3.0,
                'goal_distance': 4.0,
                'num_blocked_directions': 3
            }
        },
        {
            'name': 'Cornered Scenario',
            'context': {
                'has_los_to_goal': False,
                'min_obstacle_dist': 1.0,
                'distance_to_boundary': 0.5,  # Near boundary
                'goal_distance': 4.0,
                'num_blocked_directions': 4  # Blocked in many directions
            }
        }
    ]
    
    for test_case in test_contexts:
        print(f"\n  📋 {test_case['name']}:")
        context = test_case['context']
        
        # Evaluate which rule applies
        applicable_rule = rdr.evaluate_rules(context)
        print(f"    → Applied Rule: {applicable_rule.rule_id}")
        print(f"    → Conclusion: {applicable_rule.conclusion}")
        print(f"    → Action Params: {applicable_rule.action_params}")
    
    env.close()

if __name__ == "__main__":
    test_rdr_conditions()
    test_rdr_system()