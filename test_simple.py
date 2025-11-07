#!/usr/bin/env python3
"""
Simple test to check if depth sensor conversion works
"""

print("Testing depth sensor conversion...")

try:
    import numpy as np
    from uav_env import UAVEnv, CONFIG
    print("✅ Imports successful")
    
    print(f"✅ Depth range: {CONFIG['depth_range']}")
    print(f"✅ Depth features dimension: {CONFIG['depth_features_dim']}")
    print(f"✅ Depth resolution: {CONFIG['depth_resolution']}")
    print(f"✅ Depth FOV: {CONFIG['depth_fov']}°")
    
    # Test environment creation
    env = UAVEnv(render_mode=None, ns_cfg={'ppo_type': 'vanilla', 'use_extra_rewards': False})
    print("✅ Environment creation successful")
    
    # Test observation space
    obs, info = env.reset()
    print(f"✅ Environment reset successful, observation shape: {obs.shape}")
    
    # Test depth sensor function exists
    pos = obs[:3]  # Get position from observation
    depth_readings = env._get_depth_readings(pos)
    print(f"✅ Depth sensor reading successful, shape: {depth_readings.shape}")
    print(f"✅ Depth readings range: min={np.min(depth_readings):.3f}, max={np.max(depth_readings):.3f}")
    
    env.close()
    print("✅ All tests passed! Depth sensor conversion is working.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()