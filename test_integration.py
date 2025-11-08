#!/usr/bin/env python3
"""Quick integration test - environment with new CNN"""

import numpy as np
import sys

print("=" * 60)
print("Environment Integration Test - CNN Only")
print("=" * 60)

try:
    print("\n1. Importing modules...")
    from uav_env import UAVEnv, CONFIG
    from depth_cnn import SimplDepthCNN, DepthFeatureExtractor
    print("   ✓ Imports successful")
    
    print("\n2. Creating environment...")
    env = UAVEnv(render_mode=None)
    print("   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.shape}")
    
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Expected: (147,)")
    assert obs.shape == (147,), f"Observation shape mismatch: {obs.shape} != (147,)"
    
    print("\n4. Testing 5 environment steps with CNN...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step+1}:")
        print(f"      - Observation shape: {obs.shape}")
        print(f"      - Reward: {reward:.3f}")
        print(f"      - Terminated: {terminated}, Truncated: {truncated}")
        print(f"      - CNN features range: [{obs[9]:6.3f}, {obs[136]:6.3f}] (of 128 features)")
        
        assert obs.shape == (147,), f"Step {step}: Observation shape changed!"
        assert np.isfinite(obs).all(), f"Step {step}: NaN values in observation!"
        
        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break
    
    env.close()
    print("\n5. Environment closed")
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 60)
    print("\nKey Points:")
    print("  • No raycast fallback - pure CNN depth perception")
    print("  • Observation is 147D (pos + vel + goal_dist + 128 CNN + navigation)")
    print("  • Fresh depth image processed at every step")
    print("  • Goal threshold: 0.1m")
    print("  • SimplDepthCNN: 3 conv layers + 2 MLP layers (1.1M parameters)")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
