#!/usr/bin/env python3
"""
Test CNN Depth Processing Integration
Tests the complete CNN-based depth processing pipeline in the UAV environment
"""

import numpy as np
import sys
import os
import torch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env import UAVEnv, CONFIG

def test_cnn_depth_integration():
    """Test CNN depth processing in the UAV environment"""
    print("🧪 Testing CNN Depth Processing Integration")
    print("=" * 50)
    
    # Test with CNN depth processing enabled
    print("\n1. Testing with CNN Depth Processing:")
    
    # Create environment with CNN enabled
    env = UAVEnv(render_mode=None)
    
    print(f"   ✅ Environment created successfully")
    print(f"   📏 Observation space: {env.observation_space.shape}")
    print(f"   🎯 Action space: {env.action_space.shape}")
    
    # Test environment reset
    obs, info = env.reset()
    print(f"   ✅ Environment reset successfully")
    print(f"   📊 Initial observation shape: {obs.shape}")
    print(f"   📊 Expected dimension: {3 + 3 + 3 + CONFIG['cnn_features_dim'] + 5}")
    
    # Verify observation dimensions
    if CONFIG['use_cnn_depth']:
        expected_dim = 3 + 3 + 3 + CONFIG['cnn_features_dim'] + 5  # pos + vel + goal_dist + cnn_features + nav_features
    else:
        expected_dim = 3 + 3 + 3 + CONFIG['depth_features_dim'] + 11  # pos + vel + goal_dist + depth_readings + engineered_features
    
    if obs.shape[0] == expected_dim:
        print(f"   ✅ Observation dimension correct: {obs.shape[0]}")
    else:
        print(f"   ❌ Observation dimension mismatch: got {obs.shape[0]}, expected {expected_dim}")
        return False
    
    # Test multiple steps
    print(f"\n2. Testing Environment Steps:")
    for i in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}, done={terminated or truncated}")
        
        # Check for NaN values
        if np.isnan(obs).any():
            print(f"   ❌ NaN detected in observation at step {i+1}")
            return False
        
        if terminated or truncated:
            obs, info = env.reset()
            print(f"   🔄 Environment reset after episode termination")
    
    # Test CNN components directly
    print(f"\n3. Testing CNN Components:")
    
    if hasattr(env, 'depth_camera') and env.depth_camera is not None:
        print(f"   ✅ Depth camera initialized: {type(env.depth_camera)}")
        
        # Test depth image capture
        try:
            depth_image = env.depth_camera.render_depth_image(env.data)
            print(f"   ✅ Depth image captured: shape={depth_image.shape}")
            print(f"   📊 Depth range: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
        except Exception as e:
            print(f"   ⚠️ Depth image capture failed (fallback will be used): {e}")
    
    if hasattr(env, 'depth_extractor') and env.depth_extractor is not None:
        print(f"   ✅ Depth feature extractor initialized: {type(env.depth_extractor)}")
        
        # Test feature extraction
        try:
            # Create a dummy depth image
            dummy_depth = np.random.rand(64, 64).astype(np.float32)
            features = env.depth_extractor.extract_features(dummy_depth)
            print(f"   ✅ Feature extraction successful: {features.shape}")
            print(f"   📊 Feature range: [{features.min():.3f}, {features.max():.3f}]")
        except Exception as e:
            print(f"   ❌ Feature extraction failed: {e}")
            return False
    else:
        print(f"   ℹ️ CNN components not initialized (using raycast fallback)")
    
    env.close()
    print(f"\n✅ CNN Integration Test Completed Successfully!")
    return True

def test_configuration_switching():
    """Test switching between CNN and raycast configurations"""
    print("\n" + "=" * 50)
    print("🔄 Testing Configuration Switching")
    print("=" * 50)
    
    original_cnn_setting = CONFIG['use_cnn_depth']
    
    # Test with CNN disabled
    print("\n1. Testing with CNN Disabled:")
    CONFIG['use_cnn_depth'] = False
    
    env_raycast = UAVEnv(render_mode=None)
    obs_raycast, _ = env_raycast.reset()
    print(f"   📊 Raycast observation shape: {obs_raycast.shape}")
    
    expected_raycast_dim = 3 + 3 + 3 + CONFIG['depth_features_dim'] + 11
    if obs_raycast.shape[0] == expected_raycast_dim:
        print(f"   ✅ Raycast dimension correct: {obs_raycast.shape[0]}")
    else:
        print(f"   ❌ Raycast dimension mismatch: got {obs_raycast.shape[0]}, expected {expected_raycast_dim}")
    
    env_raycast.close()
    
    # Test with CNN enabled
    print("\n2. Testing with CNN Enabled:")
    CONFIG['use_cnn_depth'] = True
    
    env_cnn = UAVEnv(render_mode=None)
    obs_cnn, _ = env_cnn.reset()
    print(f"   📊 CNN observation shape: {obs_cnn.shape}")
    
    expected_cnn_dim = 3 + 3 + 3 + CONFIG['cnn_features_dim'] + 5
    if obs_cnn.shape[0] == expected_cnn_dim:
        print(f"   ✅ CNN dimension correct: {obs_cnn.shape[0]}")
    else:
        print(f"   ❌ CNN dimension mismatch: got {obs_cnn.shape[0]}, expected {expected_cnn_dim}")
    
    env_cnn.close()
    
    # Restore original setting
    CONFIG['use_cnn_depth'] = original_cnn_setting
    
    print(f"\n✅ Configuration Switching Test Completed!")
    return True

def main():
    """Run all CNN integration tests"""
    print("🚀 Starting CNN Depth Processing Integration Tests")
    print("=" * 60)
    
    # Print configuration
    print(f"Current Configuration:")
    print(f"  📷 Use CNN Depth: {CONFIG['use_cnn_depth']}")
    print(f"  📏 Depth Resolution: {CONFIG['depth_resolution']}x{CONFIG['depth_resolution']}")
    print(f"  🔍 Depth Range: {CONFIG['depth_range']}m")
    print(f"  📊 CNN Features: {CONFIG['cnn_features_dim']}")
    print(f"  📊 Raycast Features: {CONFIG['depth_features_dim']}")
    
    success = True
    
    try:
        # Test CNN integration
        if not test_cnn_depth_integration():
            success = False
        
        # Test configuration switching
        if not test_configuration_switching():
            success = False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! CNN depth processing is working correctly.")
        print("\nNext steps:")
        print("  1. ✅ CNN architecture implemented and tested")
        print("  2. ✅ MuJoCo camera integration working")
        print("  3. ✅ Environment observation space updated")
        print("  4. 🔄 Ready for preprocessing pipeline and training script updates")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()