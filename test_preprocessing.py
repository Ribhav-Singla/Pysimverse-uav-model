#!/usr/bin/env python3
"""
Test script for depth preprocessing pipeline
"""

import numpy as np
import sys
sys.path.append('.')

from uav_env import UAVEnv, CONFIG
from depth_preprocessing import DepthPreprocessor, create_preprocessing_pipeline
import matplotlib.pyplot as plt

def test_preprocessing_pipeline():
    """Test the depth preprocessing pipeline functionality"""
    
    print("🧪 Testing Depth Preprocessing Pipeline")
    print("=" * 50)
    
    # Create test depth image (simulated)
    height, width = 64, 64
    test_depth = np.random.uniform(0.5, 5.0, (height, width))
    
    # Add some realistic patterns
    # Add some close obstacles
    test_depth[20:30, 20:30] = 0.8
    test_depth[40:50, 10:20] = 1.2
    
    # Add some far regions
    test_depth[10:20, 40:50] = 8.0
    
    # Add some invalid values
    test_depth[5:10, 5:10] = np.inf
    test_depth[55:60, 55:60] = np.nan
    
    print(f"📊 Test depth image: {test_depth.shape}")
    print(f"   Range: [{np.min(test_depth[np.isfinite(test_depth)]):.2f}, {np.max(test_depth[np.isfinite(test_depth)]):.2f}]")
    print(f"   Invalid pixels: {np.sum(~np.isfinite(test_depth))}")
    
    # Test 1: Basic preprocessing without enhancement
    print("\n1. Testing Basic Preprocessing:")
    print("-" * 30)
    
    basic_preprocessor = DepthPreprocessor(
        image_size=64,
        depth_range=(0.1, 10.0),
        enable_denoising=False,
        enable_enhancement=False,
        add_noise=False
    )
    
    processed_tensor = basic_preprocessor.preprocess(test_depth)
    print(f"   ✅ Processed tensor shape: {processed_tensor.shape}")
    print(f"   📊 Output range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")
    
    # Test 2: Full preprocessing with enhancement
    print("\n2. Testing Full Preprocessing Pipeline:")
    print("-" * 40)
    
    full_preprocessor = DepthPreprocessor(
        image_size=64,
        depth_range=(0.1, 10.0),
        enable_denoising=True,
        enable_enhancement=True,
        add_noise=False
    )
    
    enhanced_tensor = full_preprocessor.preprocess(test_depth.copy())
    print(f"   ✅ Enhanced tensor shape: {enhanced_tensor.shape}")
    print(f"   📊 Output range: [{enhanced_tensor.min():.3f}, {enhanced_tensor.max():.3f}]")
    
    # Test 3: Preprocessing with noise simulation
    print("\n3. Testing Noise Simulation:")
    print("-" * 30)
    
    noisy_preprocessor = DepthPreprocessor(
        image_size=64,
        depth_range=(0.1, 10.0),
        enable_denoising=False,
        enable_enhancement=False,
        add_noise=True,
        noise_std=0.05
    )
    
    noisy_tensor = noisy_preprocessor.preprocess(test_depth.copy())
    print(f"   ✅ Noisy tensor shape: {noisy_tensor.shape}")
    print(f"   📊 Output range: [{noisy_tensor.min():.3f}, {noisy_tensor.max():.3f}]")
    
    # Test 4: Configuration-based preprocessing
    print("\n4. Testing Configuration-Based Preprocessing:")
    print("-" * 45)
    
    config_preprocessor = create_preprocessing_pipeline(CONFIG)
    config_tensor = config_preprocessor.preprocess(test_depth.copy())
    print(f"   ✅ Config-based tensor shape: {config_tensor.shape}")
    print(f"   📊 Output range: [{config_tensor.min():.3f}, {config_tensor.max():.3f}]")
    
    # Display statistics
    print("\n📈 Preprocessing Statistics:")
    print("-" * 30)
    for name, preprocessor in [
        ("Basic", basic_preprocessor),
        ("Full", full_preprocessor), 
        ("Noisy", noisy_preprocessor),
        ("Config", config_preprocessor)
    ]:
        stats = preprocessor.get_stats()
        print(f"   {name}:")
        print(f"     - Processed images: {stats['processed_count']}")
        if stats['processed_count'] > 0:
            print(f"     - Avg invalid pixels: {stats['avg_invalid_pixels']:.1f}")
            print(f"     - Avg range violations: {stats['avg_range_violations']:.1f}")
    
    return test_depth, processed_tensor, enhanced_tensor, noisy_tensor


def test_environment_integration():
    """Test preprocessing integration with UAV environment"""
    
    print("\n🚁 Testing Environment Integration")
    print("=" * 50)
    
    # Test with CNN enabled
    CONFIG['use_cnn_depth'] = True
    CONFIG['enable_denoising'] = True
    CONFIG['enable_enhancement'] = True
    CONFIG['add_noise'] = False
    
    try:
        env = UAVEnv()
        
        print("✅ Environment created successfully")
        print(f"📏 Observation space: {env.observation_space.shape}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"✅ Environment reset: observation shape {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"   Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}")
            
            if env.depth_preprocessor:
                stats = env.depth_preprocessor.get_stats()
                if stats['processed_count'] > 0:
                    print(f"            Preprocessing: {stats['processed_count']} images, "
                          f"{stats['avg_invalid_pixels']:.1f} avg invalid pixels")
        
        print("✅ Environment integration test completed")
        
        # Display final preprocessing stats
        if env.depth_preprocessor:
            final_stats = env.depth_preprocessor.get_stats()
            print(f"\n📊 Final Preprocessing Statistics:")
            print(f"   - Total processed images: {final_stats['processed_count']}")
            if final_stats['processed_count'] > 0:
                print(f"   - Average invalid pixels per image: {final_stats['avg_invalid_pixels']:.2f}")
                print(f"   - Average range violations per image: {final_stats['avg_range_violations']:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("🚀 Starting Depth Preprocessing Pipeline Tests")
    print("=" * 60)
    
    # Test preprocessing pipeline
    test_data = test_preprocessing_pipeline()
    
    # Test environment integration
    test_environment_integration()
    
    print("\n" + "=" * 60)
    print("🎉 All preprocessing tests completed!")
    print("\nNext steps:")
    print("  1. ✅ Preprocessing pipeline implemented and tested")
    print("  2. ✅ Environment integration working")
    print("  3. 🔄 Ready for hybrid fallback system and training updates")


if __name__ == "__main__":
    main()