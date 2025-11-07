#!/usr/bin/env python3
"""
Final Validation Suite for CNN Depth Processing System
Comprehensive validation of the complete implementation
"""

import numpy as np
import torch
import sys
import os
import time
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env import UAVEnv, CONFIG
from depth_cnn import DepthCNN, DepthFeatureExtractor
from depth_preprocessing import DepthPreprocessor, create_preprocessing_pipeline


def test_core_components():
    """Test all core components independently"""
    print("🔧 Testing Core Components")
    print("-" * 40)
    
    success = True
    
    # Test 1: CNN Architecture
    print("1. Testing CNN Architecture...")
    try:
        cnn = DepthCNN(output_features=128)
        test_input = torch.randn(2, 1, 64, 64)
        output = cnn(test_input)
        
        assert output.shape == (2, 128), f"Expected (2, 128), got {output.shape}"
        assert torch.isfinite(output).all(), "CNN output contains non-finite values"
        
        print("   ✅ CNN architecture working correctly")
        
    except Exception as e:
        print(f"   ❌ CNN architecture failed: {e}")
        success = False
    
    # Test 2: Feature Extractor
    print("2. Testing Feature Extractor...")
    try:
        extractor = DepthFeatureExtractor(model_path=None)
        test_image = np.random.uniform(0.5, 5.0, (64, 64))
        features = extractor.extract_features(test_image)
        
        assert features.shape == (128,), f"Expected (128,), got {features.shape}"
        assert np.isfinite(features).all(), "Features contain non-finite values"
        
        print("   ✅ Feature extractor working correctly")
        
    except Exception as e:
        print(f"   ❌ Feature extractor failed: {e}")
        success = False
    
    # Test 3: Preprocessing Pipeline
    print("3. Testing Preprocessing Pipeline...")
    try:
        preprocessor = DepthPreprocessor(
            image_size=64,
            depth_range=(0.1, 10.0),
            enable_enhancement=False,  # Keep simple for testing
            enable_denoising=False
        )
        
        test_image = np.random.uniform(0.5, 5.0, (64, 64))
        # Add some problematic values
        test_image[0:5, 0:5] = np.nan
        test_image[10:15, 10:15] = np.inf
        
        result = preprocessor.preprocess(test_image)
        
        assert result.shape == (1, 1, 64, 64), f"Expected (1, 1, 64, 64), got {result.shape}"
        assert torch.isfinite(result).all(), "Preprocessed output contains non-finite values"
        
        print("   ✅ Preprocessing pipeline working correctly")
        
    except Exception as e:
        print(f"   ❌ Preprocessing pipeline failed: {e}")
        success = False
    
    return success


def test_integration_modes():
    """Test both CNN and raycast integration modes"""
    print("\n🔗 Testing Integration Modes")
    print("-" * 40)
    
    success = True
    
    # Test CNN Mode
    print("1. Testing CNN Mode Integration...")
    try:
        CONFIG['use_cnn_depth'] = True
        env = UAVEnv(render_mode=None)
        
        # Verify components are initialized
        assert env.depth_camera is not None, "Depth camera not initialized"
        assert env.depth_extractor is not None, "Depth extractor not initialized"
        assert env.depth_preprocessor is not None, "Depth preprocessor not initialized"
        
        # Test environment functionality
        obs, _ = env.reset()
        assert obs.shape[0] > 100, f"CNN observation too small: {obs.shape[0]}"  # Should be ~142
        assert np.isfinite(obs).all(), "CNN observation contains non-finite values"
        
        # Test stepping
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        
        env.close()
        print("   ✅ CNN mode integration working correctly")
        
    except Exception as e:
        print(f"   ❌ CNN mode integration failed: {e}")
        success = False
    
    # Test Raycast Mode
    print("2. Testing Raycast Mode Integration...")
    try:
        CONFIG['use_cnn_depth'] = False
        env = UAVEnv(render_mode=None)
        
        # Verify fallback components
        assert env.depth_camera is None, "Depth camera should be None in raycast mode"
        assert env.depth_extractor is None, "Depth extractor should be None in raycast mode"
        assert env.depth_preprocessor is None, "Depth preprocessor should be None in raycast mode"
        
        # Test environment functionality
        obs, _ = env.reset()
        assert obs.shape[0] < 50, f"Raycast observation too large: {obs.shape[0]}"  # Should be ~36
        assert np.isfinite(obs).all(), "Raycast observation contains non-finite values"
        
        # Test stepping
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        
        env.close()
        print("   ✅ Raycast mode integration working correctly")
        
    except Exception as e:
        print(f"   ❌ Raycast mode integration failed: {e}")
        success = False
    
    return success


def test_training_compatibility():
    """Test compatibility with training systems"""
    print("\n🎓 Testing Training Compatibility")
    print("-" * 40)
    
    success = True
    
    try:
        from ppo_agent import PPOAgent
        
        # Test with CNN mode
        CONFIG['use_cnn_depth'] = True
        env = UAVEnv(render_mode=None)
        
        # Get dimensions from environment
        obs, _ = env.reset()
        state_dim = obs.shape[0]
        action_dim = 3  # Standard UAV action dimension
        
        # Create PPO agent
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=4,
            eps_clip=0.2,
            action_std_init=0.6
        )
        
        # Test agent interaction
        obs, _ = env.reset()
        state_tensor = torch.FloatTensor(obs)
        
        # Test action selection
        with torch.no_grad():
            action, _ = ppo_agent.policy_old.act(state_tensor)
        
        assert action.shape == torch.Size([1, 3]), f"Invalid action shape: {action.shape}"
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        
        env.close()
        print("   ✅ Training compatibility verified")
        
    except Exception as e:
        print(f"   ❌ Training compatibility failed: {e}")
        success = False
    
    return success


def test_error_handling():
    """Test error handling and robustness"""
    print("\n🛡️ Testing Error Handling")
    print("-" * 40)
    
    success = True
    
    # Test 1: Invalid depth values
    print("1. Testing invalid depth value handling...")
    try:
        preprocessor = DepthPreprocessor()
        
        # Create problematic depth image
        problem_image = np.full((64, 64), np.nan)
        problem_image[20:40, 20:40] = np.inf
        problem_image[50:60, 50:60] = -100.0
        
        result = preprocessor.preprocess(problem_image)
        
        assert torch.isfinite(result).all(), "Failed to handle invalid depth values"
        print("   ✅ Invalid depth values handled correctly")
        
    except Exception as e:
        print(f"   ❌ Invalid depth value handling failed: {e}")
        success = False
    
    # Test 2: Camera failure fallback
    print("2. Testing camera failure fallback...")
    try:
        CONFIG['use_cnn_depth'] = True
        env = UAVEnv(render_mode=None)
        
        # Even with camera failures (which we expect), environment should work
        obs, _ = env.reset()
        
        # Should get valid observations despite camera warnings
        assert np.isfinite(obs).all(), "Fallback observations contain invalid values"
        
        env.close()
        print("   ✅ Camera failure fallback working correctly")
        
    except Exception as e:
        print(f"   ❌ Camera failure fallback failed: {e}")
        success = False
    
    return success


def performance_benchmark():
    """Benchmark system performance"""
    print("\n⚡ Performance Benchmark")
    print("-" * 40)
    
    try:
        # Benchmark CNN processing
        CONFIG['use_cnn_depth'] = True
        env = UAVEnv(render_mode=None)
        
        # Time environment operations
        times = []
        obs, _ = env.reset()
        
        for _ in range(10):
            start_time = time.time()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        
        print(f"   📊 CNN Mode Performance:")
        print(f"      - Average step time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"      - Steps per second: {1000/avg_time:.1f}")
        
        # Quick raycast comparison
        CONFIG['use_cnn_depth'] = False
        env = UAVEnv(render_mode=None)
        
        times = []
        obs, _ = env.reset()
        
        for _ in range(10):
            start_time = time.time()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        
        avg_time_raycast = np.mean(times) * 1000
        
        print(f"   📊 Raycast Mode Performance:")
        print(f"      - Average step time: {avg_time_raycast:.2f} ms")
        print(f"      - Performance ratio: {avg_time/avg_time_raycast:.1f}x slower")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance benchmark failed: {e}")
        return False


def generate_final_report():
    """Generate final system validation report"""
    print("\n📋 Final System Validation Report")
    print("=" * 60)
    
    report = {
        'components': test_core_components(),
        'integration': test_integration_modes(),
        'training': test_training_compatibility(),
        'error_handling': test_error_handling(),
        'performance': performance_benchmark()
    }
    
    # Summary
    passed_tests = sum(report.values())
    total_tests = len(report)
    
    print(f"\n📊 Test Results Summary:")
    print(f"   ✅ Passed: {passed_tests}/{total_tests}")
    print(f"   ❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 SYSTEM VALIDATION SUCCESSFUL!")
        print("   The CNN Depth Processing System is ready for production use.")
        print("\n✨ Key Features Validated:")
        print("   • CNN-based depth feature extraction (128D)")
        print("   • Robust preprocessing pipeline with fallbacks")
        print("   • Seamless raycast fallback system")
        print("   • Full training system compatibility")
        print("   • Comprehensive error handling")
        print("   • Performance benchmarking complete")
        
        return True
    else:
        print("\n❌ SYSTEM VALIDATION FAILED!")
        print("   Please review the failed tests above.")
        
        return False


def main():
    """Main validation function"""
    print("🚀 CNN Depth Processing System - Final Validation")
    print("=" * 60)
    print("This comprehensive validation suite tests all system components")
    print("and integration points to ensure production readiness.")
    print()
    
    success = generate_final_report()
    
    if success:
        print("\n🎯 READY FOR DEPLOYMENT!")
        return 0
    else:
        print("\n🔧 REQUIRES FIXES BEFORE DEPLOYMENT!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)