#!/usr/bin/env python3
"""
Quick test of the performance comparison system
Tests the workflow without running full trials
"""

import sys
import os
from performance_comparison import PerformanceComparison, ComparisonEnvironment

def test_performance_comparison():
    """Test the performance comparison system setup"""
    print("🧪 Testing UAV Performance Comparison System")
    print("=" * 50)
    
    try:
        # Test 1: Initialize comparison system
        print("1. Initializing comparison system...")
        comparison = PerformanceComparison()
        print("   ✅ Comparison system initialized")
        
        # Test 2: Check model availability
        print("\n2. Checking model availability...")
        print(f"   Neural model: {'✅' if comparison.neural_available else '❌'}")
        print(f"   Neurosymbolic model: {'✅' if comparison.neurosymbolic_available else '❌'}")
        
        # Test 3: Create test environment
        print("\n3. Creating test environment (Level 1)...")
        env = ComparisonEnvironment(level=1)
        print(f"   ✅ Environment created with {len(env.obstacles)} obstacle(s)")
        print(f"   Start position: {env.model is not None}")
        print(f"   MuJoCo model loaded: {env.data is not None}")
        
        # Test 4: Test manual control wrapper availability
        print("\n4. Checking manual control integration...")
        wrapper_exists = os.path.exists('uav_manual_control_wrapper.py')
        print(f"   Manual control wrapper: {'✅' if wrapper_exists else '❌'}")
        
        if wrapper_exists:
            try:
                from uav_manual_control_wrapper import ManualControlSession
                print("   ✅ Manual control wrapper imports successfully")
            except Exception as e:
                print(f"   ❌ Manual control wrapper import failed: {e}")
        
        # Test 5: Test agent runner availability
        print("\n5. Checking agent runner...")
        try:
            from uav_agent_runner import UAVAgentRunner
            print("   ✅ UAV Agent Runner available")
        except Exception as e:
            print(f"   ❌ UAV Agent Runner import failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 TEST SUMMARY")
        print("=" * 50)
        
        ready_for_comparison = (
            comparison.neural_available or comparison.neurosymbolic_available
        ) and wrapper_exists
        
        if ready_for_comparison:
            print("✅ System ready for performance comparison!")
            print("\nTo run full comparison:")
            print("   python performance_comparison.py")
            print("\nWorkflow will be:")
            print("   1. Human Expert (manual control with keyboard)")
            print("   2. Neural Only (PPO agent)")
            print("   3. Neurosymbolic (PPO + RDR agent)")
            print("   4. Level summary and progression")
        else:
            print("❌ System not fully ready. Issues found:")
            if not (comparison.neural_available or comparison.neurosymbolic_available):
                print("   - No trained models available")
            if not wrapper_exists:
                print("   - Manual control wrapper missing")
        
        return ready_for_comparison
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_comparison()
    sys.exit(0 if success else 1)