#!/usr/bin/env python3
"""
Test CNN-enabled training integration
Validates that training scripts work with the new CNN depth processing
"""

import torch
import numpy as np
import sys
import os
from uav_env import UAVEnv, CONFIG
from ppo_agent import PPOAgent

def test_cnn_training_setup():
    """Test that training setup works with CNN depth processing"""
    print("🧪 Testing CNN Training Setup")
    print("=" * 50)
    
    # Test with CNN enabled
    print("\n1. Testing with CNN Depth Processing:")
    CONFIG['use_cnn_depth'] = True
    
    try:
        # Create environment
        env = UAVEnv(curriculum_learning=False)
        print(f"   ✅ Environment created: {env.observation_space.shape}")
        
        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"   📊 State dimension: {state_dim}")
        print(f"   🎯 Action dimension: {action_dim}")
        
        # Test PPO agent creation
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=80,
            eps_clip=0.2,
            action_std_init=0.6
        )
        print(f"   ✅ PPO Agent created successfully")
        
        # Test agent network sizes
        print(f"   🧠 Actor network input: {ppo_agent.policy.actor[0].in_features}")
        print(f"   🧠 Critic network input: {ppo_agent.policy.critic[0].in_features}")
        
        # Test environment interaction
        obs, _ = env.reset()
        print(f"   ✅ Environment reset: observation shape {obs.shape}")
        
        # Test agent action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs)
            action, action_logprob = ppo_agent.policy_old.act(state_tensor)
        print(f"   ✅ Action selection: action shape {action.shape}")
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        print(f"   ✅ Environment step: reward={reward:.3f}")
        
        env.close()
        print(f"   ✅ CNN training setup test PASSED")
        
    except Exception as e:
        print(f"   ❌ CNN training setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with CNN disabled (raycast mode)
    print("\n2. Testing with Raycast Depth Processing:")
    CONFIG['use_cnn_depth'] = False
    
    try:
        # Create environment
        env = UAVEnv(curriculum_learning=False)
        print(f"   ✅ Environment created: {env.observation_space.shape}")
        
        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"   📊 State dimension: {state_dim}")
        print(f"   🎯 Action dimension: {action_dim}")
        
        # Test PPO agent creation
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=80,
            eps_clip=0.2,
            action_std_init=0.6
        )
        print(f"   ✅ PPO Agent created successfully")
        
        # Test agent network sizes
        print(f"   🧠 Actor network input: {ppo_agent.policy.actor[0].in_features}")
        print(f"   🧠 Critic network input: {ppo_agent.policy.critic[0].in_features}")
        
        # Test environment interaction
        obs, _ = env.reset()
        print(f"   ✅ Environment reset: observation shape {obs.shape}")
        
        # Test agent action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs)
            action, action_logprob = ppo_agent.policy_old.act(state_tensor)
        print(f"   ✅ Action selection: action shape {action.shape}")
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        print(f"   ✅ Environment step: reward={reward:.3f}")
        
        env.close()
        print(f"   ✅ Raycast training setup test PASSED")
        
    except Exception as e:
        print(f"   ❌ Raycast training setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_mini_training_loop():
    """Test a mini training loop to ensure everything works together"""
    print("\n🏃 Testing Mini Training Loop")
    print("=" * 50)
    
    # Enable CNN for this test
    CONFIG['use_cnn_depth'] = True
    
    try:
        # Create environment and agent
        env = UAVEnv(curriculum_learning=False)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=4,  # Small for testing
            eps_clip=0.2,
            action_std_init=0.6
        )
        
        print(f"   🎯 Training setup: {state_dim}D -> {action_dim}D")
        
        # Mini training loop (few steps)
        memory = Memory()
        total_reward = 0
        steps = 0
        max_steps = 50  # Small for testing
        
        obs, _ = env.reset()
        
        for step in range(max_steps):
            # Select action
            state_tensor = torch.FloatTensor(obs)
            action, action_logprob = ppo_agent.policy_old.act(state_tensor)
            
            # Environment step
            new_obs, reward, terminated, truncated, info = env.step(action.numpy())
            done = terminated or truncated
            
            # Store in memory
            memory.states.append(obs)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            obs = new_obs
            total_reward += reward
            steps += 1
            
            if done:
                obs, _ = env.reset()
            
            # Update every 20 steps
            if (step + 1) % 20 == 0:
                ppo_agent.update(memory)
                memory.clear_memory()
                print(f"   📈 Step {step+1}: avg_reward={total_reward/steps:.3f}")
        
        env.close()
        print(f"   ✅ Mini training loop completed successfully")
        print(f"   📊 Final stats: {steps} steps, avg_reward={total_reward/steps:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mini training loop FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_variants_with_cnn():
    """Test all PPO variants work with CNN processing"""
    print("\n🎭 Testing PPO Variants with CNN")
    print("=" * 50)
    
    CONFIG['use_cnn_depth'] = True
    
    variants = [
        ('vanilla', {'use_neurosymbolic': False, 'use_extra_rewards': False, 'ppo_type': 'vanilla', 'lambda': 0.0}),
        ('ar', {'use_neurosymbolic': False, 'use_extra_rewards': True, 'ppo_type': 'ar', 'lambda': 0.0}),
        ('ns', {'use_neurosymbolic': True, 'use_extra_rewards': False, 'ppo_type': 'ns', 'lambda': 1.0})
    ]
    
    for variant_name, ns_cfg in variants:
        print(f"\n   Testing {variant_name.upper()} PPO with CNN:")
        try:
            # Create environment with variant config
            env = UAVEnv(curriculum_learning=False, ns_cfg=ns_cfg)
            
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            print(f"     📊 Dimensions: {state_dim}D -> {action_dim}D")
            
            # Create agent
            ppo_agent = PPOAgent(state_dim, action_dim, 0.001, 0.001, 0.99, 4, 0.2, 0.6)
            
            # Test few steps
            obs, _ = env.reset()
            for i in range(5):
                state_tensor = torch.FloatTensor(obs)
                action, _ = ppo_agent.policy_old.act(state_tensor)
                obs, reward, terminated, truncated, info = env.step(action.numpy())
                done = terminated or truncated
                if done:
                    obs, _ = env.reset()
            
            env.close()
            print(f"     ✅ {variant_name.upper()} PPO variant working")
            
        except Exception as e:
            print(f"     ❌ {variant_name.upper()} PPO variant failed: {e}")
            return False
    
    print(f"   ✅ All PPO variants work with CNN processing")
    return True


class Memory:
    """Simple memory class for testing"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def main():
    """Run all training integration tests"""
    print("🚀 CNN Training Integration Tests")
    print("=" * 60)
    
    success = True
    
    # Test 1: Training setup
    if not test_cnn_training_setup():
        success = False
    
    # Test 2: Mini training loop
    if not test_mini_training_loop():
        success = False
    
    # Test 3: PPO variants
    if not test_ppo_variants_with_cnn():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All training integration tests PASSED!")
        print("\nCNN Training System Status:")
        print("  ✅ CNN depth processing integrated")
        print("  ✅ Dynamic observation space handling")
        print("  ✅ PPO agent network adaptation")
        print("  ✅ All PPO variants compatible")
        print("  ✅ Training loop functionality verified")
        print("\n📚 Ready for full training runs!")
    else:
        print("❌ Some tests FAILED. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()