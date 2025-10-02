"""
Neurosymbolic Training Script for UAV Navigation
Integrates RDR rules with PPO training for enhanced learning performance.
"""

import torch
import numpy as np
from uav_env import UAVEnv, CONFIG
from neurosymbolic_ppo_agent import PPOAgent
import os
import threading
import time

class Memory:
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
    ############## Hyperparameters ##############
    env_name = "UAVEnv"
    render = False
    log_interval = 10          # print avg reward in the interval
    
    # Neurosymbolic Parameters
    enable_neurosymbolic = True
    integration_weight = 0.25   # How much to weight symbolic advice (0.0 = pure RL, 1.0 = pure symbolic)
    
    # Curriculum Learning Parameters
    curriculum_learning = True
    episodes_per_level_count = 300  # Episodes per curriculum level (equal for all levels)
    total_levels = 10           # Obstacle levels 1-10
    
    # Set equal episodes for each level
    episodes_per_level = [episodes_per_level_count] * total_levels
    total_episodes = episodes_per_level_count * total_levels  # 300 * 10 = 3000 episodes
    
    max_episodes = total_episodes
    max_timesteps = 50000        # max timesteps in one episode

    update_timestep = 1024      # update policy every 1024 timesteps
    action_std = 1.0            # Start with 100% exploration
    K_epochs = 12               # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO
    gamma = 0.999               # discount factor

    lr_actor = 0.0001           # learning rate for actor
    lr_critic = 0.0004          # learning rate for critic

    random_seed = 0
    #############################################

    # creating environment with curriculum learning
    env = UAVEnv(curriculum_learning=curriculum_learning)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # checkpoint path
    checkpoint_path = "PPO_preTrained/{}/".format(env_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        print("--------------------------------------------------------------------------------------------")

    # Action standard deviation decay parameters
    initial_action_std = action_std  # 1.0 (100% exploration)
    min_action_std = 0.1             # 10% exploration (refined)
    action_std_decay_freq = 100      # Decay every 100 episodes
    
    # Success rate tracking
    success_window = []
    milestone_success_threshold = 0.85  # Save milestone when achieving 85% success

    memory = Memory()
    
    # Initialize PPO agent with neurosymbolic capabilities
    ppo_agent = PPOAgent(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std,
        enable_neurosymbolic=enable_neurosymbolic
    )
    
    if enable_neurosymbolic:
        # Ensure the knowledge base file exists
        ppo_agent.rdr_kb.save_knowledge_base()  # Ensure the file exists 
    
    print("--------------------------------------------------------------------------------------------")
    print("🚀 NEUROSYMBOLIC PPO TRAINING CONFIGURATION")
    print("--------------------------------------------------------------------------------------------")
    if enable_neurosymbolic:
        print("🧠 NEUROSYMBOLIC INTEGRATION:")
        print(f"   - Integration Weight: {integration_weight:.2f} (symbolic influence)")
        print(f"   - RDR Knowledge Base: {len(ppo_agent.rdr_kb.rules)} initial rules")
        print()
    
    print("📊 OBSERVATION SPACE:")
    print(f"   - State Dimension: {state_dim}D")
    print(f"   - Raw LIDAR: 16 rays (normalized to [0,1])")
    print(f"   - LIDAR Features: 11 engineered features")
    if enable_neurosymbolic:
        print(f"   - Symbolic Features: 10 rule activation features")
    print()
    
    print("🧠 NETWORK ARCHITECTURE:")
    print(f"   - Hidden Dimensions: 256")
    if enable_neurosymbolic:
        print(f"   - Enhanced Input: {state_dim + 10}D (with symbolic features)")
    print()
    
    print("🎓 CURRICULUM LEARNING:")
    print(f"   - Total Levels: {total_levels} (1-10 obstacles)")
    print(f"   - Episodes per Level: {episodes_per_level_count}")
    print(f"   - Total Episodes: {max_episodes}")
    print(f"   - 50 unique maps per level (500 total)")
    print()
    
    print("💰 REWARD STRUCTURE:")
    print(f"   - Goal Reached: +1000")
    print(f"   - Collision/OOB: -100")
    print(f"   - Step Penalty: -0.01")
    print("--------------------------------------------------------------------------------------------")

    # Load existing neurosymbolic model if available
    model_path = os.path.join(checkpoint_path, "PPO_UAV_Weights_neurosymbolic.pth")
    if os.path.exists(model_path):
        if ppo_agent.load(model_path):
            print(f"✅ Loaded existing neurosymbolic model from {model_path}")
        else:
            print(f"⚠️ Failed to load neurosymbolic model, starting fresh training")
    else:
        print(f"🆕 No existing neurosymbolic model found, starting fresh training")

    # Training variables
    time_step = 0
    i_episode = 0
    running_reward = 0
    avg_length = 0
    current_curriculum_level = 1
    current_level_index = 0
    episodes_in_current_level = 0
    level_rewards = []

    # Neurosymbolic tracking
    total_symbolic_advice = 0
    successful_symbolic_advice = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        # Check if we need to advance curriculum level
        if curriculum_learning and episodes_in_current_level >= episodes_per_level[current_level_index]:
            if current_curriculum_level < total_levels:
                # Calculate average reward for completed level
                if level_rewards:
                    avg_reward = sum(level_rewards) / len(level_rewards)
                    print(f"\n🎓 CURRICULUM LEVEL {current_curriculum_level} COMPLETED")
                    print(f"   - Episodes: {episodes_per_level[current_level_index]}")
                    print(f"   - Average Reward: {avg_reward:.2f}")
                    print(f"   - Obstacles: {current_curriculum_level}")
                
                # Advance to next level
                current_curriculum_level += 1
                current_level_index += 1
                episodes_in_current_level = 0
                level_rewards = []
                env.set_curriculum_level(current_curriculum_level)
                
                print(f"🆙 ADVANCING TO CURRICULUM LEVEL {current_curriculum_level}")
                print(f"   - New obstacle count: {current_curriculum_level}")
                print(f"   - Target episodes: {episodes_per_level[current_level_index]}")
                print("-" * 60)
        
        state, _ = env.reset()
        # Set current episode number for tracking
        env.current_episode = i_episode
        episode_reward = 0
        episodes_in_current_level += 1
        
        # Adaptive action std decay
        if i_episode % action_std_decay_freq == 0:
            progress = i_episode / max_episodes
            new_action_std = max(min_action_std, initial_action_std * (1 - progress))
            ppo_agent.set_action_std(new_action_std)
            print(f"📊 Episode {i_episode}: Action std adjusted to {new_action_std:.4f} (Exploration: {new_action_std*100:.1f}%)")
            
            # Log learning rates
            current_lrs = ppo_agent.get_current_lr()
            print(f"📊 Current Learning Rates - Actor: {current_lrs['actor_lr']:.6f}, Critic: {current_lrs['critic_lr']:.6f}")
            
            # Log neurosymbolic stats
            if enable_neurosymbolic:
                ns_stats = ppo_agent.get_symbolic_stats()
                print(f"🧠 Symbolic Stats - Usage Rate: {ns_stats['advice_usage_rate']:.2f}, Success Rate: {ns_stats['advice_success_rate']:.2f}")
                print(f"🧠 Active Rules: {ns_stats['total_rules']}, Total Advice: {ns_stats['total_advice_given']}")
            
        for t in range(max_timesteps):
            time_step += 1
            
            # Select action (with neurosymbolic integration if enabled)
            action, log_prob = ppo_agent.select_action(state)
            
            # Save state, action, and log probability
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.FloatTensor(action))
            memory.logprobs.append(torch.FloatTensor(log_prob))
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Save reward and terminal flag
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # Update policy
            if time_step % update_timestep == 0:
                ppo_agent.update(memory)
                memory.clear_memory()
                time_step = 0
                
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
                
            if done or truncated:
                # Update neurosymbolic performance tracking
                if enable_neurosymbolic:
                    termination_info = getattr(env, 'last_termination_info', {})
                    success = termination_info.get('termination_reason') == 'goal_reached'
                    ppo_agent.update_symbolic_performance(state, success)
                
                # Get detailed termination information
                termination_info = getattr(env, 'last_termination_info', {})
                curriculum_info = env.get_curriculum_info() if curriculum_learning else None
                
                print(f"\n--- Episode {i_episode} Terminated ---")
                if curriculum_info:
                    print(f"Curriculum Level: {curriculum_info['current_level']} obstacles")
                    print(f"Level Progress: {episodes_in_current_level}/{episodes_per_level[current_level_index]}")
                print(f"Reason: {termination_info.get('termination_reason', 'unknown')}")
                print(f"Episode Reward: {episode_reward:.1f}")
                print(f"Episode Length: {t+1} steps")
                
                if termination_info.get('termination_reason') == 'goal_reached':
                    print("🎯 Goal reached successfully!")
                    success_window.append(1)
                else:
                    success_window.append(0)
                    
                if len(success_window) > 100:
                    success_window.pop(0)
                    
                print("-" * 40)
                break
        
        episode_length = t + 1
        
        # Track reward
        if curriculum_learning:
            level_rewards.append(episode_reward)
            
            # Log curriculum episode data
            termination_info = getattr(env, 'last_termination_info', {})
            termination_info['episode_length'] = episode_length
            env.log_curriculum_episode(
                episode_num=i_episode,
                episode_in_level=episodes_in_current_level,
                episode_reward=episode_reward,
                episode_length=episode_length,
                termination_info=termination_info
            )

        # Print episode stats
        if not (done or truncated):
            if curriculum_learning:
                print('Episode {} (Level {}) \t Length: {} \t Reward: {:.1f}'.format(
                    i_episode, current_curriculum_level, episode_length, episode_reward))
            else:
                print('Episode {} \t Length: {} \t Reward: {:.1f}'.format(
                    i_episode, episode_length, episode_reward))
        
        # Periodic model saving and progress tracking
        if i_episode % log_interval == 0:
            # Save current neurosymbolic model
            ppo_agent.save(os.path.join(checkpoint_path, "PPO_UAV_Weights_neurosymbolic.pth"))
            
            print(f"\n=== Training Progress Summary (Episode {i_episode}) ===")
            if curriculum_learning:
                print(f"🎓 Curriculum Level: {current_curriculum_level}/10 ({current_curriculum_level} obstacles)")
                print(f"📊 Level Progress: {episodes_in_current_level}/{episodes_per_level[current_level_index]} episodes")
                if level_rewards:
                    recent_avg = sum(level_rewards[-min(50, len(level_rewards)):]) / min(50, len(level_rewards))
                    print(f"📈 Recent Level Average: {recent_avg:.2f}")
            
            # Success rate tracking
            if len(success_window) >= 10:
                success_rate = sum(success_window) / len(success_window)
                print(f"🎯 Success Rate (last {len(success_window)}): {success_rate*100:.1f}%")
                
                # Save milestone if high success rate
                if success_rate >= milestone_success_threshold:
                    milestone_path = os.path.join(checkpoint_path, f"PPO_UAV_Neurosymbolic_Milestone_{i_episode}.pth")
                    ppo_agent.save(milestone_path)
                    print(f"🏆 Neurosymbolic milestone saved! Success rate: {success_rate*100:.1f}%")
            
            # Neurosymbolic stats
            if enable_neurosymbolic:
                ns_stats = ppo_agent.get_symbolic_stats()
                print(f"🧠 Neurosymbolic Performance:")
                print(f"   - Rules in KB: {ns_stats['total_rules']}")
                print(f"   - Advice Usage Rate: {ns_stats['advice_usage_rate']:.2f}")
                print(f"   - Advice Success Rate: {ns_stats['advice_success_rate']:.2f}")
                print(f"   - Total Advice Given: {ns_stats['total_advice_given']}")
            
            print("=" * 60)

    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETED!")
    print(f"📊 Total Episodes: {max_episodes}")
    print(f"🎓 Curriculum Levels: {total_levels}")
    
    if enable_neurosymbolic:
        final_stats = ppo_agent.get_symbolic_stats()
        print(f"🧠 Final Neurosymbolic Stats:")
        print(f"   - Rules Created: {final_stats['total_rules']}")
        print(f"   - Advice Usage Rate: {final_stats['advice_usage_rate']:.2f}")
        print(f"   - Advice Success Rate: {final_stats['advice_success_rate']:.2f}")
    
    # Save final neurosymbolic model
    final_model_path = os.path.join(checkpoint_path, "PPO_UAV_Weights_neurosymbolic.pth")
    ppo_agent.save(final_model_path)
    print(f"✅ Final neurosymbolic model saved to {final_model_path}")
    
    if len(success_window) > 0:
        final_success_rate = sum(success_window) / len(success_window)
        print(f"🎯 Final Success Rate: {final_success_rate*100:.1f}%")
    
    print(f"{'='*80}")

if __name__ == '__main__':
    main()