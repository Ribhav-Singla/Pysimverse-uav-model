import torch
import numpy as np
from uav_env import UAVEnv, CONFIG
from ppo_agent import PPOAgent
import os

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
    solved_reward = 1000         # threshold for saving high-reward models (training continues)
    log_interval = 10          # print avg reward in the interval
    
    # Curriculum Learning Parameters
    curriculum_learning = True
    episodes_per_level_count = 500  # Episodes per curriculum level
    total_levels = 10           # Obstacle levels 1-10
    
    # Set equal episodes for each level
    episodes_per_level = [episodes_per_level_count] * total_levels
    total_episodes = episodes_per_level_count * total_levels  # 500 * 10 = 5000 episodes
    
    max_episodes = total_episodes
    max_timesteps = 2500         # Match env max_steps

    update_timestep = 512       # FURTHER REDUCED for very frequent updates to break stagnation
    action_std = 0.6            # SIGNIFICANTLY INCREASED exploration to break oscillation patterns
    K_epochs = 6                # FURTHER REDUCED for more responsive policy changes
    eps_clip = 0.3              # INCREASED for larger policy updates - escape local optima
    gamma = 0.98                # FURTHER REDUCED for immediate reward focus

    lr_actor = 0.0005           # SIGNIFICANTLY INCREASED for rapid policy adaptation
    lr_critic = 0.0015          # SIGNIFICANTLY INCREASED for better value updates

    # Neurosymbolic parameters (λ = 0 ensures original behavior is preserved)
    lambda_param = 0.0          # Balance between neural (0) and symbolic (1) - KEEP AT 0 FOR NOW
    lambda_decay = False        # Whether to gradually decay lambda over training - KEEP FALSE FOR NOW
    lambda_final = 0.0          # Final lambda value if decaying (not used when lambda_decay=False)
    
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
        # env.seed(random_seed)  # Not needed for newer gym versions
        np.random.seed(random_seed)
        print("--------------------------------------------------------------------------------------------")

    memory = Memory()
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, lambda_param)
    
    # Example of adding a symbolic rule (won't affect behavior when lambda=0)
    # Uncomment this when you're ready to use rules
    # ppo_agent.add_obstacle_avoidance_rule(lidar_threshold=0.6)
    
    print("--------------------------------------------------------------------------------------------")
    print("🎓 CURRICULUM LEARNING ENABLED")
    print(f"   - Training across {total_levels} difficulty levels (1-10 obstacles)")
    print(f"   - Total episodes: {total_episodes}")
    print("   - Episodes per level (equal distribution):")
    for i, episodes in enumerate(episodes_per_level):
        print(f"     Level {i+1} ({i+1} obstacles): {episodes} episodes")
    print(f"   - 20 different maps per obstacle level")
    print("📊 LOGGING ENABLED")
    print("   - Episode data: 'curriculum_learning_log.csv'")
    print("   - Obstacle detections: 'obstacle_detection_log.csv'")
    print("   - Detection threshold: 1.5m")
    print("   - CSV files will be refreshed for each training session")
    print("🧠 NEUROSYMBOLIC CONFIGURATION")
    print(f"   - Lambda: {lambda_param:.2f} (0 = pure neural, 1 = pure symbolic)")
    print(f"   - Lambda decay: {'Enabled' if lambda_decay else 'Disabled'}")
    if lambda_decay:
        print(f"   - Lambda final value: {lambda_final:.2f}")
    print("   - Symbolic rules: None (placeholder only)")
    print("--------------------------------------------------------------------------------------------")
    
    # Load pre-trained weights (using relative path)
    pretrained_weights = os.path.join(checkpoint_path, "PPO_UAV_Weights.pth")
    if os.path.exists(pretrained_weights):
        print("Loading pre-trained weights from:", pretrained_weights)
        try:
            ppo_agent.load(pretrained_weights)
            print("Pre-trained weights loaded successfully!")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: Could not load weights due to architecture mismatch: {e}")
            print("This is expected when changing action space dimensions (4D -> 3D)")
            print("Starting training with a new model")
            # Remove old incompatible weights
            os.remove(pretrained_weights)
            print("Removed incompatible old weights file")
    else:
        print(f"No existing weights found at {pretrained_weights}")
        print("Training will start with a new model")

    # logging variables
    time_step = 0
    best_reward = -float('inf')
    
    # Adaptive action standard deviation decay - MAINTAIN HIGH EXPLORATION
    initial_action_std = action_std
    min_action_std = 0.3        # HIGH minimum to prevent getting stuck in local optima
    action_std_decay_rate = 0.02 # SLOWER decay rate
    action_std_decay_freq = 500  # LESS frequent decay - maintain exploration longer
    
    # Initialize curriculum tracking
    current_curriculum_level = 1
    current_level_index = 0  # Index for episodes_per_level list
    episodes_in_current_level = 0
    level_rewards = []  # Track rewards for current level
    
    # training loop (start from episode 1)
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
        # Set current episode number for CSV logging
        env.current_episode = i_episode
        episode_reward = 0
        episodes_in_current_level += 1
        
        # Decay action std for more precise actions as training progresses
        if i_episode % action_std_decay_freq == 0:
            new_action_std = max(min_action_std, initial_action_std - 
                               (initial_action_std - min_action_std) * 
                               (i_episode / max_episodes))
            ppo_agent.set_action_std(new_action_std)
            print(f"Adjusted action std to {new_action_std:.4f}")
            
            # Update lambda if lambda decay is enabled
            if lambda_decay:
                # Gradually increase lambda's importance (more symbolic guidance)
                # from initial lambda to final lambda
                new_lambda = lambda_param + (lambda_final - lambda_param) * (i_episode / max_episodes)
                ppo_agent.set_lambda(new_lambda)
                print(f"Adjusted lambda to {new_lambda:.4f}")
            
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action, log_prob = ppo_agent.select_action(state)
            
            # Save state, action, and log probability
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            
            state, reward, done, truncated, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo_agent.update(memory)
                memory.clear_memory()
                time_step = 0
            episode_reward += reward
            if render:
                env.render()
            if done or truncated:
                # Get detailed termination information
                termination_info = getattr(env, 'last_termination_info', {})
                curriculum_info = env.get_curriculum_info() if curriculum_learning else None
                
                print(f"\n--- Episode {i_episode} Terminated ---")
                if curriculum_info:
                    print(f"Curriculum Level: {curriculum_info['current_level']} obstacles")
                    print(f"Level Progress: {episodes_in_current_level}/{episodes_per_level}")
                print(f"Start Position: [{CONFIG['start_pos'][0]:.1f}, {CONFIG['start_pos'][1]:.1f}, {CONFIG['start_pos'][2]:.1f}]")
                print(f"Goal Position: [{CONFIG['goal_pos'][0]:.1f}, {CONFIG['goal_pos'][1]:.1f}, {CONFIG['goal_pos'][2]:.1f}]")
                print(f"Reason: {termination_info.get('termination_reason', 'unknown')}")
                print(f"Final Position: {termination_info.get('final_position', 'unknown')}")
                print(f"Goal Distance: {termination_info.get('goal_distance', 0):.4f}")
                print(f"Episode Length: {termination_info.get('episode_length', t+1)} steps")
                print(f"Final Velocity: {termination_info.get('final_velocity', 0):.2f}")
                print(f"Episode Reward: {episode_reward:.1f}")
                
                if termination_info.get('collision_detected', False):
                    print("❌ Collision with obstacle detected!")
                elif termination_info.get('out_of_bounds', False):
                    print("🚫 UAV went out of bounds!")
                elif termination_info.get('termination_reason') == 'goal_reached':
                    print("🎯 Goal reached successfully!")
                elif termination_info.get('termination_reason') == 'max_steps_reached':
                    print("⏰ Maximum steps reached!")
                print("-" * 40)
                break
        
        episode_length = t + 1
        
        # Track reward and log curriculum episode data
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

        # Check if reward threshold is reached (but don't stop training)
        if episode_reward > solved_reward:
            print("########## Reward Threshold Reached! ##########")
            print(f"Goal reached with reward: {episode_reward:.1f}")
            print(f"Episode length: {episode_length} steps")
            # Save a special checkpoint for this high-performing model
            high_reward_path = os.path.join(checkpoint_path, "PPO_UAV_HighReward.pth")
            ppo_agent.save(high_reward_path)
            print(f"High reward model saved to {high_reward_path}")
            # Continue training to further improve performance

        # Print current episode stats (simplified for non-termination cases)
        if not (done or truncated):
            if curriculum_learning:
                print('Episode {} (Level {}) \t Length: {} \t Reward: {:.1f} \t Start: [{:.1f},{:.1f}] \t Goal: [{:.1f},{:.1f}]'.format(
                    i_episode, current_curriculum_level, episode_length, episode_reward,
                    CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['goal_pos'][0], CONFIG['goal_pos'][1]))
            else:
                print('Episode {} \t Length: {} \t Reward: {:.1f} \t Start: [{:.1f},{:.1f}] \t Goal: [{:.1f},{:.1f}]'.format(
                    i_episode, episode_length, episode_reward,
                    CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['goal_pos'][0], CONFIG['goal_pos'][1]))
        
        # Save model periodically and track best performance
        if i_episode % log_interval == 0:
            # Always update the same weights file with the latest model
            ppo_agent.save(os.path.join(checkpoint_path, "PPO_UAV_Weights.pth"))
            
            # Print training progress summary
            print(f"\n=== Training Progress Summary (Episode {i_episode}) ===")
            if curriculum_learning:
                print(f"🎓 Curriculum Level: {current_curriculum_level}/10 ({current_curriculum_level} obstacles)")
                print(f"📊 Level Progress: {episodes_in_current_level}/{episodes_per_level[current_level_index]} episodes")
                if level_rewards:
                    recent_avg = sum(level_rewards[-min(50, len(level_rewards)):]) / min(50, len(level_rewards))
                    print(f"📈 Recent Average Reward (Level {current_curriculum_level}): {recent_avg:.2f}")
                    
            current_action_std = ppo_agent.policy.action_var[0].sqrt().item()
            print(f"Current Action Std: {current_action_std:.4f}")
            print(f"Latest Episode Reward: {episode_reward:.1f}")
            print(f"Latest Episode Length: {episode_length} steps")
            
            # Print neurosymbolic info
            ns_info = ppo_agent.get_neurosymbolic_info()
            print(f"Neurosymbolic λ: {ns_info['lambda']:.4f} ({len(ns_info['rules'])} rules)")
            
            # Only print "New best" message if there's improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"✓ New best model saved with reward: {episode_reward:.1f}")
            else:
                print(f"📝 Model updated (best: {best_reward:.1f})")
            print("=" * 50)

if __name__ == '__main__':
    main()