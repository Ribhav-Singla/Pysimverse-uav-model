import torch
import numpy as np
from uav_env import UAVEnv, CONFIG
from ppo_agent import PPOAgent
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train UAV with RDR system')
    parser.add_argument('--lambda', dest='ns_lambda', type=float, default=1.0,
                        help='Neurosymbolic lambda value (0=pure RL, 1=RDR when available) (default: 1.0)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes per curriculum level (default: 50)')
    args = parser.parse_args()
    
    ############## Hyperparameters ##############
    env_name = "UAVEnv"
    render = False
    solved_reward = 1000         # threshold to save best model (but continue training)
    log_interval = 50          # print avg reward in the interval
    
    # Curriculum Learning Parameters
    curriculum_learning = True
    episodes_per_level_count = args.episodes  # Episodes per curriculum level (from command line)
    total_levels = 10           # Obstacle levels 1-10
    
    # Set equal episodes for each level
    episodes_per_level = [episodes_per_level_count] * total_levels
    total_episodes = episodes_per_level_count * total_levels  # 300 * 10 = 3000 episodes
    
    max_episodes = total_episodes
    max_timesteps = 50000        # max timesteps in one episode

    update_timestep = 1024      # OPTIMIZED: update policy every 1024 timesteps (was 2048)
    action_std = 1.0            # OPTIMIZED: Start with 100% exploration (was 0.3)
    K_epochs = 11               # update policy for K epochs (reduced for faster updates)
    eps_clip = 0.1              # clip parameter for PPO (reduced for finer control)
    gamma = 0.999               # increased for longer-term planning

    lr_actor = 0.00005          # learning rate for actor (reduced for stability)
    lr_critic = 0.0002          # learning rate for critic (reduced for stability)

    random_seed = 0
    #############################################

    # Neurosymbolic config from command line arguments
    ns_lambda = args.ns_lambda  # Use command line argument
    use_neurosymbolic = (ns_lambda > 0.0)  # Enable neurosymbolic if lambda > 0
    ns_cfg = {
        'use_neurosymbolic': use_neurosymbolic,
        'lambda': ns_lambda,
        'warmup_steps': 100,
        'high_speed': 0.9,
        'blocked_strength': 0.1,
        # Robust LOS / cooldown / distance-aware defaults
        'los_angle_margin_deg': 5.0,
        'los_confirm_steps': 3,
        'near_miss_threshold_m': 0.5,
        'near_miss_cooldown_steps': 10,
        'distance_far_m': 3.0,
        'distance_min_scale': 0.4
    }

    # creating environment with curriculum learning
    env = UAVEnv(curriculum_learning=curriculum_learning, ns_cfg=ns_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # checkpoint path with lambda value
    checkpoint_path = "PPO_preTrained/{}/".format(env_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Create weights filename with lambda value
    lambda_str = f"lambda_{ns_lambda:.1f}".replace(".", "_")  # Convert 1.0 to "lambda_1_0"
    weights_filename = f"PPO_UAV_Weights_{lambda_str}.pth"

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)  # Not needed for newer gym versions
        np.random.seed(random_seed)
        print("--------------------------------------------------------------------------------------------")

    # OPTIMIZED: Adaptive action standard deviation decay (epsilon decay)
    initial_action_std = action_std  # 1.0 (100% exploration)
    min_action_std = 0.1             # 10% exploration (refined)
    action_std_decay_freq = 100      # Decay every 100 episodes (more gradual)
    
    # ADDED: Success rate tracking for early stopping
    success_window = []
    early_stop_threshold = 0.80  # Stop if 80% success rate

    memory = Memory()
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    # === Metrics tracking ===
    episode_indices = []
    episode_rewards_list = []
    episode_success_list = []  # 1 if goal reached this episode else 0

    def save_training_plots(lambda_val):
        if len(episode_indices) == 0:
            return
        
        # Calculate exponential moving averages
        alpha = 0.05  # Smoothing factor (0.05 = more smoothing, 0.3 = less smoothing)
        rewards_np = np.array(episode_rewards_list, dtype=float)
        success_np = np.array(episode_success_list, dtype=float)
        
        # Initialize EMA arrays
        ema_rewards = np.zeros_like(rewards_np)
        ema_success = np.zeros_like(success_np)
        
        # Calculate EMA
        ema_rewards[0] = rewards_np[0]
        ema_success[0] = success_np[0]
        for i in range(1, len(rewards_np)):
            ema_rewards[i] = alpha * rewards_np[i] + (1 - alpha) * ema_rewards[i-1]
            ema_success[i] = alpha * success_np[i] + (1 - alpha) * ema_success[i-1]
        
        # Use full length arrays for plotting
        roll_rewards = ema_rewards
        roll_success = ema_success
        x_roll = episode_indices

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.scatter(episode_indices, rewards_np, color='lightgray', alpha=0.6, s=8, label='Reward')
        plt.plot(x_roll, roll_rewards, color='blue', linewidth=2, label='Reward EMA')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward')
        
        # Fixed y-axis range for consistent reward visualization
        plt.ylim(-1000, 200)
        
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.scatter(episode_indices, success_np, color='lightgray', alpha=0.6, s=8, label='Success (0/1)')
        plt.plot(x_roll, roll_success, color='green', linewidth=2, label='Success Rate EMA')
        plt.xlabel('Episode')
        plt.ylabel('Success')
        plt.title('Goal Reached')
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Include lambda value in plot filename
        plot_filename = f'training_metrics_lambda_{lambda_val:.1f}.png'.replace('.', '_')
        plt.savefig(plot_filename, dpi=150)
        plt.close()

    print("--------------------------------------------------------------------------------------------")
    print("🚀 OPTIMIZED PPO TRAINING CONFIGURATION")
    print("--------------------------------------------------------------------------------------------")
    print("🧭 NEUROSYMBOLIC CONFIGURATION:")
    print(f"   - Lambda (λ): {ns_lambda} {'(RDR when available)' if ns_lambda >= 1.0 else '(Pure RL)' if ns_lambda == 0.0 else '(Hybrid mode)'}")
    print(f"   - Weights filename: {weights_filename}")
    print(f"   - Episodes per level: {episodes_per_level_count}")
    print(f"   - RDR system: {'ENABLED' if use_neurosymbolic else 'DISABLED'}")
    print()
    print("📊 OBSERVATION SPACE UPGRADE:")
    print(f"   - State Dimension: {state_dim}D (was 25D)")
    print(f"   - Raw LIDAR: 16 rays (normalized to [0,1])")
    print(f"   - LIDAR Features: 11 engineered features")
    print(f"     * Min/Mean distance, Obstacle direction, Danger level")
    print(f"     * Directional clearance (4 sectors), Goal alignment")
    print()
    print("🧠 NETWORK ARCHITECTURE UPGRADE:")
    print(f"   - Hidden Dimensions: 256 (was 128)")
    print(f"   - Actor Parameters: ~197K (was ~52K)")
    print(f"   - Critic Parameters: ~197K (was ~52K)")
    print()
    print("💰 SIMPLIFIED REWARD STRUCTURE:")
    print(f"   - Goal Reached: +1000 (was +100)")
    print(f"   - Collision/OOB: -100 (unchanged)")
    print(f"   - Progress Reward: 10.0 × progress (simplified)")
    print(f"   - LIDAR Proximity Penalties: -0.5 to -5.0")
    print(f"   - Safe Navigation Bonus: +0.5")
    print()
    print("🎲 ADAPTIVE EXPLORATION (ACTION STD):")
    print(f"   - Initial: {initial_action_std} (100% exploration)")
    print(f"   - Final: {min_action_std} (10% exploration)")
    print(f"   - Decay: Linear every {action_std_decay_freq} episodes")
    print()
    print("📉 ADAPTIVE LEARNING RATE:")
    print(f"   - Initial Actor LR: {lr_actor}")
    print(f"   - Initial Critic LR: {lr_critic}")
    print(f"   - Decay: Exponential (gamma=0.9997 per update)")
    print()
    print("🛡️  COLLISION CURRICULUM:")
    print(f"   - Episodes 0-500: 0.15m (lenient)")
    print(f"   - Episodes 500-1500: 0.13m")
    print(f"   - Episodes 1500-3000: 0.11m")
    print(f"   - Episodes 3000+: 0.10m (strict)")
    print()
    print("🚀 VELOCITY CURRICULUM (NEW!):")
    print(f"   - Episodes 0-500: 0.5 m/s max (SLOW - exploration)")
    print(f"   - Episodes 500-1000: 0.8 m/s max")
    print(f"   - Episodes 1000-2000: 1.2 m/s max")
    print(f"   - Episodes 2000-3000: 1.5 m/s max")
    print(f"   - Episodes 3000+: 2.0 m/s max (FULL SPEED)")
    print(f"   - Gradual acceleration: 30% per step")
    print()
    print("⚡ TRAINING EFFICIENCY:")
    print(f"   - Update Frequency: {update_timestep} timesteps (was 2048)")
    print(f"   - Early Stopping: {early_stop_threshold*100:.0f}% success rate over 100 episodes")
    print()
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
    print("--------------------------------------------------------------------------------------------")
    
    # Load pre-trained weights (using lambda-specific filename)
    pretrained_weights = os.path.join(checkpoint_path, weights_filename)
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
        
        # Use command line lambda value (no per-episode changes)
        # ns_lambda is already set from command line arguments
        use_neurosymbolic = (ns_lambda > 0.0)

        state, _ = env.reset()
        # Set current episode number for CSV logging and adaptive thresholds
        env.current_episode = i_episode
        episode_reward = 0
        episodes_in_current_level += 1
        
        # OPTIMIZED: Adaptive action std decay (linear decay from 1.0 to 0.1)
        if i_episode % action_std_decay_freq == 0:
            # Linear decay: std = max(min_std, initial_std * (1 - progress))
            progress = i_episode / max_episodes
            new_action_std = max(min_action_std, initial_action_std * (1 - progress))
            ppo_agent.set_action_std(new_action_std)
            print(f"📊 Episode {i_episode}: Action std adjusted to {new_action_std:.4f} (Exploration: {new_action_std*100:.1f}%)")
            
            # Log current learning rates
            current_lrs = ppo_agent.get_current_lr()
            print(f"📊 Current Learning Rates - Actor: {current_lrs['actor_lr']:.6f}, Critic: {current_lrs['critic_lr']:.6f}")
            
            # Log collision threshold
            collision_thresh = env._get_collision_threshold()
            print(f"📊 Current Collision Threshold: {collision_thresh:.3f}m")
            
            # Velocity constraints removed - full [-1, 1] range available
            print(f"� Velocity Range: [-1.0, 1.0] (no constraints)")
            
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old to get PPO proposal
            ppo_action, ppo_log_prob = ppo_agent.select_action(state)
            # Ensure PPO action has consistent shape (1, action_dim)
            ppo_action_np = np.array(ppo_action, dtype=np.float32)
            if ppo_action_np.ndim == 1:
                ppo_action_np = np.expand_dims(ppo_action_np, axis=0)

            # Binary lambda logic: lambda=1 -> RDR (if specific rule), lambda=0 -> RL
            final_action = ppo_action_np
            action_source = "PPO"
            
            if use_neurosymbolic and ns_lambda >= 1.0:
                # Check if RDR system has a specific (non-default) rule available
                if env.has_specific_rdr_rule():
                    # Specific rule available - get and use RDR action
                    sym = np.array(env.symbolic_action(), dtype=np.float32)
                    if sym.ndim == 1:
                        sym = np.expand_dims(sym, axis=0)
                    final_action = sym
                    action_source = "RDR"
                    
                    if (t % 100) == 0:
                        rule_id = env.current_rule.rule_id if env.current_rule else "Unknown"
                        print(f"[RDR] Episode {i_episode} Step {t}: Using RDR rule {rule_id}")
                else:
                    # No specific rule available - fall back to RL agent
                    final_action = ppo_action_np  # Use PPO action when no RDR rule is available
                    action_source = "PPO_FALLBACK"
                    if (t % 100) == 0:
                        print(f"[FALLBACK] Episode {i_episode} Step {t}: No specific RDR rule available, using PPO")
            elif ns_lambda <= 0.0:
                # Pure RL control - use PPO action (already set)
                if (t % 100) == 0:
                    print(f"[PPO] Episode {i_episode} Step {t}: Using PPO action (lambda=0)")
            # Note: No blending case since lambda is binary (0 or 1)

            # Compute log_prob of the executed action under old policy (for PPO update)
            try:
                state_tensor = torch.FloatTensor(state)
                action_mean = ppo_agent.policy_old.actor(state_tensor)
                cov_mat = torch.diag(ppo_agent.policy_old.action_var).unsqueeze(dim=0)
                from torch.distributions import MultivariateNormal
                dist = MultivariateNormal(action_mean, cov_mat)
                action_tensor = torch.FloatTensor(final_action)
                exec_log_prob = dist.log_prob(action_tensor).detach().numpy().astype(np.float32)
            except Exception:
                # Fallback to PPO-sampled log prob (should rarely happen)
                exec_log_prob = np.array(ppo_log_prob, dtype=np.float32)

            # Save state, executed action, and its log probability (homogeneous shapes)
            memory.states.append(np.array(state, dtype=np.float32))
            memory.actions.append(final_action)
            memory.logprobs.append(exec_log_prob)
            
            # Env expects flat action; it will flatten anyway, but ensure (action_dim,)
            state, reward, done, truncated, _ = env.step(final_action.flatten())

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                # Check for NaN weights before update
                if ppo_agent.policy.check_and_fix_network_weights():
                    print("🔧 Network weights were reset due to NaN detection.")
                
                ppo_agent.update(memory)
                
                # Check for NaN weights after update
                if ppo_agent.policy.check_and_fix_network_weights():
                    print("🔧 Network weights were reset after PPO update.")
                
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
                print(f"Goal Distance: {termination_info.get('goal_distance', 0):.2f}")
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
                
                # ADDED: Track success rate for early stopping
                if termination_info.get('termination_reason') == 'goal_reached':
                    success_window.append(1)
                else:
                    success_window.append(0)
                
                if len(success_window) > 100:
                    success_window.pop(0)
                
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

        # === Metrics tracking ===
        episode_indices.append(i_episode)
        episode_rewards_list.append(episode_reward)
        term_reason = termination_info.get('termination_reason', '') if 'termination_info' in locals() else getattr(env, 'last_termination_info', {}).get('termination_reason', '')
        success_flag = 1 if term_reason in ['goal_reached', 'goal_reached_and_stabilized'] else 0
        episode_success_list.append(success_flag)

        # Save model after every episode
        ppo_agent.save(os.path.join(checkpoint_path, weights_filename))
        print(f"💾 Model saved after episode {i_episode} (Reward: {episode_reward:.1f})")

        # Track goal achievement (but model already saved per episode)
        if episode_reward > solved_reward:
            print("########## Goal Reached! ##########")
            print(f"Excellent performance with reward: {episode_reward:.1f}")
            print(f"Episode length: {episode_length} steps")
            print("🔄 Continuing training for more robustness...")
            print("-" * 50)
        
        # Track success rate for monitoring (but model already saved per episode)
        if len(success_window) >= 100:
            success_rate = sum(success_window) / len(success_window)
            if success_rate >= early_stop_threshold and i_episode % 100 == 0:
                print(f"\n{'='*60}")
                print(f"🎯 HIGH SUCCESS RATE: Achieved {success_rate*100:.1f}% success rate!")
                print(f"   Episode: {i_episode}/{max_episodes}")
                print(f"   Success threshold: {early_stop_threshold*100:.0f}%")
                print(f"{'='*60}")
                print(f"✅ Model already saved per episode! Continuing training for more robustness...")
                print("-" * 60)

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
        
        # Print periodic training progress summary (model already saved per episode)
        if i_episode % log_interval == 0:
            # Print training progress summary
            print(f"\n=== Training Progress Summary (Episode {i_episode}) ===")
            if curriculum_learning:
                print(f"🎓 Curriculum Level: {current_curriculum_level}/10 ({current_curriculum_level} obstacles)")
                print(f"📊 Level Progress: {episodes_in_current_level}/{episodes_per_level[current_level_index]} episodes")
                if level_rewards:
                    recent_avg = sum(level_rewards[-min(50, len(level_rewards)):]) / min(50, len(level_rewards))
                    print(f"📈 Recent Average Reward (Level {current_curriculum_level}): {recent_avg:.2f}")
            
            # ENHANCED: More detailed logging
            current_action_std = ppo_agent.policy.action_var[0].sqrt().item()
            current_lrs = ppo_agent.get_current_lr()
            collision_thresh = env._get_collision_threshold()
            
            print(f"🎲 Current Action Std: {current_action_std:.4f} ({current_action_std*100:.1f}% exploration)")
            print(f"📚 Learning Rates - Actor: {current_lrs['actor_lr']:.6f}, Critic: {current_lrs['critic_lr']:.6f}")
            print(f"🛡️  Collision Threshold: {collision_thresh:.3f}m")
            print(f"🚀 Velocity Range: [-1.0, 1.0] (no constraints)")
            print(f"💰 Latest Episode Reward: {episode_reward:.1f}")
            print(f"⏱️  Latest Episode Length: {episode_length} steps")
            
            # Success rate tracking
            if len(success_window) > 0:
                recent_success_rate = sum(success_window) / len(success_window)
                print(f"🎯 Recent Success Rate: {recent_success_rate*100:.1f}% (last {len(success_window)} episodes)")
                
            # Boundary avoidance tracking
            boundary_violations = [1 if termination_info.get('termination_reason') == 'out_of_bounds' else 0]
            if boundary_violations:
                print(f"🧭 Boundary Issue: {'Western' if termination_info.get('final_position', [0])[0] < -3.9 else 'Other'} boundary")
                    
            # Track best performance (model already saved per episode)
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"✓ New best performance achieved with reward: {episode_reward:.1f}")
            else:
                print(f"📝 Current best: {best_reward:.1f}")
            print("=" * 50)

            # Save metrics plot periodically
            save_training_plots(ns_lambda)

if __name__ == '__main__':
    main()