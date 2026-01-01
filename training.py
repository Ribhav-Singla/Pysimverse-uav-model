import torch
import numpy as np
from uav_env import UAVEnv, CONFIG
from ppo_agent import PPOAgent
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from collections import deque
import time
import pickle

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
    parser = argparse.ArgumentParser(description='Train UAV with different PPO variants')
    parser.add_argument('--ppo_type', type=str, choices=['vanilla', 'ar', 'ns'], default='ns',
                        help='PPO variant to train: vanilla (basic PPO), ar (PPO with additional rewards), ns (neurosymbolic PPO) (default: ns)')
    parser.add_argument('--episodes', type=int, default=40,
                        help='Episodes per curriculum level (default: 40)')
    parser.add_argument('--start_level', type=int, default=1,
                        help='Starting curriculum level (default: 1)')
    parser.add_argument('--end_level', type=int, default=15,
                        help='Ending curriculum level (default: 15)')
    args = parser.parse_args()
    
    ############## Hyperparameters ##############
    env_name = "UAVEnv"
    render = False
    solved_reward = 1000         # threshold to save best model (but continue training)
    log_interval = 5          # print avg reward in the interval
    
    # Curriculum Learning Parameters
    curriculum_learning = True
    episodes_per_level_count = args.episodes  # Episodes per curriculum level (from command line)
    start_level = args.start_level  # Starting level (from command line)
    end_level = args.end_level      # Ending level (from command line)
    total_levels = 15           # Obstacle levels 1-15
    
    # Validate level range
    if start_level < 1 or end_level > total_levels or start_level > end_level:
        print(f"❌ Invalid level range: {start_level}-{end_level}. Must be between 1-{total_levels}")
        return
    
    # Calculate training scope
    levels_to_train = end_level - start_level + 1
    
    # Set equal episodes for each level
    episodes_per_level = [episodes_per_level_count] * total_levels
    total_episodes = episodes_per_level_count * levels_to_train  # Only train selected levels
    max_episodes = total_episodes
    max_timesteps = 20000        # max timesteps in one episode

    update_timestep = 1024      # OPTIMIZED: update policy every 1024 timesteps (was 2048)
    action_std = 1.0            # OPTIMIZED: Start with 100% exploration (was 0.3)
    K_epochs = 10               # update policy for K epochs (reduced for faster updates)
    eps_clip = 0.1              # clip parameter for PPO (reduced for finer control)
    gamma = 0.999               # increased for longer-term planning

    lr_actor = 0.00005           # learning rate for actor (standard PPO value: 3e-4)
    lr_critic = 0.0002           # learning rate for critic (standard PPO value: 1e-3)

    random_seed = 0
    #############################################

    # Map PPO type to configuration
    ppo_type = args.ppo_type
    if ppo_type == 'vanilla':
        # Vanilla PPO: lambda=0, no extra rewards
        ns_lambda = 0.0
        use_neurosymbolic = False
        use_extra_rewards = False
        print(f"🤖 Training Mode: VANILLA PPO (Basic rewards only)")
    elif ppo_type == 'ar':
        # AR PPO: lambda=0, with extra rewards (boundary penalties, goal detection)
        ns_lambda = 0.0
        use_neurosymbolic = False
        use_extra_rewards = True
        print(f"💰 Training Mode: AR PPO (Augmented Rewards)")
    elif ppo_type == 'ns':
        # NS PPO: lambda=1, neurosymbolic behavior
        ns_lambda = 1.0
        use_neurosymbolic = True
        use_extra_rewards = False  # NS uses neurosymbolic logic instead
        print(f"🧠 Training Mode: NS PPO (Neurosymbolic)")
    
    ns_cfg = {
        'use_neurosymbolic': use_neurosymbolic,
        'use_extra_rewards': use_extra_rewards,
        'ppo_type': ppo_type,
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

    # checkpoint path with PPO type
    checkpoint_path = "PPO_preTrained/{}/".format(env_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Create weights filename with PPO type
    ppo_type_map = {
        'vanilla': 'Vanilla_PPO',
        'ar': 'AR_PPO', 
        'ns': 'NS_PPO'
    }
    weights_filename = f"{ppo_type_map[ppo_type]}_UAV_Weights.pth"

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
    episode_r1_usage_list = []  # Percentage of steps using R1 rule (NS PPO only)
    episode_r2_usage_list = []  # Percentage of steps using R2 rule (NS PPO only)

    def save_training_plots(ppo_type, real_time=False):
        if len(episode_indices) == 0:
            return
        
        # Calculate exponential moving averages for success rate
        alpha = 0.05  # Smoothing factor (0.05 = more smoothing, 0.3 = less smoothing)
        success_np = np.array(episode_success_list, dtype=float)
        
        # Initialize EMA arrays
        ema_success = np.zeros_like(success_np)
        
        # Calculate EMA
        ema_success[0] = success_np[0]
        for i in range(1, len(success_np)):
            ema_success[i] = alpha * success_np[i] + (1 - alpha) * ema_success[i-1]
        
        # Use full length arrays for plotting
        roll_success = ema_success
        x_roll = episode_indices

        plt.figure(figsize=(12, 6))
        
        # Success rate plot
        plt.scatter(episode_indices, success_np, color='black', alpha=1.0, s=8, label='Success (0/1)')
        plt.plot(x_roll, roll_success, color='green', linewidth=2, label='Success Rate EMA')
        
        # Add current success marker if real-time
        if real_time and len(episode_indices) > 0:
            current_success = success_np[-1]
            current_episode = episode_indices[-1]
            success_color = 'green' if current_success == 1 else 'red'
            plt.scatter([current_episode], [current_success], color=success_color, s=50, zorder=10, 
                       label=f'Current: {"Success" if current_success == 1 else "Failed"}')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Goal Achievement', fontsize=12)
        plt.title(f'Goal Achievement Progress - {ppo_type.upper()} PPO', fontsize=14, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        plt.tight_layout()
        
        # Professional filename
        plot_filename = f'goal_achievement_{ppo_type.upper()}.png'
        
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if real_time:
            print(f"📊 Plot updated: {plot_filename}")

    def save_r2_usage_plot(ppo_type):
        """Plot R1 and R2 rule usage percentage over episodes (NS PPO only)"""
        if ppo_type != 'ns' or len(episode_indices) == 0:
            return
        
        r1_usage_np = np.array(episode_r1_usage_list, dtype=float)
        r2_usage_np = np.array(episode_r2_usage_list, dtype=float)
        
        # Combined plot with both R1 and R2
        plt.figure(figsize=(12, 6))
        
        # Plot R1 line chart
        plt.plot(episode_indices, r1_usage_np, color='green', linewidth=2, marker='s', 
                markersize=3, alpha=0.7, label='R1 Clear Path %')
        
        # Plot R2 line chart
        plt.plot(episode_indices, r2_usage_np, color='blue', linewidth=2, marker='o', 
                markersize=3, alpha=0.7, label='R2 Boundary Safety %')
        
        # Add moving average for R1
        if len(r1_usage_np) >= 10:
            window = min(20, len(r1_usage_np) // 5)
            moving_avg_r1 = np.convolve(r1_usage_np, np.ones(window)/window, mode='valid')
            x_avg = episode_indices[window-1:]
            plt.plot(x_avg, moving_avg_r1, color='darkgreen', linewidth=2.5, 
                    label=f'R1 MA (window={window})', alpha=0.8, linestyle='--')
        
        # Add moving average for R2
        if len(r2_usage_np) >= 10:
            window = min(20, len(r2_usage_np) // 5)
            moving_avg_r2 = np.convolve(r2_usage_np, np.ones(window)/window, mode='valid')
            x_avg = episode_indices[window-1:]
            plt.plot(x_avg, moving_avg_r2, color='darkblue', linewidth=2.5, 
                    label=f'R2 MA (window={window})', alpha=0.8, linestyle='--')
        
        plt.xlabel('Episode Number', fontsize=12)
        plt.ylabel('Rule Usage (%)', fontsize=12)
        plt.title('RDR Rules Usage - NS PPO', fontsize=14, fontweight='bold')
        plt.ylim(-5, 105)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        plt.tight_layout()
        
        plot_filename = 'rdr_rules_usage_NS.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 RDR rules usage plot saved: {plot_filename}")
        
        # Separate R1 plot
        plt.figure(figsize=(12, 6))
        plt.plot(episode_indices, r1_usage_np, color='green', linewidth=2, marker='s', 
                markersize=3, alpha=0.7, label='R1 Clear Path %')
        
        if len(r1_usage_np) >= 10:
            window = min(20, len(r1_usage_np) // 5)
            moving_avg_r1 = np.convolve(r1_usage_np, np.ones(window)/window, mode='valid')
            x_avg = episode_indices[window-1:]
            plt.plot(x_avg, moving_avg_r1, color='darkgreen', linewidth=2.5, 
                    label=f'R1 MA (window={window})', alpha=0.8, linestyle='--')
        
        plt.xlabel('Episode Number', fontsize=12)
        plt.ylabel('R1 Rule Usage (%)', fontsize=12)
        plt.title('R1 Clear Path Rule Usage - NS PPO', fontsize=14, fontweight='bold')
        plt.ylim(-5, 105)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        plt.tight_layout()
        
        r1_plot_filename = 'r1_rule_usage_NS.png'
        plt.savefig(r1_plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 R1 usage plot saved: {r1_plot_filename}")
        
        # Separate R2 plot
        plt.figure(figsize=(12, 6))
        plt.plot(episode_indices, r2_usage_np, color='blue', linewidth=2, marker='o', 
                markersize=3, alpha=0.7, label='R2 Boundary Safety %')
        
        if len(r2_usage_np) >= 10:
            window = min(20, len(r2_usage_np) // 5)
            moving_avg_r2 = np.convolve(r2_usage_np, np.ones(window)/window, mode='valid')
            x_avg = episode_indices[window-1:]
            plt.plot(x_avg, moving_avg_r2, color='darkblue', linewidth=2.5, 
                    label=f'R2 MA (window={window})', alpha=0.8, linestyle='--')
        
        plt.xlabel('Episode Number', fontsize=12)
        plt.ylabel('R2 Rule Usage (%)', fontsize=12)
        plt.title('R2 Boundary Safety Rule Usage - NS PPO', fontsize=14, fontweight='bold')
        plt.ylim(-5, 105)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        plt.tight_layout()
        
        r2_plot_filename = 'r2_rule_usage_NS.png'
        plt.savefig(r2_plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 R2 usage plot saved: {r2_plot_filename}")


    def plot_episode_reward(episode_num, reward, ppo_type):
        """Quick plot update after each episode"""
        if episode_num % 10 == 0 or episode_num <= 20:  # Plot every 10 episodes, or first 20
            save_training_plots(ppo_type, real_time=True)
    
    def print_reward_progress(episode_num, reward, window_size=10):
        """Print reward progress with moving average"""
        if len(episode_rewards_list) >= window_size:
            recent_avg = sum(episode_rewards_list[-window_size:]) / window_size
            print(f"📈 Ep {episode_num}: Reward = {reward:.1f} | Avg(last {window_size}) = {recent_avg:.1f}")
        else:
            print(f"📈 Ep {episode_num}: Reward = {reward:.1f}")
        
        # Print reward milestone achievements
        if reward > 800:
            print(f"🌟 MILESTONE: Episode {episode_num} achieved {reward:.1f} reward!")
        elif reward > 500:
            print(f"⭐ GOOD: Episode {episode_num} achieved {reward:.1f} reward!")
        elif reward > 0:
            print(f"✓ POSITIVE: Episode {episode_num} achieved {reward:.1f} reward")
        else:
            print(f"📉 Episode {episode_num}: {reward:.1f} reward")

    print("--------------------------------------------------------------------------------------------")
    print("🚀 OPTIMIZED PPO TRAINING CONFIGURATION")
    print("--------------------------------------------------------------------------------------------")
    print("🧭 PPO VARIANT CONFIGURATION:")
    print(f"   - PPO Type: {ppo_type.upper()}")
    print(f"   - Lambda (λ): {ns_lambda}")
    print(f"   - Extra Rewards: {'ENABLED' if use_extra_rewards else 'DISABLED'}")
    print(f"   - Weights filename: {weights_filename}")
    print(f"   - Episodes per level: {episodes_per_level_count}")
    print(f"   - RDR system: {'ENABLED' if use_neurosymbolic else 'DISABLED'}")
    if use_extra_rewards:
        print(f"   - Boundary penalty: -1 per step when within 1m of boundary")
        print(f"   - LIDAR goal detection: 1.5x reward when moving toward detected goal")
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
    print("💰 REWARD STRUCTURE:")
    if ppo_type == 'vanilla':
        print(f"   - VANILLA PPO: Basic rewards only")
        print(f"   - Goal Reached: +100")
        print(f"   - Collision/OOB: -100")
        print(f"   - Progress Reward: 10.0 × progress")
        print(f"   - Step penalty: {CONFIG['step_reward']} per step")
    elif ppo_type == 'ar':
        print(f"   - AR PPO: Augmented rewards enabled")
        print(f"   - Goal Reached: +100")
        print(f"   - Collision/OOB: -100")
        print(f"   - Progress Reward: 10.0 × progress")
        print(f"   - Boundary Penalty: -1 per step when approaching boundary")
        print(f"   - LIDAR Goal Detection: 1.5x reward when moving toward goal")
        print(f"   - Step penalty: {CONFIG['step_reward']} per step")
    else:  # ns
        print(f"   - NS PPO: Neurosymbolic with RDR system")
        print(f"   - Goal Reached: +100")
        print(f"   - Collision/OOB: -100")
        print(f"   - Progress Reward: 10.0 × progress")
        print(f"   - Step penalty: {CONFIG['step_reward']} per step")
        print(f"   - RDR rule-based actions when available")
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
    print(f"   - Training levels {start_level}-{end_level} (of {total_levels} total levels)")
    print(f"   - Total episodes: {total_episodes}")
    print("   - Episodes per level (equal distribution):")
    for i in range(start_level - 1, end_level):
        episodes = episodes_per_level[i]
        print(f"     Level {i+1} ({i+1} obstacles): {episodes} episodes")
    print(f"   - 20 different maps per obstacle level")
    print("📊 LOGGING ENABLED")
    print(f"   - Episode data: 'curriculum_learning_log_{ppo_type}.csv'")
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
    current_curriculum_level = start_level  # Start from specified level
    current_level_index = start_level - 1  # Index for episodes_per_level list (0-indexed)
    episodes_in_current_level = 0
    level_rewards = []  # Track rewards for current level
    
    # training loop (start from episode 1)
    for i_episode in range(1, max_episodes+1):
        # Check if we need to advance curriculum level
        if curriculum_learning and episodes_in_current_level >= episodes_per_level[current_level_index]:
            if current_curriculum_level < end_level:  # Only advance up to end_level
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
        
        # Track RDR rule usage for NS PPO
        r1_step_count = 0
        r2_step_count = 0
        total_steps_in_episode = 0
        
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
                    
                    # Track RDR rule usage
                    if ppo_type == 'ns' and env.current_rule:
                        if env.current_rule.rule_id == "R1_CLEAR_PATH":
                            r1_step_count += 1
                        elif env.current_rule.rule_id == "R2_BOUNDARY_SAFETY":
                            r2_step_count += 1
                    
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
            
            # Track total steps for R2 usage calculation
            total_steps_in_episode += 1

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
        
        # Calculate RDR rule usage percentages for NS PPO
        if ppo_type == 'ns':
            r1_usage_percentage = (r1_step_count / total_steps_in_episode * 100) if total_steps_in_episode > 0 else 0.0
            r2_usage_percentage = (r2_step_count / total_steps_in_episode * 100) if total_steps_in_episode > 0 else 0.0
            episode_r1_usage_list.append(r1_usage_percentage)
            episode_r2_usage_list.append(r2_usage_percentage)
        else:
            episode_r1_usage_list.append(0.0)  # Not applicable for other PPO types
            episode_r2_usage_list.append(0.0)  # Not applicable for other PPO types
        
        # Track reward
        if curriculum_learning:
            level_rewards.append(episode_reward)

        # === Metrics tracking ===
        episode_indices.append(i_episode)
        episode_rewards_list.append(episode_reward)
        term_reason = termination_info.get('termination_reason', '') if 'termination_info' in locals() else getattr(env, 'last_termination_info', {}).get('termination_reason', '')
        success_flag = 1 if term_reason in ['goal_reached', 'goal_reached_and_stabilized'] else 0
        episode_success_list.append(success_flag)

        # Print reward progress for every episode
        print_reward_progress(i_episode, episode_reward, window_size=10)
        
        # Plot reward after each episode (with throttling)
        plot_episode_reward(i_episode, episode_reward, ppo_type)

        # Save model after every episode
        ppo_agent.save(os.path.join(checkpoint_path, weights_filename))
        print(f"💾 Model saved after episode {i_episode}")

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

        # Print current episode stats with reward highlighting
        reward_status = "🎯" if success_flag == 1 else "❌"
        if not (done or truncated):
            if curriculum_learning:
                print('{} Episode {} (Level {}) \t Length: {} \t Reward: {:.1f} \t Start: [{:.1f},{:.1f}] \t Goal: [{:.1f},{:.1f}]'.format(
                    reward_status, i_episode, current_curriculum_level, episode_length, episode_reward,
                    CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['goal_pos'][0], CONFIG['goal_pos'][1]))
            else:
                print('{} Episode {} \t Length: {} \t Reward: {:.1f} \t Start: [{:.1f},{:.1f}] \t Goal: [{:.1f},{:.1f}]'.format(
                    reward_status, i_episode, episode_length, episode_reward,
                    CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['goal_pos'][0], CONFIG['goal_pos'][1]))
        
        # Print immediate reward feedback
        if episode_reward > 500:
            print(f"🏆 Excellent performance! Reward: {episode_reward:.1f}")
        elif episode_reward > 0:
            print(f"✅ Positive reward: {episode_reward:.1f}")
        elif episode_reward < -500:
            print(f"⚠️  Poor performance: {episode_reward:.1f}")
        
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

            # Save comprehensive metrics plot periodically
            save_training_plots(ppo_type, real_time=False)
            print(f"📊 Comprehensive training plots updated at episode {i_episode}")
            
            # Save R2 usage plot for NS PPO
            if ppo_type == 'ns':
                save_r2_usage_plot(ppo_type)
    
    # Save episode rewards to pickle file
    rewards_data = {
        'episode_rewards': episode_rewards_list,
        'episode_indices': episode_indices,
        'ppo_type': ppo_type
    }
    rewards_filename = f'episode_rewards_{ppo_type}.pkl'
    with open(rewards_filename, 'wb') as f:
        pickle.dump(rewards_data, f)
    print(f"\n💾 Episode rewards saved to {rewards_filename}")

if __name__ == '__main__':
    main()