# UAV RL Training Script
import numpy as np
import time
import os
from collections import deque
import pickle

# Import our custom modules
from uav_env import UAVNavigationEnv
from ppo_agent import PPOAgent

class UAVTrainer:
    """
    Training manager for UAV RL agent
    """
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'total_episodes': 1000,
            'max_episode_steps': 500,
            'update_frequency': 2048,  # Update every N steps
            'save_frequency': 100,     # Save model every N episodes
            'eval_frequency': 50,      # Evaluate every N episodes
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'k_epochs': 10,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'device': 'cpu',  # Use 'cuda' if available
            'random_goal': False  # Whether to randomize goal positions
        }
        
        if config:
            self.config.update(config)
        
        # Initialize environment
        self.env = UAVNavigationEnv(max_episode_steps=self.config['max_episode_steps'])
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            clip_epsilon=self.config['clip_epsilon'],
            k_epochs=self.config['k_epochs'],
            entropy_coef=self.config['entropy_coef'],
            value_coef=self.config['value_coef'],
            device=self.config['device']
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.success_rate = deque(maxlen=100)
        self.collision_rate = deque(maxlen=100)
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def _setup_rendering(self):
        """Setup MuJoCo model and viewer for rendering"""
        try:
            # Load the MuJoCo model that the environment uses
            self.model = mujoco.MjModel.from_xml_path("uav_model.xml")
            self.data = mujoco.MjData(self.model)
            print("🎬 Rendering setup complete!")
        except Exception as e:
            print(f"⚠️ Warning: Could not setup rendering: {e}")
            self.config['render'] = False
    
    def _should_render_episode(self, episode):
        """Determine if this episode should be rendered"""
        if not self.config['render']:
            return False
        
        # Render specific episodes from the list
        if episode in self.config['render_episodes']:
            return True
        
        # Render every N episodes after the initial list
        if episode > max(self.config['render_episodes'], default=0):
            return episode % self.config['render_frequency'] == 0
        
        return False
    
    def _render_episode(self, episode, agent_states, agent_actions):
        """Render an episode showing the agent's behavior"""
        if not self.config['render'] or not self.model or not self.data:
            return
        
        print(f"🎬 Rendering episode {episode}...")
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Reset to initial state
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:3] = agent_states[0][:3]  # Initial position
                self.data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation
                
                step_delay = 0.05  # Delay between steps for visualization
                
                for i, (state, action) in enumerate(zip(agent_states, agent_actions)):
                    if not viewer.is_running():
                        break
                    
                    # Update UAV position from agent state
                    self.data.qpos[:3] = state[:3]
                    
                    # Apply action as motor controls (fixed broadcasting)
                    # Ensure action is 3D and convert to 4 motor thrusts
                    if len(action) == 3:
                        # Convert 3D velocity action to 4 motor thrusts
                        base_thrust = 3.0
                        motor_thrusts = np.array([
                            base_thrust + action[0] + action[1],  # Front-right
                            base_thrust - action[0] + action[1],  # Front-left
                            base_thrust + action[0] - action[1],  # Back-right
                            base_thrust - action[0] - action[1]   # Back-left
                        ])
                        self.data.ctrl[:4] = np.clip(motor_thrusts, 1.0, 6.0)
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    time.sleep(step_delay)
                    
                    # Print progress every 20 steps
                    if i % 20 == 0:
                        pos = state[:3]
                        print(f"  Step {i}: UAV at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                
                # Hold final frame for a moment
                time.sleep(1.0)
                print(f"✅ Episode {episode} rendering complete!")
                
        except Exception as e:
            print(f"⚠️ Rendering error: {e}")
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting UAV RL Training...")
        print(f"Total episodes: {self.config['total_episodes']}")
        print(f"Max steps per episode: {self.config['max_episode_steps']}")
        print(f"Update frequency: {self.config['update_frequency']}")
        
        total_steps = 0
        best_success_rate = 0.0
        
        for episode in range(self.config['total_episodes']):
            episode_start_time = time.time()
            
            # Reset environment
            state, info = self.env.reset(options={'random_goal': self.config['random_goal']})
            episode_reward = 0
            episode_length = 0
            
            success = False
            collision = False
            
            # Store episode data for rendering
            episode_states = []
            episode_actions = []
            should_render = self._should_render_episode(episode)
            
            while True:
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Store data for rendering if needed
                if should_render:
                    episode_states.append(state.copy())
                    episode_actions.append(action.copy())
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, log_prob, reward, value, done)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Check for success or collision
                if 'goal_reached' in info and info['goal_reached']:
                    success = True
                if 'collision' in info and info['collision']:
                    collision = True
                
                # Update agent
                if total_steps % self.config['update_frequency'] == 0:
                    # Get next state value for GAE computation
                    if not done:
                        _, _, next_value = self.agent.select_action(next_state)
                        next_value = next_value[0]
                    else:
                        next_value = 0
                    
                    # Update policy
                    update_info = self.agent.update()
                    
                    if update_info:
                        print(f"Step {total_steps}: Policy updated - "
                              f"Loss: {update_info['total_loss']:.4f}")
                
                # Move to next state
                state = next_state
                
                if done:
                    break
            
            # Episode finished
            episode_time = time.time() - episode_start_time
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_times.append(episode_time)
            self.success_rate.append(1.0 if success else 0.0)
            self.collision_rate.append(1.0 if collision else 0.0)
            
            # Render episode if needed
            if should_render and len(episode_states) > 0:
                self._render_episode(episode, episode_states, episode_actions)
            
            # Print episode summary
            if episode % 10 == 0 or episode < 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                current_success_rate = np.mean(self.success_rate) if self.success_rate else 0.0
                current_collision_rate = np.mean(self.collision_rate) if self.collision_rate else 0.0
                
                render_indicator = "🎬" if should_render else ""
                print(f"Episode {episode:4d} {render_indicator} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Length: {episode_length:5d} | "
                      f"Time: {episode_time:6.2f}s | "
                      f"Success: {current_success_rate:.2f} | "
                      f"Collision: {current_collision_rate:.2f}")
            
            # Save model periodically
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                model_path = f"models/ppo_uav_episode_{episode}.pth"
                self.agent.save_model(model_path)
                print(f"💾 Model saved: {model_path}")
                
                # Save best model based on success rate
                current_success_rate = np.mean(self.success_rate) if self.success_rate else 0.0
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    best_model_path = "models/ppo_uav_best.pth"
                    self.agent.save_model(best_model_path)
                    print(f"🏆 New best model saved: {best_model_path} (Success rate: {best_success_rate:.3f})")
            
            # Generate plots periodically
            if episode % self.config['eval_frequency'] == 0 and episode > 0:
                self.plot_training_curves()
                self.save_training_data()
        
        print("✅ Training completed!")
        
        # Final model save
        final_model_path = "models/ppo_uav_final.pth"
        self.agent.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Generate final plots
        self.plot_training_curves()
        self.save_training_data()
    
    def plot_training_curves(self):
        """Generate training performance plots"""
        if len(self.episode_rewards) < 10:
            return
        
        # Skip plotting if rendering is enabled to avoid threading issues on macOS
        if self.config['render']:
            print("📊 Skipping plot generation during rendering to avoid threading conflicts")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UAV RL Training Progress', fontsize=16)
        
        # Moving averages for smoothing
        window_size = min(50, len(self.episode_rewards) // 4)
        
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window), 'valid') / window
        
        episodes = range(len(self.episode_rewards))
        smooth_episodes = range(window_size-1, len(self.episode_rewards))
        
        # 1. Episode Rewards
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        smooth_rewards = moving_average(self.episode_rewards, window_size)
        axes[0, 0].plot(smooth_episodes, smooth_rewards, color='blue', linewidth=2, label='Smoothed')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Episode Lengths (Time to reach goal or collision)
        axes[0, 1].plot(episodes, self.episode_lengths, alpha=0.3, color='green', label='Raw')
        smooth_lengths = moving_average(self.episode_lengths, window_size)
        axes[0, 1].plot(smooth_episodes, smooth_lengths, color='green', linewidth=2, label='Smoothed')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length (Steps)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Success Rate
        if len(self.success_rate) > 0:
            success_data = list(self.success_rate)
            success_episodes = range(len(episodes) - len(success_data), len(episodes))
            axes[1, 0].plot(success_episodes, success_data, alpha=0.3, color='gold', label='Raw')
            
            if len(success_data) >= window_size:
                smooth_success = moving_average(success_data, window_size)
                smooth_success_episodes = range(len(episodes) - len(success_data) + window_size - 1, len(episodes))
                axes[1, 0].plot(smooth_success_episodes, smooth_success, 
                               color='gold', linewidth=2, label='Smoothed')
        
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate (Last 100 Episodes)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Episode Time
        axes[1, 1].plot(episodes, self.episode_times, alpha=0.3, color='red', label='Raw')
        smooth_times = moving_average(self.episode_times, window_size)
        axes[1, 1].plot(smooth_episodes, smooth_times, color='red', linewidth=2, label='Smoothed')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Episode Duration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"plots/training_curves_episode_{len(self.episode_rewards)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved: {plot_path}")
    
    def save_training_data(self):
        """Save training data for later analysis"""
        training_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'success_rate': list(self.success_rate),
            'collision_rate': list(self.collision_rate),
            'config': self.config
        }
        
        data_path = f"logs/training_data_episode_{len(self.episode_rewards)}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"💾 Training data saved: {data_path}")
    
    def evaluate_agent(self, num_episodes=10, render=False):
        """Evaluate trained agent"""
        print(f"🔍 Evaluating agent for {num_episodes} episodes...")
        
        success_count = 0
        collision_count = 0
        total_rewards = []
        total_lengths = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            success = False
            collision = False
            
            while True:
                # Use deterministic policy for evaluation
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if 'goal_reached' in info and info['goal_reached']:
                    success = True
                if 'collision' in info and info['collision']:
                    collision = True
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            
            if success:
                success_count += 1
            if collision:
                collision_count += 1
            
            print(f"Eval Episode {episode+1}: Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, Success={success}, Collision={collision}")
        
        # Summary statistics
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        
        print("\n📈 Evaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.1f} steps")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Collision Rate: {collision_rate:.2%}")
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'collision_rate': collision_rate
        }

def main():
    """Main training function"""
    # Training configuration
    config = {
        'total_episodes': 50,  # Shorter for demo with rendering
        'max_episode_steps': 200,  # Shorter episodes
        'update_frequency': 100,  # More frequent updates
        'save_frequency': 25,
        'eval_frequency': 20,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'k_epochs': 10,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'device': 'cpu',
        'random_goal': False,
        'render': True,  # Enable rendering
        'render_episodes': [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49],  # Render these episodes
        'render_frequency': 5  # Render every 5 episodes after initial list
    }
    
    # Create trainer and start training
    trainer = UAVTrainer(config)
    trainer.train()
    
    # Evaluate final agent
    trainer.evaluate_agent(num_episodes=5)

if __name__ == "__main__":
    main()