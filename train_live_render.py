#!/usr/bin/env python3
"""
Simple UAV RL Training with Real-time Rendering
Shows the training environment during episodes
"""

from train_uav_rl import UAVTrainer
import mujoco
import mujoco.viewer
import numpy as np
import time

class UAVTrainerWithLiveRender(UAVTrainer):
    """UAV Trainer with live rendering during training"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Setup live rendering
        if self.config.get('render', False):
            self._setup_live_rendering()
    
    def _setup_live_rendering(self):
        """Setup live MuJoCo rendering"""
        try:
            self.model = mujoco.MjModel.from_xml_path("uav_model.xml")
            self.data = mujoco.MjData(self.model)
            self.viewer = None
            print("🎬 Live rendering setup complete!")
        except Exception as e:
            print(f"⚠️ Warning: Could not setup live rendering: {e}")
            self.config['render'] = False
    
    def train_episode_with_render(self, episode):
        """Train one episode with live rendering"""
        if not self.config.get('render', False) or not self.model:
            return self.train_episode_without_render(episode)
        
        # Only render specific episodes to avoid overwhelming the display
        should_render = episode in self.config.get('render_episodes', [])
        
        if not should_render:
            return self.train_episode_without_render(episode)
        
        print(f"🎬 Rendering episode {episode} live...")
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Reset environment
                state, info = self.env.reset(options={'random_goal': self.config['random_goal']})
                episode_reward = 0
                episode_length = 0
                success = False
                collision = False
                
                # Reset MuJoCo simulation
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:3] = state[:3]  # Set initial position
                self.data.qpos[3:7] = [1, 0, 0, 0]  # Initial orientation
                
                while viewer.is_running():
                    # Select action
                    action, log_prob, value = self.agent.select_action(state)
                    
                    # Take step in environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Update MuJoCo visualization with new position
                    self.data.qpos[:3] = next_state[:3]
                    
                    # Apply simple hover thrust for visualization
                    self.data.ctrl[:] = [3.0, 3.0, 3.0, 3.0]
                    
                    # Step physics and update viewer
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # Store transition for learning
                    self.agent.store_transition(state, action, log_prob, reward, value, done)
                    
                    # Update metrics
                    episode_reward += reward
                    episode_length += 1
                    
                    # Check for success or collision
                    if 'goal_reached' in info and info['goal_reached']:
                        success = True
                    if 'collision' in info and info['collision']:
                        collision = True
                    
                    # Move to next state
                    state = next_state
                    
                    # Small delay for visualization
                    time.sleep(0.02)
                    
                    if done:
                        break
                
                # Show final position for a moment
                time.sleep(1.0)
                print(f"✅ Episode {episode} completed with reward: {episode_reward:.2f}")
                
                return episode_reward, episode_length, success, collision
                
        except Exception as e:
            print(f"⚠️ Rendering error: {e}")
            return self.train_episode_without_render(episode)
    
    def train_episode_without_render(self, episode):
        """Train one episode without rendering"""
        # Reset environment
        state, info = self.env.reset(options={'random_goal': self.config['random_goal']})
        episode_reward = 0
        episode_length = 0
        success = False
        collision = False
        
        while True:
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, log_prob, reward, value, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Check for success or collision
            if 'goal_reached' in info and info['goal_reached']:
                success = True
            if 'collision' in info and info['collision']:
                collision = True
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        return episode_reward, episode_length, success, collision

def main():
    """Run training with live rendering"""
    print("🚀 Starting UAV RL Training with Live Rendering...")
    
    # Configuration for live rendering demo
    config = {
        'total_episodes': 20,
        'max_episode_steps': 150,
        'update_frequency': 75,
        'save_frequency': 10,
        'eval_frequency': 10,
        'learning_rate': 1e-3,
        'gamma': 0.95,
        'clip_epsilon': 0.3,
        'k_epochs': 5,
        'entropy_coef': 0.02,
        'value_coef': 0.5,
        'device': 'cpu',
        'random_goal': False,
        'render': True,
        'render_episodes': [0, 2, 5, 10, 15, 19]  # Selected episodes for rendering
    }
    
    trainer = UAVTrainerWithLiveRender(config)
    
    print(f"\n🎯 Training Configuration:")
    print(f"  Episodes: {config['total_episodes']}")
    print(f"  Episodes with rendering: {config['render_episodes']}")
    print(f"  Max steps per episode: {config['max_episode_steps']}")
    
    # Run simplified training loop
    total_steps = 0
    
    for episode in range(config['total_episodes']):
        episode_start_time = time.time()
        
        # Train episode (with or without rendering)
        episode_reward, episode_length, success, collision = trainer.train_episode_with_render(episode)
        
        episode_time = time.time() - episode_start_time
        total_steps += episode_length
        
        # Store metrics
        trainer.episode_rewards.append(episode_reward)
        trainer.episode_lengths.append(episode_length)
        trainer.episode_times.append(episode_time)
        trainer.success_rate.append(1.0 if success else 0.0)
        trainer.collision_rate.append(1.0 if collision else 0.0)
        
        # Update agent if enough steps
        if total_steps % config['update_frequency'] == 0:
            update_info = trainer.agent.update()
            if update_info:
                print(f"Step {total_steps}: Policy updated - Loss: {update_info['total_loss']:.4f}")
        
        # Print episode summary
        current_success_rate = np.mean(trainer.success_rate) if trainer.success_rate else 0.0
        current_collision_rate = np.mean(trainer.collision_rate) if trainer.collision_rate else 0.0
        
        render_indicator = "🎬" if episode in config['render_episodes'] else ""
        print(f"Episode {episode:3d} {render_indicator} | "
              f"Reward: {episode_reward:6.2f} | "
              f"Length: {episode_length:3d} | "
              f"Time: {episode_time:5.2f}s | "
              f"Success: {current_success_rate:.2f} | "
              f"Collision: {current_collision_rate:.2f}")
    
    print("\n🎉 Training completed!")
    print("🏆 Final performance summary:")
    print(f"  Average reward: {np.mean(trainer.episode_rewards[-5:]):.2f}")
    print(f"  Success rate: {np.mean(trainer.success_rate):.1%}")

if __name__ == "__main__":
    main()