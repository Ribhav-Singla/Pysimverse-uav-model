import torch
import numpy as np
from uav_env import UAVEnv
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
    solved_reward = 1000         # stop training if avg_reward > solved_reward
    log_interval = 5          # print avg reward in the interval
    max_episodes = 10000      # max training episodes (increased for better learning)
    max_timesteps = 50000        # max timesteps in one episode

    update_timestep = 2048      # update policy every n timesteps (reduced for more frequent updates)
    action_std = 0.3            # constant std for action distribution (reduced for more precise actions)
    K_epochs = 10               # update policy for K epochs (reduced for faster updates)
    eps_clip = 0.1              # clip parameter for PPO (reduced for finer control)
    gamma = 0.999               # increased for longer-term planning

    lr_actor = 0.0001           # learning rate for actor (reduced to prevent overfitting)
    lr_critic = 0.0005          # learning rate for critic (reduced to prevent overfitting)

    random_seed = 0
    #############################################

    # creating environment
    env = UAVEnv()
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
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    # Load pre-trained weights (using absolute path)
    pretrained_weights = "D:\\Pysimverse-uav-model\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth"
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
    
    # Adaptive action standard deviation decay
    initial_action_std = action_std
    min_action_std = 0.1
    action_std_decay_rate = 0.05
    action_std_decay_freq = 500  # episodes
    
    # training loop (start from episode 1)
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        
        # Decay action std for more precise actions as training progresses
        if i_episode % action_std_decay_freq == 0:
            new_action_std = max(min_action_std, initial_action_std - 
                               (initial_action_std - min_action_std) * 
                               (i_episode / max_episodes))
            ppo_agent.set_action_std(new_action_std)
            print(f"Adjusted action std to {new_action_std:.4f}")
            
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
                print(f"\n--- Episode {i_episode} Terminated ---")
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
                break
        
        episode_length = t + 1

        # stop training if episode_reward > solved_reward
        if episode_reward > solved_reward:
            print("########## Solved! ##########")
            print(f"Goal reached with reward: {episode_reward:.1f}")
            print(f"Episode length: {episode_length} steps")
            # Only save to the specified weights file path
            ppo_agent.save('D:\\Pysimverse-uav-model\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth')
            print("Model saved to D:\\Pysimverse-uav-model\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth")
            break

        # Print current episode stats (simplified for non-termination cases)
        if not (done or truncated):
            print('Episode {} \t Length: {} \t Reward: {:.1f}'.format(i_episode, episode_length, episode_reward))
        
        # Save model periodically and track best performance
        if i_episode % log_interval == 0:
            # Always update the same weights file with the latest model
            ppo_agent.save('D:\\Pysimverse-uav-model\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth')
            
            # Print training progress summary
            print(f"\n=== Training Progress Summary (Episode {i_episode}) ===")
            current_action_std = ppo_agent.policy.action_var[0].sqrt().item()
            print(f"Current Action Std: {current_action_std:.4f}")
            print(f"Latest Episode Reward: {episode_reward:.1f}")
            print(f"Latest Episode Length: {episode_length} steps")
            
            # Only print "New best" message if there's improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"✓ New best model saved with reward: {episode_reward:.1f}")
            else:
                print(f"📝 Model updated (best: {best_reward:.1f})")
            print("=" * 50)

if __name__ == '__main__':
    main()