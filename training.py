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
    solved_reward = 300         # stop training if avg_reward > solved_reward
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
    pretrained_weights = "D:\\pysimverse\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth"
    if os.path.exists(pretrained_weights):
        print("Loading pre-trained weights from:", pretrained_weights)
        try:
            ppo_agent.load(pretrained_weights)
            print("Pre-trained weights loaded successfully!")
        except RuntimeError as e:
            print(f"WARNING: Could not load weights due to architecture mismatch: {e}")
            print("Starting training with a new model (expanded architecture)")
    else:
        print(f"WARNING: Could not find weights at {pretrained_weights}")
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
                break
        
        episode_length = t + 1

        # stop training if episode_reward > solved_reward
        if episode_reward > solved_reward:
            print("########## Solved! ##########")
            # Only save to the specified weights file path
            ppo_agent.save('D:\\pysimverse\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth')
            print("Model saved to D:\\pysimverse\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth")
            break

        # Print current episode stats
        print('Episode {} \t Length: {} \t Reward: {:.1f}'.format(i_episode, episode_length, episode_reward))
        
        # Save model periodically and track best performance
        if i_episode % log_interval == 0:
            # Always update the same weights file with the latest model
            ppo_agent.save('D:\\pysimverse\\PPO_preTrained\\UAVEnv\\PPO_UAV_Weights.pth')
            
            # Only print "New best" message if there's improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"✓ New best model saved with reward: {episode_reward:.1f}")

if __name__ == '__main__':
    main()