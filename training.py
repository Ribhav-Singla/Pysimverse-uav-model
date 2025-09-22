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
    log_interval = 4          # print avg reward in the interval
    max_episodes = 50       # max training episodes
    max_timesteps = 5000        # max timesteps in one episode

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.6            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

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
        env.seed(random_seed)
        np.random.seed(random_seed)
        print("--------------------------------------------------------------------------------------------")

    memory = Memory()
    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
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
            running_reward += reward
            if render:
                env.render()
            if done or truncated:
                break
        
        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            ppo_agent.save(checkpoint_path + 'PPO_{}_{}.pth'.format(env_name, i_episode))
            break

        # save every 5 episodes
        if i_episode % 5 == 0:
            ppo_agent.save(checkpoint_path + 'PPO_{}_{}.pth'.format(env_name, i_episode))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
