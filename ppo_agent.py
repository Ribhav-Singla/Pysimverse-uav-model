# PPO Agent Implementation using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions are bounded
        )
        
        # Actor standard deviation (learnable)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.5)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        features = self.shared_layers(state)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_std = F.softplus(self.actor_std) + 1e-5  # Ensure positive std
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        """Evaluate actions for PPO update"""
        action_mean, action_std, value = self.forward(state)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, k_epochs=10, entropy_coef=0.01,
                 value_coef=0.5, device='cpu'):
        
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for rollouts
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Training statistics
        self.total_updates = 0
        
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state, deterministic)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().numpy() if log_prob is not None else None
        value = value.cpu().numpy().flatten()
        
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using GAE"""
        gae_lambda = 0.95
        
        returns = []
        advantages = []
        gae = 0
        
        # Convert to tensors
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Add next value for last step
        values = np.append(values, next_value)
        
        # Compute GAE advantages
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return np.array(returns), np.array(advantages)
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # Training statistics
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_loss_sum = 0
        
        # PPO update
        for _ in range(self.k_epochs):
            # Evaluate actions with current policy
            log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
            values = values.squeeze()
            
            # Compute ratio for clipping
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_loss_sum += entropy_loss.item()
        
        # Clear buffer
        self.clear_buffer()
        
        # Update counter
        self.total_updates += 1
        
        # Return training statistics
        return {
            'total_loss': total_loss / self.k_epochs,
            'policy_loss': policy_loss_sum / self.k_epochs,
            'value_loss': value_loss_sum / self.k_epochs,
            'entropy_loss': entropy_loss_sum / self.k_epochs
        }
    
    def clear_buffer(self):
        """Clear rollout buffer"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates
        }, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']

# Example usage
if __name__ == "__main__":
    # Test the PPO agent
    state_dim = 9  # [pos(3), vel(3), goal(3)]
    action_dim = 3  # [vx, vy, vz]
    
    agent = PPOAgent(state_dim, action_dim)
    
    # Test action selection
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(state)
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Sample state: {state}")
    print(f"Sample action: {action}")
    print(f"Action log prob: {log_prob}")
    print(f"State value: {value}")