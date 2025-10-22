import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        # UPGRADED: Hidden dimension from 128 to 256
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
        # Initialize network weights to prevent NaN
        self._initialize_weights()

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
    
    def _initialize_weights(self):
        """Initialize network weights to prevent NaN and improve stability"""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Xavier/Glorot initialization for better gradient flow
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    torch.nn.init.constant_(layer.bias, 0.0)
                elif isinstance(layer, nn.LayerNorm):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
    
    def check_and_fix_network_weights(self):
        """Check for NaN weights and reset them if found"""
        nan_found = False
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"🚨 WARNING: NaN detected in {name}! Resetting weights.")
                nan_found = True
        
        if nan_found:
            print("🔧 Reinitializing network weights due to NaN detection.")
            self._initialize_weights()
            return True
        return False

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        # Input validation - check for extreme state values
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("🚨 WARNING: Invalid state input detected! Cleaning state values.")
            state = torch.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clamp state values to reasonable range to prevent network instability
        state = torch.clamp(state, -100.0, 100.0)
        
        action_mean = self.actor(state)
        
        # NaN protection for action mean
        if torch.isnan(action_mean).any():
            print("🚨 WARNING: NaN detected in action_mean during act()! Replacing with zeros.")
            action_mean = torch.zeros_like(action_mean)
        
        # Clamp action_mean to reasonable bounds
        action_mean = torch.clamp(action_mean, -10.0, 10.0)
        
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # NaN protection for action and log probability
        if torch.isnan(action).any():
            print("🚨 WARNING: NaN detected in action! Replacing with zeros.")
            action = torch.zeros_like(action)
        
        if torch.isnan(action_logprob).any():
            print("🚨 WARNING: NaN detected in action_logprob! Replacing with zeros.")
            action_logprob = torch.zeros_like(action_logprob)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        # NaN protection - check for NaN values in action_mean
        if torch.isnan(action_mean).any():
            print("🚨 WARNING: NaN detected in action_mean! Replacing with zeros.")
            action_mean = torch.zeros_like(action_mean)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        # Additional NaN protection for covariance matrix
        if torch.isnan(cov_mat).any():
            print("🚨 WARNING: NaN detected in covariance matrix! Resetting to identity.")
            cov_mat = torch.eye(action_mean.shape[-1]).expand_as(cov_mat)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # ADDED: Learning rate schedulers for adaptive LR
        from torch.optim.lr_scheduler import StepLR
        # Decay LR by 30% every 1000 episodes (called from training loop)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9997)  # Per-update decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        old_action_std = self.policy_old.action_var[0].sqrt().item()
        new_action_std = old_action_std - action_std_decay_rate
        if (new_action_std < min_action_std):
            new_action_std = min_action_std
        self.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)

        return action.numpy(), action_logprob.numpy()

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards with NaN protection
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Check for extreme reward values before normalization
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"🚨 WARNING: Invalid rewards detected! NaN: {torch.isnan(rewards).sum()}, Inf: {torch.isinf(rewards).sum()}")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Safe reward normalization with epsilon protection
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if torch.isnan(reward_mean) or torch.isnan(reward_std) or reward_std < 1e-6:
            print(f"🚨 WARNING: Invalid reward statistics! Mean: {reward_mean}, Std: {reward_std}")
            rewards = torch.zeros_like(rewards)  # Reset to zeros if statistics are invalid
        else:
            rewards = (rewards - reward_mean) / (reward_std + 1e-7)

        # convert list to tensor (using numpy for efficiency)
        old_states = torch.squeeze(torch.tensor(np.array(memory.states), dtype=torch.float32))
        old_actions = torch.squeeze(torch.tensor(np.array(memory.actions), dtype=torch.float32))
        old_logprobs = torch.squeeze(torch.tensor(np.array(memory.logprobs), dtype=torch.float32))

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # NaN/Inf protection for loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"🚨 WARNING: Invalid loss detected! Skipping this update.")
                continue
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # ADDED: Step the learning rate scheduler
        self.scheduler.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def get_current_lr(self):
        """Get current learning rates for logging"""
        return {
            'actor_lr': self.optimizer.param_groups[0]['lr'],
            'critic_lr': self.optimizer.param_groups[1]['lr']
        }
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
