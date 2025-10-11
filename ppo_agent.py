import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class SymbolicRule:
    """
    Base class for symbolic rules in a neurosymbolic RL framework.
    Rules map MDP states to actions.
    """
    def __init__(self, name: str = "base_rule"):
        self.name = name
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action recommendation from this rule based on current state.
        
        Args:
            state: Environment state vector
            
        Returns:
            action: Recommended action based on this symbolic rule
        """
        # This is a placeholder to be implemented by specific rules
        raise NotImplementedError("Symbolic rule must implement get_action method")

class SymbolicRuleEngine:
    """
    Engine that manages and applies symbolic rules for neurosymbolic RL.
    """
    def __init__(self):
        self.rules: List[SymbolicRule] = []
        
    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a symbolic rule to the engine"""
        self.rules.append(rule)
        
    def get_symbolic_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get combined action recommendation from all symbolic rules.
        
        Args:
            state: Environment state vector
            
        Returns:
            action: Combined action recommendation from all symbolic rules
        """
        # For now, this is a placeholder that returns zeros (no action)
        # In a complete implementation, this would aggregate recommendations from all rules
        if not self.rules:
            # If no rules defined, return zeros with proper shape (assuming 3D action space)
            return np.zeros(3)  
            
        # If we have rules, aggregate their recommendations
        if self.rules:
            # Simple implementation: just return the first rule's recommendation
            # In a complete implementation, you might combine multiple rules with weights
            return self.rules[0].get_action(state)
        
        # For now with no active rules, return zeros which will cause no effect when lambda=0
        return np.zeros(3)

# Example of a concrete symbolic rule - this is where you add new rules
class ObstacleAvoidanceRule(SymbolicRule):
    """
    Example rule: Simple obstacle avoidance based on LIDAR readings.
    This is just an example - you would implement your own rules here.
    """
    def __init__(self, name="obstacle_avoidance", lidar_threshold=0.5):
        super().__init__(name)
        self.lidar_threshold = lidar_threshold
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action recommendation based on obstacle proximity.
        
        In UAV environment, state structure is:
        - Position (3): state[0:3]
        - Velocity (3): state[3:6]
        - Goal direction (3): state[6:9]
        - LIDAR readings (16): state[9:25]
        """
        # Extract LIDAR readings (16 rays around the UAV)
        lidar_readings = state[9:25]
        
        # Get the index of the closest obstacle
        min_dist_idx = np.argmin(lidar_readings)
        min_dist = lidar_readings[min_dist_idx]
        
        # If obstacle is too close, recommend avoidance action
        if min_dist < self.lidar_threshold:
            # Compute the angle of the closest obstacle (in radians)
            angle = min_dist_idx * (2 * np.pi / len(lidar_readings))
            
            # Recommend velocity in the opposite direction of the obstacle
            vx = -np.cos(angle)  # Move away from obstacle
            vy = -np.sin(angle)
            vz = 0.0  # No vertical movement
            
            return np.array([vx, vy, vz])
        
        # If no obstacles close by, recommend moving toward the goal
        # The goal direction is already in the state
        goal_direction = state[6:9]
        # Normalize to get unit vector
        goal_norm = np.linalg.norm(goal_direction)
        if goal_norm > 0:
            goal_direction = goal_direction / goal_norm
        
        # Scale to reasonable velocity
        goal_direction = goal_direction * 0.5
        
        # In UAV environment, we typically keep z=0 (constant height)
        return np.array([goal_direction[0], goal_direction[1], 0.0])

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6, lambda_param=0.0):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Neurosymbolic parameters
        self.lambda_param = lambda_param  # Balance between neural (0) and symbolic (1)
        self.symbolic_engine = SymbolicRuleEngine()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Neural policy components
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        old_action_std = self.policy_old.action_var[0].sqrt().item()
        new_action_std = old_action_std - action_std_decay_rate
        if (new_action_std < min_action_std):
            new_action_std = min_action_std
        self.set_action_std(new_action_std)
        
    def set_lambda(self, lambda_param):
        """
        Set the lambda parameter for neurosymbolic blending.
        Lambda=0 means pure neural, Lambda=1 means pure symbolic.
        
        Args:
            lambda_param: Value between 0 and 1
        """
        self.lambda_param = max(0.0, min(1.0, lambda_param))  # Clamp to [0,1]
        
    def add_symbolic_rule(self, rule):
        """
        Add a symbolic rule to the rule engine.
        
        Args:
            rule: A SymbolicRule object that maps states to actions
        """
        self.symbolic_engine.add_rule(rule)
        
    def add_obstacle_avoidance_rule(self, lidar_threshold=0.5):
        """
        Add an obstacle avoidance rule to the rule engine.
        This is a convenience method for a common rule type.
        
        Args:
            lidar_threshold: Distance threshold for obstacle avoidance (lower = more aggressive)
        """
        rule = ObstacleAvoidanceRule(lidar_threshold=lidar_threshold)
        self.add_symbolic_rule(rule)
        return rule

    def select_action(self, state):
        # Get neural network action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            neural_action, action_logprob = self.policy_old.act(state_tensor)
            neural_action_np = neural_action.numpy()
        
        # Get symbolic action recommendation (if lambda > 0)
        if self.lambda_param > 0:
            symbolic_action_np = self.symbolic_engine.get_symbolic_action(state)
            
            # Blend neural and symbolic actions using lambda
            # ans = λ*neural + (1-λ)*symbolic
            blended_action = (
                self.lambda_param * neural_action_np + 
                (1 - self.lambda_param) * symbolic_action_np
            )
            
            # NOTE: When lambda=0 (default), this reduces to just the neural action
            return blended_action, action_logprob.numpy()
        else:
            # When lambda=0, use pure neural action (original behavior)
            return neural_action_np, action_logprob.numpy()

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

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
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def get_neurosymbolic_info(self):
        """
        Get information about the neurosymbolic configuration.
        
        Returns:
            dict: Information about the lambda parameter and number of symbolic rules
        """
        return {
            "lambda": self.lambda_param,
            "num_rules": len(self.symbolic_engine.rules),
            "rules": [rule.name for rule in self.symbolic_engine.rules]
        }
