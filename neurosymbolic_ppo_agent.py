"""
Neurosymbolic PPO Agent
Integrates symbolic RDR rules with RL learning for enhanced UAV navigation.
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from neurosymbolic_rdr import RDRKnowledgeBase, NeuroSymbolicIntegrator

class NeuroSymbolicActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, use_symbolic_features=True):
        super(NeuroSymbolicActorCritic, self).__init__()
        
        self.use_symbolic_features = use_symbolic_features
        
        # Enhanced input dimension if using symbolic features
        input_dim = state_dim
        if use_symbolic_features:
            # Add extra dimensions for symbolic rule activations
            input_dim += 10  # Number of rule types we track
        
        # Actor network with symbolic awareness
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
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

        # Critic network with symbolic awareness
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
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

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self, state):
        raise NotImplementedError

    def encode_symbolic_features(self, observation, applicable_rules):
        """Encode symbolic rule information as additional features"""
        if not self.use_symbolic_features:
            return observation
        
        # Create rule activation vector
        rule_features = np.zeros(10)  # 10 different rule types
        
        rule_type_mapping = {
            "avoid_obstacle": 0,
            "turn_towards_goal": 1,
            "slow_down": 2,
            "speed_up": 3,
            "move_forward": 4,
            "move_backward": 5,
            "move_left": 6,
            "move_right": 7,
            "hover": 8,
            "no_advice": 9
        }
        
        for rule in applicable_rules:
            rule_type = rule.action_advice.value
            if rule_type in rule_type_mapping:
                idx = rule_type_mapping[rule_type]
                rule_features[idx] = rule.confidence
        
        # Concatenate original observation with rule features
        enhanced_obs = np.concatenate([observation, rule_features])
        return enhanced_obs

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

class NeuroSymbolicPPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 action_std_init=0.6, integration_weight=0.3, knowledge_file="uav_navigation_rules.json"):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.integration_weight = integration_weight
        
        # Initialize neurosymbolic components
        self.rdr_kb = RDRKnowledgeBase(knowledge_file)
        self.ns_integrator = NeuroSymbolicIntegrator(self.rdr_kb, integration_weight)
        
        print(f"🧠 Neurosymbolic PPO Agent initialized with {len(self.rdr_kb.rules)} rules")
        
        # Enhanced state dimension for symbolic features
        enhanced_state_dim = state_dim + 10  # 10 rule activation features
        
        self.policy = NeuroSymbolicActorCritic(state_dim, action_dim, action_std_init, use_symbolic_features=True)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = NeuroSymbolicActorCritic(state_dim, action_dim, action_std_init, use_symbolic_features=True)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # Learning rate schedulers
        from torch.optim.lr_scheduler import StepLR
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9997)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # Performance tracking
        self.symbolic_advice_count = 0
        self.successful_advice_count = 0
        self.episode_count = 0

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        """Select action with neurosymbolic integration"""
        # Extract features for symbolic reasoning
        features = self.ns_integrator.extract_observation_features(state)
        applicable_rules = self.rdr_kb.get_applicable_rules(features)
        
        # Enhance state with symbolic features
        enhanced_state = self.policy.encode_symbolic_features(state, applicable_rules)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state_tensor)
            
        # Get RL action
        rl_action = action.cpu().numpy().flatten()
        
        # Integrate with symbolic advice
        final_action = self.ns_integrator.integrate_with_rl_action(rl_action, state)
        
        # Track symbolic advice usage
        if self.ns_integrator.advice_log and len(self.ns_integrator.advice_log) > 0:
            last_advice = self.ns_integrator.advice_log[-1]
            if last_advice["advice_type"] != "no_advice":
                self.symbolic_advice_count += 1
        
        return final_action, action_logprob.cpu().numpy().flatten()

    def update(self, memory):
        """Update policy with neurosymbolic integration"""
        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(memory.states).to(torch.float32), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(torch.float32), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs).to(torch.float32), 1).detach()
        
        # Enhance states with symbolic features for all states in memory
        enhanced_states = []
        for i, state in enumerate(memory.states):
            state_np = state.cpu().numpy() if torch.is_tensor(state) else state
            features = self.ns_integrator.extract_observation_features(state_np)
            applicable_rules = self.rdr_kb.get_applicable_rules(features)
            enhanced_state = self.policy.encode_symbolic_features(state_np, applicable_rules)
            enhanced_states.append(torch.FloatTensor(enhanced_state))
        
        enhanced_states = torch.stack(enhanced_states).detach()
        
        # Monte Carlo estimate of rewards
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
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(enhanced_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Actor loss with entropy bonus
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = self.MseLoss(state_values, rewards)
            
            # Total loss with entropy regularization
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            # Take gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            
            self.optimizer.step()
        
        # Decay learning rate
        self.scheduler.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def update_symbolic_performance(self, observation, success):
        """Update symbolic rule performance tracking"""
        self.ns_integrator.update_rule_performance(observation, success)
        if success and self.symbolic_advice_count > 0:
            self.successful_advice_count += 1

    def get_symbolic_stats(self):
        """Get statistics about symbolic rule usage"""
        if self.symbolic_advice_count == 0:
            return {
                "advice_usage_rate": 0.0,
                "advice_success_rate": 0.0,
                "total_rules": len(self.rdr_kb.rules),
                "total_advice_given": 0
            }
        
        return {
            "advice_usage_rate": self.symbolic_advice_count / max(self.episode_count, 1),
            "advice_success_rate": self.successful_advice_count / self.symbolic_advice_count,
            "total_rules": len(self.rdr_kb.rules),
            "total_advice_given": self.symbolic_advice_count
        }

    def save(self, checkpoint_path):
        """Save both RL model and symbolic knowledge"""
        # Save RL model
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'symbolic_stats': self.get_symbolic_stats()
        }, checkpoint_path)
        
        # Save symbolic knowledge base
        self.rdr_kb.save_knowledge_base()
        
        # Save advice log
        self.ns_integrator.save_advice_log()

    def load(self, checkpoint_path):
        """Load both RL model and symbolic knowledge"""
        try:
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            if 'symbolic_stats' in checkpoint:
                stats = checkpoint['symbolic_stats']
                print(f"Loaded model with symbolic stats: {stats}")
                
            print(f"Loaded NeuroSymbolic PPO model from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_current_lr(self):
        """Get current learning rates"""
        return {
            'actor_lr': self.optimizer.param_groups[0]['lr'],
            'critic_lr': self.optimizer.param_groups[1]['lr']
        }

    def add_human_rule(self, conditions, action_advice, confidence=0.9, priority=5):
        """Allow human experts to add rules during training"""
        rule_id = self.rdr_kb.add_rule(conditions, action_advice, confidence, priority, "human")
        print(f"Added human rule {rule_id}: {action_advice.value} with confidence {confidence}")
        return rule_id

# Compatibility wrapper for existing code
class PPOAgent(NeuroSymbolicPPOAgent):
    """Compatibility wrapper to maintain existing interface"""
    
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 action_std_init=0.6, enable_neurosymbolic=True):
        
        if enable_neurosymbolic:
            super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init)
        else:
            # Fallback to original PPO implementation
            from ppo_agent import PPOAgent as OriginalPPOAgent
            self.__dict__.update(OriginalPPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init).__dict__)