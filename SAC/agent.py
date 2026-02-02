



import torch
from utilities.neuralnet import Critic,Actor
from utilities.replay_buffer import ReplayBuffer


class SACagent():
    def __init__(self,obs_dim, act_dim, hidden_size, alpha):
        self.alpha = alpha
        self.actor = Actor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)
        
        self.target_critic = Critic(obs_dim, hidden_size)
        self.replay_buffer = ReplayBuffer()
   
    
    def soft_policy_update(self):
        raise ValueError
    
    def select_action(self, obs):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    

    def compute_state_loss(self, values, states, log_prob):
        return (0.5 * ( states - (values - log_prob)).mean()) ** 2).mean()
    
    def compute_state_value_loss(self, values, returns, states):
        return (0.5 * (values - (returns + self.gamma * states.mean()))**2)


    def compute_policy_loss(self, states):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        q_values = self.critic(states)[range(len(actions)), actions]  # Q(s,a) for sampled actions
        loss = (self.alpha * log_probs - q_values).mean()
        return loss