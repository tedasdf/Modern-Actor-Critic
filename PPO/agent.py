
import torch
import torch.optim as optim
from trpo.agent import Rollout
from utilities.neuralnet import Actor, Critic
import torch.nn.functional as F

class PPOagent():
    def __init__(self, obs_dim, act_dim, hidden_size):
        self.actor = Actor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)

        self.actor_op = optim.Adam(self.actor.parameters())
        self.critic_op = optim.Adam(self.critic.parameters())

        self.rollout = Rollout()

    def select_action(self, obs):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        return ratios, advantages
    
    def loss_clip(self, ratios, advantages ):
        return torch.min(ratios* advantages, torch.clip(ratios, 1 - self.epsilon , 1 + self.epsilon) * advantages)

    def actor_update(self, states, actions, advantages, old_log_probs, n_cg_steps=10):
        ratios , advantages = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        loss = -self.loss_clip(ratios, advantages).mean()
        self.actor_op.zero_grad()
        loss.backward()
        self.actor_op.step()        

    def critic_update(self, states, targets, l2_reg=1e-3):
        values = self.critic(states).squeeze(-1)
        loss = F.mse_loss(values, targets)
        for p in self.critic.parameters():
            loss += l2_reg * p.pow(2).sum()
        self.critic_op.zero_grad()
        loss.backward()
        self.critic_op.step()
        return loss.item()


