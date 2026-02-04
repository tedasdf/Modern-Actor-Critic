
import torch
import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import GuassianActor, Critic


class A3Cagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, lr):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        self.gamma = gamma

    def select_action(self, obs):
        action, log_prob, entropy = self.actor.sample(obs)
        
        return action, log_prob, entropy

    def compute_losses(self, rollout, targets):
        states = torch.stack(rollout["obs"])
        actions = torch.stack(rollout["actions"])
        log_probs = torch.stack(rollout["log_probs"]).squeeze(-1)
        
        values = self.critic(states).squeeze(-1)
        advantages = targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (targets - values).pow(2).mean()
        return actor_loss ,critic_loss
    
    def boostrapping_target(self, obs, rewards, masks):
        with torch.no_grad():
            R = self.critic(obs.unsqueeze(0)).squeeze(0)
        returns = []
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            R = reward + self.gamma * R * mask
            returns.insert(0, R)
        return torch.stack(returns).squeeze(-1)
    
    def update(self, actor_loss, critic_loss , actor_weight=1.0, critic_weight=0.5):

        total_loss = actor_weight * actor_loss + critic_weight * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

