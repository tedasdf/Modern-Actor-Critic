
import torch
import torch.optim as optim
from helper.rollout import Rollout
from helper.neuralnet import GuassianActor, Critic
import torch.nn.functional as F

class PPOagent():
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 hidden_dim,
                 gamma,
                 epsilon,
                 kl_target,
                 act_lr,
                 critic_lr,
                 lam
                 ):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        self.actor_op = optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_op = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.rollout = Rollout(use_mu=True, use_std=True)

        self.gamma = gamma
        self.epsilon = epsilon
        self.kl_target = kl_target
        self.lam = lam


    def select_action(self, obs):
        mu, std = self.actor(obs)
        dist = torch.distributions.Normal(mu, std)

        z = dist.rsample()
        action = torch.tanh(z)

        # Log prob with tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, mu, std
    
    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        # Forward pass: get mean and std from your GaussianActor
        mu, std = self.actor(states)  # mu, std: [batch, act_dim]
        dist = torch.distributions.Normal(mu, std)

        eps = 1e-6
        pre_tanh_action = torch.atanh(actions.clamp(-1 + eps, 1 - eps))  # inverse tanh

        log_probs = dist.log_prob(pre_tanh_action) - torch.log(1 - actions.pow(2) + eps)
        log_probs = log_probs.sum(dim=-1)  # sum over action dims

        # PPO probability ratio
        ratios = torch.exp(log_probs - old_log_probs)

        # Optional: normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return ratios, advantages

    
    def loss_clip(self, ratios, advantages):
        # ratios = πθ(a|s) / πθ_old(a|s)
        clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        return torch.min(ratios * advantages, clipped * advantages).mean()

    def loss_adaptive(self, ratios, advantages, kl):

        if kl < self.kl_target * 0.5:   # too small update
            self.epsilon = min(0.3, self.epsilon * 2)
        elif kl > self.kl_target * 1.5: # too big update
            self.epsilon = max(0.05, self.epsilon / 2)
        
        clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        return torch.min(ratios * advantages, clipped * advantages).mean()

    def actor_update(self, states, actions, advantages, old_log_probs, old_mu, old_std):
        
        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)

        eps = 1e-6
        pre_tanh_action = torch.atanh(actions.clamp(-1 + eps, 1 - eps))
        log_probs = dist.log_prob(pre_tanh_action) - torch.log(1 - actions.pow(2) + eps)
        log_probs = log_probs.sum(dim=-1)
        
        ratios , advantages = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)

        if old_mu is not None and old_std is not None:
            with torch.no_grad():
                old_dist = torch.distributions.Normal(old_mu, old_std)
                kl = torch.distributions.kl_divergence(old_dist, dist).sum(dim=-1).mean()
        else:
            kl = torch.tensor(0.0)
        
        loss = self.loss_adaptive(ratios, advantages, kl)
    
        self.actor_op.zero_grad()
        (-loss).backward()
        self.actor_op.step()
        
        return loss.item()

    def critic_update(self, states, targets, l2_reg=1e-3):
        values = self.critic(states).squeeze(-1)
        loss = F.mse_loss(values, targets)
        for p in self.critic.parameters():
            loss += l2_reg * p.pow(2).sum()
        self.critic_op.zero_grad()
        loss.backward()
        self.critic_op.step()
        return loss.item()


