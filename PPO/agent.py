
import torch
import torch.optim as optim
from helper.rollout import Rollout
from helper.neuralnet import GuassianActor, Critic
import torch.nn.functional as F

class PPOagent():
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 hidden_size,
                 gamma,
                 epsilon,
                 kl_target,
                 act_lr,
                 critic_lr,
                 ):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)

        self.actor_op = optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_op = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.rollout = Rollout()

        self.gamma = gamma
        self.epsilon = epsilon
        self.kl_target = kl_target
        self.lam = lam


    def select_action(self, obs):
        action, log_prob = self.actor.sample(obs)

        return action, log_prob
    
    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        return ratios, advantages
    
    def loss_clip(self, ratios, advantages):
        # ratios = πθ(a|s) / πθ_old(a|s)
        clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        return torch.min(ratios * advantages, clipped * advantages).mean()

    def loss_adaptive(self, ratios, advantages, kl):
        # adapt epsilon based on current KL
        epsilon = self.epsilon
        if kl < self.kl_target / 1.5:   # too small update
            epsilon = epsilon * 2
        elif kl > self.kl_target * 1.5: # too big update
            epsilon = epsilon * 0.5
        
        clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        return torch.min(ratios * advantages, clipped * advantages).mean(), epsilon

    def actor_update(self, states, actions, advantages, old_log_probs, n_cg_steps=10):
        
        ratios , advantages = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)

        with torch.no_grad():
            mu_old, std_old = self.actor(states)  # you should save old mu/std in rollout
            mu_new, std_new = self.actor(states)
            kl = torch.log(std_new/std_old) + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2 * std_new.pow(2)) - 0.5
            kl = kl.sum(dim=-1).mean()
        
        
        loss, epsilon_new = self.loss_adaptive(ratios, advantages, kl)
    
        self.actor_op.zero_grad()
        (-loss).backward()
        self.actor_op.step()
        self.epsilon = epsilon_new 
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


