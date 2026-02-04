
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import DDPGAct, DDPGCrit, OUActionNoise
from helper.replay_buffer import ReplayBuffer

class DDPGagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, capacity, batch_size , tau, gamma, actor_lr, critic_lr):
        self.actor = DDPGAct(obs_dim, hidden_dim, act_dim, 3e-3 ) 
        self.critic = DDPGCrit(obs_dim, hidden_dim, act_dim, 3e-4 )
        self.noise = OUActionNoise(mu=np.zeros(act_dim), sigma=0.15, theta=0.2, dt=1e-2)

        self.target_actor = DDPGAct(obs_dim, hidden_dim, act_dim, 3e-3 )
        self.target_critic = DDPGCrit(obs_dim, hidden_dim, act_dim, 3e-4 )

        self.tau = tau
        self.gamma = gamma
        
        self.replay_buffer = ReplayBuffer(capacity, batch_size)

        self.act_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.crit_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs, use_noise=True):
        action = self.actor(obs)
        if use_noise:
            action = torch.add(action,torch.tensor(self.noise(), dtype=action.dtype, device=action.device))
    
        return action
        
    def soft_target_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_targ.mul_(1 - self.tau)
                p_targ.add_(self.tau * p)
            
            for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_targ.mul_(1 - self.tau)
                p_targ.add_(self.tau * p)

    def compute_loss(self, next_obs, masks, rewards, state_tensor, obs_tensor, action_tensor):
        with torch.no_grad():
            next_action = self.target_actor(next_obs) 
            next_value = self.target_critic(next_obs, next_action)
            targets = rewards + self.gamma * next_value * masks  
        
        values = self.critic(state_tensor, action_tensor) 
        critic_loss = F.mse_loss(values, targets, reduction='mean')
        
        action_tensor = self.actor(obs_tensor)
        actor_loss = -self.critic(obs_tensor, action_tensor).mean()

        return critic_loss, actor_loss

    def update_network(self, obs, actions, rewards, next_obs, masks):

        # -------- Critic update --------
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            q_next = self.target_critic(next_obs, next_actions)
            targets = rewards + self.gamma * q_next * masks

        q = self.critic(obs, actions)
        critic_loss = F.mse_loss(q, targets)

        self.crit_optim.zero_grad()
        critic_loss.backward()
        self.crit_optim.step()

        # -------- Actor update --------
        # Freeze critic BEFORE forward
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_actions = self.actor(obs)
        actor_loss = -self.critic(obs, actor_actions).mean()

        self.act_optim.zero_grad()
        actor_loss.backward()
        self.act_optim.step()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True

        # -------- Target networks --------
        self.soft_target_update()
        return critic_loss, actor_loss