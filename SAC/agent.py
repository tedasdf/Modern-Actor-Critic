
import torch
import torch.optim as optim
from utilities.neuralnet import Critic,Actor, StateValCritc
from utilities.replay_buffer import ReplayBuffer


class SACagent():
    def __init__(self, obs_dim, act_dim, hidden_size, alpha, gamma, tau, act_lr, state_lr, state_act_lr):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim, hidden_size)
        self.state_critic = Critic(obs_dim, hidden_size)
        self.target_critic = Critic(obs_dim, hidden_size)

        self.state_action_critic = StateValCritc(obs_dim, hidden_size)

        self.replay_buffer = ReplayBuffer()

        self.act_opt = optim.Adam(self.actor.parameters(), lr=act_lr)
        self.v_opt = optim.Adam(self.state_critic.parameters(), lr=state_lr)
        self.q_opt = optim.Adam(self.state_action_critic.parameters(), lr=state_act_lr)
   
    def soft_policy_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.state_critic.parameters(), self.target_critic.parameters()):
                p_targ.mul_(1 - self.tau)
                p_targ.add_(self.tau * p)

    
    def select_action(self, obs):
        action, _ = self.actor.sample(obs)

        return action
    

    def update_v(self, obs):
        state_val = self.state_critic(obs)
        a, log_pi = self.actor.sample(obs)
        
        state_act_val = self.state_action_critic(obs,a)

        with torch.no_grad():
            target = (state_act_val - log_pi * self.alpha).detach()
        
        loss = 0.5 * (state_val - target).pow(2).mean()
        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()

       
    def compute_q_loss(self, obs, actions, rewards, next_obs, masks):
        state_act_val = self.state_action_critic(obs, actions)
        
        with torch.no_grad():
            next_state_val = self.target_critic(next_obs)
            state_act_hat = (rewards + self.gamma * next_state_val * masks).detach()

        loss = 0.5 * (state_act_val - state_act_hat).pow(2).mean()
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

    def compute_policy_loss(self, obs):
        a, log_pi = self.actor.sample(obs)
        q_val = self.state_action_critic(obs, a)
        loss = (self.alpha * log_pi - q_val).mean()

        self.act_opt.zero_grad()
        loss.backward()
        self.act_opt.step()

