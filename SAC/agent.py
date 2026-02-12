
import torch
import torch.optim as optim
from helper.neuralnet import Critic,GaussianActor, StateValCritc
from helper.replay_buffer import ReplayBuffer
import numpy as np

class SACagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, alpha, gamma, tau, act_lr, state_lr, state_act_lr, capacity, batch_size, auto_alpha, target_entropy, learning_starts ):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.state_critic = Critic(obs_dim, hidden_dim)
        self.target_critic = Critic(obs_dim, hidden_dim)

        self.state_action_critic = StateValCritc(obs_dim, act_dim, hidden_dim)

        self.replay_buffer = ReplayBuffer(capacity, batch_size, learning_starts)

        self.act_opt = optim.Adam(self.actor.parameters(), lr=act_lr)
        self.v_opt = optim.Adam(self.state_critic.parameters(), lr=state_lr)
        self.q_opt = optim.Adam(self.state_action_critic.parameters(), lr=state_act_lr)

        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            # learn log_alpha to keep alpha > 0
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=act_lr)
            # target entropy for Pendulum
            self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        else:
            self.alpha = alpha

    def get_alpha(self):
        if self.auto_alpha:
            return self.log_alpha.exp()
        else:
            return self.alpha
        
    def soft_policy_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.state_critic.parameters(), self.target_critic.parameters()):
                p_targ.mul_(1 - self.tau)
                p_targ.add_(self.tau * p)

    
    def select_action(self, obs):
        action, _ , _= self.actor.sample(obs)

        return action
    

    def update_v(self, obs):
        state_val = self.state_critic(obs)
        a, log_pi, _  = self.actor.sample(obs)
        
        log_pi = log_pi.detach()
        
        state_act_val = self.state_action_critic(obs,a)

        with torch.no_grad():
            target = (state_act_val - log_pi * self.alpha).detach()
        
        return 0.5 * (state_val - target).pow(2).mean()
       
    def update_q(self, obs, actions, rewards, next_obs, masks):
        state_act_val = self.state_action_critic(obs, actions)
        
        with torch.no_grad():
            next_state_val = self.target_critic(next_obs)
            state_act_hat = (rewards * self.alpha + self.gamma * next_state_val * masks).detach()

        return 0.5 * (state_act_val - state_act_hat).pow(2).mean()


    def update_policy(self, obs):
        a, log_pi, _ = self.actor.sample(obs)
        q_val = self.state_action_critic(obs, a)
        
        alpha = self.get_alpha()
        
        # Actor loss
        policy_loss = (alpha * log_pi - q_val).mean()

        if self.auto_alpha:
            # Alpha loss to match target entropy
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        return policy_loss

