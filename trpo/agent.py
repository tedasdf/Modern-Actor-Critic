
import torch

import torch.optim as optim
from helper.rollout import Rollout
from TRPO.utility import conjugate_gradient, flat_grad, get_flat_params, line_search, set_flat_params
import torch.nn.functional as F
from helper.neuralnet import GaussianActor, Critic



class TRPOagent():
    def __init__(self, obs_dim, act_dim, hidden_size, gamma, lam, lr, damping , max_kl, n_cg_steps):
        self.damping = damping
        self.max_kl = max_kl
        self.gamma = gamma
        self.lam = lam
        self.rollout = Rollout()

        self.actor = GaussianActor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)

        self.critic_op = optim.Adam(self.critic.parameters(), lr=lr)

        self.n_cg_steps = n_cg_steps

    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        ratios = torch.exp(log_probs - old_log_probs)
        return -(ratios * advantages).mean() # {Equation 13} --- E_{s_t, a_t} ~ pi_old [ pi_new(a_t|s_t) / pi_old(a_t|s_t) * A_old(s_t, a_t) ]

    def compute_kl(self, states, old_mu, old_std):
        new_mu, new_std = self.actor(states)
        old_dist = torch.distributions.Normal(old_mu, old_std)
        new_dist = torch.distributions.Normal(new_mu, new_std)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)
        return kl.mean()
        
    def fisher_vector_product(self, v, states, old_mu, old_std):
        kl = self.compute_kl(states, old_mu, old_std)
        kl_grads = flat_grad(kl, self.actor, retain_graph=True, create_graph=True) 
        kl_v = (kl_grads * v).sum()
        fvp = flat_grad(kl_v, self.actor, retain_graph=True).detach() # ∇θ (∇θ KL ⋅ v)
        return fvp + self.damping * v # Ap, Curvature applied to p # damping * v ~~ Tikhonov regularisation
    
    def select_action(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        return action, log_prob

    def actor_update(self, states, actions, advantages, old_log_probs):
        old_loss = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        g = flat_grad(old_loss, self.actor).detach()

        prev_params = get_flat_params(self.actor)
        old_mu, old_std = self.actor(states)
        step_dir = conjugate_gradient(lambda v: self.fisher_vector_product(v, states, old_mu, old_std), g, n_steps=self.n_cg_steps)
        
        shs = 0.5 * torch.dot(step_dir, self.fisher_vector_product(step_dir, states, old_mu, old_std))
        step_size = torch.sqrt(self.max_kl / shs)
        full_step = step_size * step_dir                    
        expected_improve = torch.dot(g, full_step)

        # Line search
        def loss_fn():
            return self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        
        success, new_params = line_search(self.actor, loss_fn, prev_params, full_step, expected_improve)
        set_flat_params(self.actor, new_params)

        new_loss = loss_fn().item()
        actual_improve = old_loss - new_loss

        # Return values for logging
        return actual_improve, expected_improve.item(), success
    
    def critic_update(self, states, targets, l2_reg=1e-3):
        values = self.critic(states).squeeze(-1)
        loss = F.mse_loss(values, targets)
        for p in self.critic.parameters():
            loss += l2_reg * p.pow(2).sum()
        self.critic_op.zero_grad()
        loss.backward()
        self.critic_op.step()
        return loss.item()
