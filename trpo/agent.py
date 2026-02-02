
import torch

import torch.optim as optim
from trpo.utility import conjugate_gradient, flat_grad, get_flat_params, line_search, set_flat_params
import torch.nn.functional as F
from utilities.neuralnet import Actor, Critic

######### UTILITES ###############


class Rollout():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []

    def store(self, state, action, reward, mask, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.masks.append(mask)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def retrieve(self):
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs).detach()
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        masks = torch.tensor(self.masks, dtype=torch.float32)

        return states, actions, old_log_probs, rewards, masks

    def clear(self):
        self.states, self.actions, self.old_log_probs, self.rewards, self.masks = [], [], [], [], []



class TRPOagent():
    def __init__(self, obs_dim, act_dim, hidden_size, lr, damping , max_kl):
        self.damping = damping
        self.max_kl = max_kl
        self.rollout = Rollout()

        self.actor = Actor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, act_dim, hidden_size)

        self.critic_op = optim.Adam(self.critic.parameters, lr)

    def compute_surrogate_loss(self, states, actions, advantages, old_log_probs):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        return -(ratios * advantages).mean() # {Equation 13} --- E_{s_t, a_t} ~ pi_old [ pi_new(a_t|s_t) / pi_old(a_t|s_t) * A_old(s_t, a_t) ]

    def compute_kl(self, states, old_logits):
        new_logits = self.network.actor(states)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        kl = torch.distributions.kl_divergence(old_dist, new_dist)
        return kl.mean()
    
    def fisher_vector_product(self, v, states, old_logits):
        kl = self.compute_kl(states, old_logits)
        kl_grads = flat_grad(kl, self.actor, retain_graph=True) 
        kl_v = (kl_grads * v).sum()
        fvp = flat_grad(kl_v, self.actor).detach() # ∇θ (∇θ KL ⋅ v)
        return fvp + self.damping * v # Ap, Curvature applied to p # damping * v ~~ Tikhonov regularisation
    
    def select_action(self, obs):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def memory_update(self, transition_tuple):
        self.rollout.store(*transition_tuple)
    
    def actor_update(self, states, actions, advantages, old_log_probs, n_cg_steps=10):
        loss = self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        g = flat_grad(loss, self.actor).detach()

        prev_params = get_flat_params(self.actor)
        old_logits = self.actor(states)
        step_dir = conjugate_gradient(lambda v: self.fisher_vector_product(v, states, old_logits), g, n_steps=n_cg_steps)
        
        shs = 0.5 * torch.dot(step_dir, self.fisher_vector_product(step_dir, states, old_logits)) # a measure of how large the candidate step is in KL-space
        step_size = torch.sqrt(self.max_kl / shs)           # enforces the KL trust-region constraint
        full_step = step_size * step_dir                    
        expected_improve = torch.dot(g, full_step)          # first order estimate of how much reward we'll gain

        # Line search
        def loss_fn():
            return self.compute_surrogate_loss(states, actions, advantages, old_log_probs)
        
        success, new_params = line_search(self.network.actor, loss_fn, prev_params, full_step, expected_improve)
        set_flat_params(self.network.actor, new_params)
        return success
    
    def critic_update(self, states, targets, l2_reg=1e-3):
        values = self.critic(states).squeeze(-1)
        loss = F.mse_loss(values, targets)
        for p in self.critic.parameters():
            loss += l2_reg * p.pow(2).sum()
        self.critic_op.zero_grad()
        loss.backward()
        self.critic_op.step()
        return loss.item()
