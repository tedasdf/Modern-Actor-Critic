import torch
from helper.neuralnet import GaussianActor
from helper.rollout import Rollout


class GRPOagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, beta=0.1, tau=0.005, entropy_coef=0.0, lr=3e-4):
        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.target_actor = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.rollout = Rollout()

        # Optimizer
        self.actor_op = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Hyperparameters
        self.beta = beta          # KL regularization coefficient
        self.tau = tau            # soft update factor
        self.entropy_coef = entropy_coef
        self.eps = 1e-6

        # Initialize target network
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

    def select_action(self, obs):
        mu, std = self.actor(obs)
        dist = torch.distributions.Normal(mu, std)

        z = dist.rsample()
        action = torch.tanh(z)

        # Log prob with tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.eps)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, mu, std

    def compute_log_probs(self, states, actions, actor):
        mu, std = actor(states)
        dist = torch.distributions.Normal(mu, std)

        pre_tanh_action = torch.atanh(actions.clamp(-1 + self.eps, 1 - self.eps))
        log_probs = dist.log_prob(pre_tanh_action) - torch.log(1 - actions.pow(2) + self.eps)
        
        return log_probs.sum(dim=-1), dist

    def actor_update(self, states, actions, advantages):
        # Detach advantages to prevent backprop through rollout
        advantages = advantages.detach()

        # Current policy log probs
        log_probs, dist = self.compute_log_probs(states, actions, self.actor)

        # Reference policy log probs (target actor)
        ref_log_probs, _ = self.compute_log_probs(states, actions, self.target_actor)

        # f-KL term
        log_ratio = (ref_log_probs - log_probs).clamp(-10, 10)
        kl_f = torch.exp(log_ratio) - log_ratio - 1

        # GRPO loss (maximize)
        loss = -(advantages * log_probs - self.beta * kl_f).mean()

        # Optional: entropy regularization
        if self.entropy_coef > 0:
            entropy = dist.entropy().sum(dim=-1).mean()
            loss = loss - self.entropy_coef * entropy

        # Update actor
        self.actor_op.zero_grad()
        loss.backward()
        self.actor_op.step()

        # Soft target update
        self.soft_update_target()

        return loss.item()

    def soft_update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
