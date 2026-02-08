import torch

class Rollout():
    def __init__(self, use_entropy=False, use_mu=False, use_std=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []

        self.use_entropy = use_entropy
        self.use_mu = use_mu
        self.use_std = use_std

        if use_entropy:
            self.entropys = []
        if use_mu:
            self.mu = []
        if use_std:
            self.std = []

    def store(self, state, action, reward, mask, log_prob, entropy=None, mu=None, std=None):
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))  # float for continuous
        self.masks.append(torch.tensor(mask, dtype=torch.float32))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))

        if self.use_entropy and entropy is not None:
            self.entropys.append(torch.tensor(entropy, dtype=torch.float32))
        if self.use_mu and mu is not None:
            self.mu.append(torch.tensor(mu, dtype=torch.float32))
        if self.use_std and std is not None:
            self.std.append(torch.tensor(std, dtype=torch.float32))

    def retrieve(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        rewards = torch.stack(self.rewards)
        masks = torch.stack(self.masks)

        entropy = torch.stack(self.entropys) if self.use_entropy else None
        mu = torch.stack(self.mu) if self.use_mu else None
        std = torch.stack(self.std) if self.use_std else None

        return states, actions, old_log_probs, rewards, masks, mu, std, entropy

    def clear(self):
        self.states, self.actions, self.rewards, self.masks, self.log_probs = [], [], [], [], []
        if self.use_entropy: self.entropys = []
        if self.use_mu: self.mu = []
        if self.use_std: self.std = []
