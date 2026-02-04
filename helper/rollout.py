import torch

class Rollout():
    def __init__(self, use_entropy=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.use_entropy = use_entropy
        if use_entropy:
            self.entropys = []

    def store(self, state, action, reward, mask, log_prob,entropy=None):
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))  # float for continuous
        self.masks.append(torch.tensor(mask, dtype=torch.float32))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
        if entropy is not None:
            self.entropys.append(torch.tensor(entropy, dtyp=torch.float32))
    
    def retrieve(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)           # FIX: no dtype long
        old_log_probs = torch.stack(self.log_probs).detach()
        rewards = torch.stack(self.rewards)
        masks = torch.stack(self.masks)
        if self.use_entropy:
            entropy = torch.stack(self.entropys)
            return states, actions, old_log_probs, rewards, masks, entropy
        return states, actions, old_log_probs, rewards, masks

    def clear(self):
        self.states, self.actions, self.rewards, self.masks, self.log_probs = [], [], [], [], []
        if self.use_entropy:
            self.entropys = []
