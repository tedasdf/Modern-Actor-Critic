import torch

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

