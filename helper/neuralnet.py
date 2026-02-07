import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class GuassianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs):
        x = self.net(obs)
        mu = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        # log_std = torch.log(std)
        # epsilon = torch.randn_like(mu)       # ε ~ N(0, I)
        # a = mu + std * epsilon               # reparameterization
        # log_prob = (-0.5 * ((epsilon)**2 + 2*log_std + np.log(2*np.pi))).sum(dim=-1)
        return action, log_prob, entropy

    def entropy(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        return dist.entropy().sum(dim=-1) 


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self,x):
        logits = self.actor(x)
        return logits

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        value = self.network(x)
        return value


class StateValCritc(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt 
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x 
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class DDPGAct(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, final_layer_bound):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Tanh()
        )

        self.init_network(final_layer_bound)

    def init_network(self, final_layer_bound):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size(1)
                bound = 1. / np.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.uniform_(layer.bias, -bound, bound)

        self.apply(init_layer)
        nn.init.uniform_(self.action_net[-2].weight, -final_layer_bound, final_layer_bound)
        nn.init.uniform_(self.action_net[-2].bias, -final_layer_bound, final_layer_bound)
            

    def forward(self, state):
        return self.action_net(state)
    
class DDPGCrit(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, final_layer_bound):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), #fc1
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(n_actions, hidden_dim),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.init_network(final_layer_bound)
        

    def init_network(self, final_layer_bound):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size(1)
                bound = 1. / np.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.uniform_(layer.bias, -bound, bound)

        self.apply(init_layer)
        nn.init.uniform_(self.output_net[-1].weight, -final_layer_bound, final_layer_bound)
        nn.init.uniform_(self.output_net[-1].bias, -final_layer_bound, final_layer_bound)
            
    def forward(self, state, action):
        state_value = self.state_net(state)
        action_value = self.action_net(action)

        state_action_value = self.output_net(torch.add(state_value, action_value))
        return state_action_value


class DeepDetNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, sigma=0.15, theta=0.2):
        super().__init__()

        self.actor = DDPGAct(state_dim, hidden_dim[0], hidden_dim[1], action_dim, 0.003)
        self.critic = DDPGCrit(state_dim, hidden_dim[0], hidden_dim[1], action_dim, 0.003)
        self.noise = OUActionNoise(mu=np.zeros(action_dim), sigma=sigma, theta=theta, dt=1e-2)

    def critic_forward(self, state, action):
        return self.critic(state, action)

    def forward(self, state, use_noise=True):
        action = self.actor(state)
    
        if use_noise:
            action = torch.add(action,torch.tensor(self.noise(), dtype=action.dtype, device=action.device))
    
        value = self.critic(state, action)
        return action, value
