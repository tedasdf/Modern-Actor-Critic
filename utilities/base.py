
import torch


def GAE_compute(agent, states, rewards, masks):
    with torch.no_grad():
        values = agent.critic(states).detach().squeeze(-1)

    advantages = torch.zeros_like(states)
    gae = 0 
    next_value = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + agent.gamma*next_value*masks[t] - values[t]
        
        gae = (agent.gamma*agent.lam)*gae + delta
        
        advantages[t] = gae
        next_value = values[t]
    
    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
    # TODO : Ablation without normalise
    # remove the need to use a baseline for the Q values
    return targets, advantages


def boostrapping(agent, rewards, next_values, masks):

    value = torch.zeros_like(rewards)
    delta = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + agent.gamma*next_values[t]*masks[t]

        value[t] = delta


    