
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



def replay_to_tensor(states, actions, rewards, next_states, dones, device=None):
    device = device or torch.device("cpu")
    
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Handle discrete vs continuous actions
    if len(actions.shape) == 1:
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    else:
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
