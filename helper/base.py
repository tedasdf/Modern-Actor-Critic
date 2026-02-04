
import torch


def compute_boostrapping_multi_envs(agent, rollout):
    rewards = torch.stack(rollout["rewards"])
    masks = torch.stack(rollout["masks"])
    
    # Ensure this is detached so gradients don't flow through the target calculation
    with torch.no_grad():
        next_obs_tensor = rollout["next_obs"][-1] 
        next_value = agent.critic(next_obs_tensor).squeeze(-1)
    
    returns = []
    R = next_value
    for step in reversed(range(len(rewards))):
        # masks[step] is 0 if the episode ended, which correctly cuts off the bootstrap
        R = rewards[step] + agent.gamma * R * masks[step]    
        returns.insert(0, R.clone()) # clone to be safe

    return torch.stack(returns)

def compute_booostrapping_loop(agent, rollout):
    rewards = torch.stack(rollout["rewards"])
    masks = torch.stack(rollout["masks"])
    
    with torch.no_grad():
        next_obs_tensor = rollout["next_obs"][-1]
        next_action = agent.actor(next_obs_tensor) 
        next_value = agent.critic(next_obs_tensor, next_action).squeeze(-1)
    
    targets = []
    R = next_value
    for step in reversed(range(len(rewards))):
        R = rewards[step] + agent.gamma * R * masks[step]    
        targets.insert(0, R.clone()) # clone to be safe

    return torch.stack(targets)


def compute_bootstrapping(agent, next_states, masks, rewards):
    with torch.no_grad():
        # critic value of next states
        next_values = agent.critic(next_states)  # shape: [batch]
        targets = rewards + agent.gamma * next_values * masks
    return targets

def GAE_compute(agent, states, rewards, masks):
    with torch.no_grad():
        values = agent.critic(states).detach().squeeze(-1)

    advantages = torch.zeros(len(rewards), dtype=torch.float32)
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


