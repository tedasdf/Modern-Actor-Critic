import gymnasium as gym
import torch.optim as optim
import torch
import torch.multiprocessing as mp

from A2C.agent import GuassianActor, Critic
from omegaconf import OmegaConf
from hydra.utils import instantiate

obs_dim = 3       # Pendulum obs space
act_dim = 1       # Pendulum action space
hidden_dim = 256

# Global networks
global_actor = GuassianActor(obs_dim, act_dim, hidden_dim)
global_critic = Critic(obs_dim, hidden_dim)

# Share memory across processes
global_actor.share_memory()
global_critic.share_memory()

def worker_fn(worker_id, global_actor, global_critic, optimizer, gamma=0.99, rollout_length=20):
    env = gym.make("Pendulum-v1")
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    cfg = OmegaConf.load('A3C\config.yaml')
    
    agent = instantiate(
        cfg.agent_continuous,
        obs_dim=obs_dim, 
        act_dim=act_dim,
    )

    # Copy global weights to local network
    agent.actor.load_state_dict(global_actor.state_dict())
    agent.critic.load_state_dict(global_critic.state_dict())
    
    ep_reward = 0
    EPISODE_NUM = 1000
    ep_counter  = 0

    while ep_counter  < EPISODE_NUM:

        rollout = {"obs": [], "actions": [], "rewards": [], "log_probs": [], "masks": []}
        
        for _ in range(rollout_length):
            action, log_prob, _ = agent.actor.sample(obs.unsqueeze(0))
            action_np = action.squeeze(0).detach().numpy()
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward

            # store rollout
            rollout["obs"].append(obs)
            rollout["actions"].append(action)
            rollout["rewards"].append(torch.tensor([reward], dtype=torch.float32))
            rollout["log_probs"].append(log_prob)
            rollout["masks"].append(torch.tensor([0.0] if done else [1.0], dtype=torch.float32))

            obs = torch.tensor(next_obs, dtype=torch.float32)
            
            if done:
                ep_counter += 1
                print(f"[Worker {worker_id}] Episode finished. Reward: {ep_reward:.2f}, episode: {ep_counter}")
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
                ep_reward = 0

        targets = agent.boostrapping_target(obs, rollout["rewards"], rollout["masks"])
        
        # Compute local losses
        actor_loss, critic_loss = agent.compute_losses(rollout, targets)
        total_loss = actor_loss + 0.5 * critic_loss
        print(f"[Worker {worker_id}] Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")
        
        # Zero grads and compute gradients on local network
        optimizer.zero_grad()
        total_loss.backward()
        
        # Copy gradients to global network
        for local_param, global_param in zip(agent.actor.parameters(), global_actor.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad.clone()
        
        for local_param, global_param in zip(agent.critic.parameters(), global_critic.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad.clone()
        
        optimizer.step()
        
        # Sync local networks with updated global networks
        agent.actor.load_state_dict(global_actor.state_dict())
        agent.critic.load_state_dict(global_critic.state_dict())


if __name__ == "__main__":
    mp.set_start_method("spawn")  # needed for Windows
    
    optimizer = optim.Adam(list(global_actor.parameters()) + list(global_critic.parameters()), lr=3e-4)
    
    num_workers = 4
    processes = []
    
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_fn, args=(worker_id, global_actor, global_critic, optimizer))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
