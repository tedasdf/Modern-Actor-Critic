import gymnasium as gym
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import wandb
import argparse
from omegaconf import OmegaConf
from hydra.utils import instantiate
from A2C.agent import GuassianActor, Critic


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_fn(worker_id, global_actor, global_critic, optimizer, cfg):
    # Seed per worker
    set_seed(cfg.seed + worker_id)

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Local agent
    agent = instantiate(cfg.agent_continuous, obs_dim=obs_dim, act_dim=act_dim)
    agent.actor.load_state_dict(global_actor.state_dict())
    agent.critic.load_state_dict(global_critic.state_dict())

    wandb.init(
        project="RL_experiment",
        name=f"worker_{worker_id}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    ep_reward = 0
    ep_counter = 0
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    while ep_counter < cfg.num_epoch:
        rollout = {"obs": [], "actions": [], "rewards": [], "log_probs": [], "masks": []}

        # Collect n-step rollout
        for _ in range(cfg.num_steps):
            action, log_prob, _ = agent.actor.sample(obs.unsqueeze(0))
            action_np = np.clip(2.0 * action.squeeze(0).detach().numpy(),
                                env.action_space.low,
                                env.action_space.high)
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward

            rollout["obs"].append(obs)
            rollout["actions"].append(action)
            rollout["rewards"].append(torch.tensor([reward], dtype=torch.float32))
            rollout["log_probs"].append(log_prob)
            rollout["masks"].append(torch.tensor([0.0] if done else [1.0], dtype=torch.float32))

            obs = torch.tensor(next_obs, dtype=torch.float32)

            if done:
                ep_counter += 1
                print(f"[Worker {worker_id}] Episode {ep_counter} finished. Reward: {ep_reward:.2f}")
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
                ep_reward = 0

        # Compute bootstrapped targets
        targets = agent.boostrapping_target(obs, rollout["rewards"], rollout["masks"])

        # Compute local losses
        actor_loss, critic_loss = agent.compute_losses(rollout, targets)
        total_loss = actor_loss + 0.5 * critic_loss

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

        # Sync local weights with updated global weights
        agent.actor.load_state_dict(global_actor.state_dict())
        agent.critic.load_state_dict(global_critic.state_dict())

        wandb.log({
            "worker_id": worker_id,
            "train/actor_loss": actor_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "env/steps": ep_counter,
            "eval/return": ep_reward
        })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Required for Windows

    args = parse_args()
    cfg = OmegaConf.load("A3C/config.yaml")

    # Override config with command-line arguments
    if args.lr is not None:
        cfg.agent_continuous.lr = args.lr
    if args.hidden_dim is not None:
        cfg.agent_continuous.hidden_dim = args.hidden_dim
    seed = args.seed if args.seed is not None else cfg.seed

    set_seed(seed)

    # Create a dummy env to get dimensions
    dummy_env = gym.make("Pendulum-v1")
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    hidden_dim = cfg.agent_continuous.hidden_dim
    dummy_env.close()

    print(f"Observation dimension: {obs_dim}, Action dimension: {act_dim}")

    # Global networks
    global_actor = GuassianActor(obs_dim, act_dim, hidden_dim)
    global_critic = Critic(obs_dim, hidden_dim)

    global_actor.share_memory()
    global_critic.share_memory()

    optimizer = optim.Adam(
        list(global_actor.parameters()) + list(global_critic.parameters()),
        lr=cfg.agent_continuous.lr
    )

    # Launch workers
    num_workers = cfg.num_workers
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_fn, args=(worker_id, global_actor, global_critic, optimizer, cfg))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
