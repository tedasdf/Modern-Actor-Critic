import argparse
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch, wandb
import numpy as np 

from helper.base import replay_to_tensor

def train(env, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n

    # Instantiate agent with parameters from cfg
    agent = instantiate(
        cfg.agent_continuous,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        ou_cfg=cfg.OUActionNoise
    )

    # WandB logging
    wandb.init(
        project="RL_experiment",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )

    obs, info = env.reset()
    ep_reward = 0

    print(f"[INFO] Training started for {cfg.num_epoch} epochs, obs_dim={obs_dim}, act_dim={act_dim}")

    for epoch in range(cfg.num_epoch):
        print(f"\n[Epoch {epoch+1}/{cfg.num_epoch}]")
        for step in range(cfg.num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Select action
            action = agent.select_action(obs_tensor)
            action_np = action.detach().cpu().numpy().flatten()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            ep_reward += reward

            # Print step info every 10 steps
            if step % 10 == 0:
                print(f"  Step {step+1}/{cfg.num_step} | Reward: {reward:.2f} | Episode Reward: {ep_reward:.2f}")

            # Store transition in replay buffer
            agent.replay_buffer.store(obs, action_np, reward, next_obs, done)
            obs = next_obs

            # Reset env if done
            if done:
                print(f"  Episode finished. Total Reward: {ep_reward:.2f}")
                obs, info = env.reset()
                ep_reward = 0

            # Update agent if buffer is ready
            if agent.replay_buffer.check_length():
                s, a, r, ns, m = agent.replay_buffer.sample()
                s_t, a_t, r_t, ns_t, m_t = replay_to_tensor(s, a, r, ns, m)
                critic_loss, actor_loss = agent.update_network(s_t, a_t, r_t, ns_t, m_t)
                print(f"    Update step: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
                wandb.log({
                    "train/actor_loss": actor_loss,
                    "train/critic_loss": critic_loss,
                    "env/episode_reward": ep_reward,
                })

        # Optional: log after each epoch
        wandb.log({"epoch": epoch, "epoch_reward": ep_reward})

    print("[INFO] Training finished!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_actor", type=float)
    parser.add_argument("--lr_critic", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load("DDPG/config.yaml")

    if args.lr_actor: cfg.agent_continuous.actor_lr = args.lr_actor
    if args.lr_critic: cfg.agent_continuous.critic_lr = args.lr_critic
    if args.hidden_dim: cfg.agent_continuous.hidden_dim = args.hidden_dim
    if args.tau: cfg.agent_continuous.tau = args.tau
    if args.gamma: cfg.agent_continuous.gamma = args.gamma

    env_id = cfg.env_id
    env = gym.make(env_id)

    # Apply wrappers for image-based envs if needed
    obs_shape = env.observation_space.shape
    if obs_shape is not None and len(obs_shape) == 3:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)

    print(f"[INFO] Env '{env_id}' created. Observation shape: {env.observation_space.shape}, Action shape: {env.action_space.shape if hasattr(env.action_space,'shape') else env.action_space.n}")

    train(env, cfg, args.seed)
