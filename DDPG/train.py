import argparse
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch, wandb
import numpy as np 

from helper.base import replay_to_tensor


def decay_noise(noise, step, cfg):
    if cfg.noise_decay == "None":
        return

    if cfg.noise_decay == "linear":
        frac = max(0.0, 1.0 - step / cfg.noise_decay_steps)
        noise.sigma = noise.initial_sigma * frac

    elif cfg.noise_decay == "exp":
        noise.sigma = noise.initial_sigma * np.exp(-step / cfg.noise_decay_steps)


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
            global_step = epoch * cfg.num_step + step

            decay_noise(agent.noise, global_step, cfg)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Select action
            action = agent.select_action(obs_tensor)
            action_np = action.detach().cpu().numpy().flatten()

            wandb.log({
                "exploration/action_std": action_np.std(),
                "exploration/noise_sigma": agent.noise.sigma,
            }, step=global_step)
                        
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            reward *= cfg.reward_scale
            ep_reward += reward

            # Print step info every 10 steps
            # if global_step % 1000 == 0:
            #     print(f"    Update step: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")

            # Store transition in replay buffer
            agent.replay_buffer.store(obs, action_np, reward, next_obs, done)
            obs = next_obs

            wandb.log({
                "replay/buffer_size": agent.replay_buffer.size,
            }, step=global_step)
                
            # Reset env if done
            if done:
                # Optional: log after each epoch
                wandb.log({
                    "env/episode_reward": ep_reward,
                    "env/episode_length": step + 1,
                }, step=global_step)

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
                }, step=global_step)

       
    print("[INFO] Training finished!")


def parse_args():
    parser = argparse.ArgumentParser()

    # ===== Optimization =====
    parser.add_argument("--actor_lr", type=float)
    parser.add_argument("--critic_lr", type=float)
    parser.add_argument("--batch_size", type=int)

    # ===== Network =====
    parser.add_argument("--hidden_dim", type=int)

    # ===== Bootstrapping variance =====
    parser.add_argument("--tau", type=float)
    parser.add_argument("--gamma", type=float)

    # ===== Exploration (OU noise) =====
    parser.add_argument("--ou_sigma", type=float)
    parser.add_argument("--ou_theta", type=float)
    parser.add_argument(
        "--noise_decay",
        type=str,
        choices=["None", "linear", "exp"]
    )

    # ===== Data regime =====
    parser.add_argument("--buffer_size", type=int)
    parser.add_argument("--learning_starts", type=int)
    parser.add_argument("--updates_per_step", type=int)

    # ===== Reward scaling =====
    parser.add_argument("--reward_scale", type=float)

    # ===== Misc =====
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def apply_override(arg, setter):
    if arg is not None:
        setter(arg)

if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load("DDPG/config.yaml")

    # ===== Optimization =====
    apply_override(args.actor_lr, lambda v: setattr(cfg.agent_continuous, "actor_lr", v))
    apply_override(args.critic_lr, lambda v: setattr(cfg.agent_continuous, "critic_lr", v))
    apply_override(args.batch_size, lambda v: setattr(cfg.agent_continuous, "batch_size", v))

    # ===== Network =====
    apply_override(args.hidden_dim, lambda v: setattr(cfg.agent_continuous, "hidden_dim", v))

    # ===== Bootstrapping variance =====
    apply_override(args.tau, lambda v: setattr(cfg.agent_continuous, "tau", v))
    apply_override(args.gamma, lambda v: setattr(cfg.agent_continuous, "gamma", v))

    # ===== Exploration =====
    apply_override(args.ou_sigma, lambda v: setattr(cfg.OUActionNoise, "sigma", v))
    apply_override(args.ou_theta, lambda v: setattr(cfg.OUActionNoise, "theta", v))
    apply_override(args.noise_decay, lambda v: setattr(cfg, "noise_decay", v))

    # ===== Data regime =====
    apply_override(args.buffer_size, lambda v: setattr(cfg.agent_continuous, "buffer_size", v))
    apply_override(args.learning_starts, lambda v: setattr(cfg, "learning_starts", v))
    apply_override(args.updates_per_step, lambda v: setattr(cfg, "updates_per_step", v))

    # ===== Reward scaling =====
    apply_override(args.reward_scale, lambda v: setattr(cfg, "reward_scale", v))

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
