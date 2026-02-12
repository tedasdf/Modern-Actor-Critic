import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import numpy as np
import random
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from helper.base import GAE_compute
import wandb


def train(env, num_epoch, num_step, cfg):

    wandb.init(
        project="PPO_experiment",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    agent = instantiate(
        cfg.agent_continuous,  # or cfg.agent_discrete
        obs_dim=int(obs_dim),
        act_dim=int(act_dim)
    )

    minibatch_size = cfg.minibatch_size
    obs, info = env.reset()

    print(f"Starting training for {num_epoch} epochs, {num_step} steps per epoch")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    for epoch in range(num_epoch):
        ep_reward = 0

        for step in range(num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            action, log_prob, mu, std = agent.select_action(obs_tensor)

            # Convert to numpy and clip
            env_action = env.action_space.low + (action.detach().numpy() + 1) * 0.5 * (env.action_space.high - env.action_space.low)

            next_obs, rew, term, trunc, info = env.step(env_action)
            done = term or trunc
            ep_reward += rew

            agent.rollout.store(obs, action, rew, 1 - done, log_prob, mu=mu, std=std)
            obs = next_obs

            if done:
                # print(f"[Epoch {epoch} Step {step}] Episode done. Reward: {ep_reward:.2f}")
                obs, info = env.reset()
                ep_reward = 0

            # Debug print every 10 steps
            # if step % 10 == 0:
            #     print(f"[Epoch {epoch} Step {step}] Action: {action_np}, Reward: {rew:.2f}")

        # Compute targets and advantages
        states, actions, old_log_probs, rewards, masks, mu_old, std_old, _  = agent.rollout.retrieve()
        targets, advantages = GAE_compute(agent, states, rewards, masks)

        batch_size = states.size(0)
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_targets = targets[mb_idx]
            mb_old_mu = mu_old[mb_idx]
            mb_old_std = std_old[mb_idx]

            # Update actor and critic
            actor_loss = agent.actor_update(mb_states, mb_actions, mb_advantages, mb_old_log_probs, mb_old_mu, mb_old_std)
            critic_loss = agent.critic_update(mb_states, mb_targets)

            # print(f"[Epoch {epoch} Minibatch {start // minibatch_size}] "
            #       f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            wandb.log({
                "train/actor_loss": actor_loss,
                "train/critic_loss": critic_loss,
                "env/epoch_reward": ep_reward
            })

        agent.rollout.clear()
        print(f"Epoch {epoch} finished. Total reward this epoch: {ep_reward:.2f}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    import argparse

    cfg = OmegaConf.load("PPO/config.yaml")

    # ===== Add argparse for all sweepable parameters =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--act_lr", type=float)
    parser.add_argument("--critic_lr", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--kl_target", type=float)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--entropy_coef", type=float)
    parser.add_argument("--num_step", type=int)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--minibatch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ===== Apply arguments to config if provided =====
    if args.hidden_dim:        cfg.agent_continuous.hidden_dim = args.hidden_dim
    if args.act_lr:            cfg.agent_continuous.act_lr = args.act_lr
    if args.critic_lr:         cfg.agent_continuous.critic_lr = args.critic_lr
    if args.epsilon:           cfg.agent_continuous.epsilon = args.epsilon
    if args.kl_target:         cfg.agent_continuous.kl_target = args.kl_target
    if args.lam:               cfg.agent_continuous.lam = args.lam
    if args.gamma:             cfg.agent_continuous.gamma = args.gamma
    if args.entropy_coef:      cfg.agent_continuous.entropy_coef = args.entropy_coef
    if args.num_step:          cfg.num_step = args.num_step
    if args.num_epoch:         cfg.num_epoch = args.num_epoch
    if args.minibatch_size:    cfg.minibatch_size = args.minibatch_size
    if args.num_workers:       cfg.num_workers = args.num_workers
    if args.seed:              cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    EPOCH_NUM = cfg.num_epoch
    NUM_STEP = cfg.num_step

    env = gym.make(cfg.env_id)
    obs_shape = env.observation_space.shape
    is_image = obs_shape is not None and len(obs_shape) == 3

    if is_image:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)
        print(f"Applied image wrappers. Observation shape: {env.observation_space.shape}")
    else:
        print(f"Vector-based state. Observation shape: {obs_shape}")

    train(env, EPOCH_NUM, NUM_STEP, cfg)
