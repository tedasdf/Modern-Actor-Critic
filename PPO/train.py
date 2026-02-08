

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
        config=cfg
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
    for _ in range(num_epoch):
        
        ep_reward = 0
        for _ in range(num_step):
            # use the observation to seleect act
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = agent.select_action(obs_tensor)

            # env step
            next_obs, rew, term, trunc, info = env.step(action.item())
            done = term or trunc
            agent.rollout.store((obs, action, rew, 1 - done, log_prob))

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()

            # store in rollout with the r
        states, actions, old_log_probs, rewards, masks = agent.rollout.retrieve()
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
            
            # Update actor
            actor_loss = agent.actor_update(mb_states, mb_actions, mb_advantages, mb_old_log_probs)

            # Update critic
            critic_loss = agent.critic_update(mb_states, mb_targets)
            wandb.log({
                "train/actor_loss": actor_loss,
                "train/critic_loss": critic_loss,
                "env/episode_reward": ep_reward
            })

            
        agent.rollout.clear()
    
    env.close()

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--act_lr", type=float)
    parser.add_argument("--critic_lr", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Override config values if passed
    if args.hidden_dim: cfg.agent_continuous.hidden_dim = args.hidden_dim
    if args.act_lr: cfg.agent_continuous.act_lr = args.act_lr
    if args.critic_lr: cfg.agent_continuous.critic_lr = args.critic_lr
    if args.seed: cfg.seed = args.seed


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
         
        
