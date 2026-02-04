

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import time

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PPO.agent import PPOagent
from helper.base import GAE_compute


def train(env, num_epoch, num_step):



    cfg = OmegaConf.load('PPO\config.yaml')
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
            agent.actor_update(mb_states, mb_actions, mb_advantages, mb_old_log_probs)

            # Update critic
            critic_loss = agent.critic_update(mb_states, mb_targets)
            
        agent.rollout.clear()
    
    env.close()

if __name__ == "__main__":
    envs_to_test = [
        {"id": "Pendulum-v1", "kwargs": {}},
        {"id": "LunarLander-v3", "kwargs": {"continuous": True}}
    ]

    EPOCH_NUM = 100
    NUM_STEP = 5

    for env_spec in envs_to_test:
        print(f"\n--- Starting Environment: {env_spec['id']} ---")
        
        env = gym.make(
            env_spec["id"], 
            render_mode="human", 
            **env_spec["kwargs"]
        )
        
        obs_shape = env.observation_space.shape
        is_image = obs_shape is not None and len(obs_shape) == 3

        if is_image:
            env = GrayscaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, (84, 84))
            env = FrameStackObservation(env, stack_size=4)
            print(f"Applied image wrappers. Observation shape: {env.observation_space.shape}")
        else:
            print(f"Vector-based state. Observation shape: {obs_shape}")

        
        train(env, EPOCH_NUM, NUM_STEP)
         
        
