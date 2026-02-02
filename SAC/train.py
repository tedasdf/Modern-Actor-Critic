

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import time, torch, OmegaConf
from utilities.base import replay_to_tensor

from hydra.utils import instantiate
from SAC.agent import SACagent

def train(env, epoch_num, num_step):
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    cfg = OmegaConf.load('SAC\config.yaml')
    
    agent = instantiate(
        cfg.agent_continuous,  # or cfg.agent_discrete
        obs_dim=obs_dim,
        act_dim=act_dim
    )
    
    for epoch in range(epoch_num):
        obs, info = env.reset()
        ep_reward = 0
        
        for _ in range(num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.select_action(obs_tensor)

            next_obs, rew, term, trunc, info = env.step(action.item())
            done = term or trunc
            
            agent.replay_buffer.store(obs, action, rew, next_obs, 1- done)

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()
        
        if agent.replay_buffer.check_length():

            obs, actions,rewards, next_obs, masks = agent.replay_buffer.sample()
            obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, masks_tensor = replay_to_tensor(obs, actions, rewards, next_obs, masks)

            v_loss = agent.update_v(obs_tensor)
            q_loss = agent.update_q(obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, masks_tensor)
            policy_loss = agent.update_policy(obs_tensor)

            # manually step
            agent.v_opt.zero_grad()
            v_loss.backward()
            agent.v_opt.step()

            agent.q_opt.zero_grad()
            q_loss.backward()
            agent.q_opt.step()

            agent.act_opt.zero_grad()
            policy_loss.backward()
            agent.act_opt.step()


if __name__ == "__main__":
    envs_to_test = [
        {"id": "Pendulum-v1", "kwargs": {}},
        {"id": "LunarLander-v3", "kwargs": {"continuous": True}}
    ]

    EPOCH_NUM = 1000
    NUM_STEP = 1000
    
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

        train(env)
        observation, info = env.reset()

         
        