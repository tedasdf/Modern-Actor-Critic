import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import torch

from helper.base import GAE_compute
from hydra.utils import instantiate
from omegaconf import OmegaConf

def train(env, num_epoch, num_steps):

    cfg = OmegaConf.load('TRPO\config.yaml')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    agent = instantiate(
        cfg.agent_continuous,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim)
    )

    obs, info = env.reset()
    for epoch in range(num_epoch):
    
        ep_reward = 0
    
        for _ in range(num_steps):
            # use the observation to seleect act
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = agent.select_action(obs_tensor)

            action = action.detach().cpu().numpy().flatten()
            # env step
            next_obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            agent.rollout.store(obs, action, rew, 1 - done, log_prob)

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()

            # store in rollout with the r
        states, actions, old_log_probs, rewards, masks = agent.rollout.retrieve()
        targets, advantages = GAE_compute(agent, states, rewards, masks)
        
        # Update actor
        agent.actor_update(states, actions, advantages, old_log_probs)

        # Update critic
        critic_loss = agent.critic_update(states, targets)

        agent.rollout.clear()
        print(f"Epoch {epoch+1}/{num_epoch} | Reward: {ep_reward:.2f} | "
                  f"Critic Loss: {critic_loss:.4f} =")

    env.close()

if __name__ == "__main__":
    envs_to_test = [
        {"id": "Pendulum-v1", "kwargs": {}},
        {"id": "LunarLander-v3", "kwargs": {"continuous": True}}
    ]

    EPOCH_NUM = 1000
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