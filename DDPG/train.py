
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

from helper.base import replay_to_tensor

def train(env, epoch_num, num_step):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    cfg = OmegaConf.load('DDPG\config.yaml')
    
    agent = instantiate(
        cfg.agent_continuous,  # or cfg.agent_discrete
        obs_dim=int(obs_dim),
        act_dim=int(act_dim)
    )
    obs, info = env.reset()
    print(obs.shape)
    
    for epoch in range(epoch_num):
       
        ep_reward = 0
        
        for _ in range(num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            action = agent.select_action(obs_tensor)
            action_np = action.detach().cpu().numpy().flatten()
           
            next_obs, rew, term, trunc, info = env.step(action_np)
           
            done = term or trunc
            agent.replay_buffer.store(obs, action_np, rew, next_obs, done)

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()

            if agent.replay_buffer.check_length():

                state, actions, rewards, next_state, masks = agent.replay_buffer.sample()
                state_tensor, actions_tensor, rewards_tensor, next_state_tensor, masks_tensor = replay_to_tensor(state, actions, rewards, next_state, masks)


                critic_loss, actor_loss = agent.update_network(state_tensor, actions_tensor, rewards_tensor, next_state_tensor, masks_tensor)
                
                print(
                    f"[Loss] "
                    f"actor: {actor_loss:.4f} | "
                    f"critic: {critic_loss:.4f}"
                )

if __name__ == "__main__":
    envs_to_test = [
        {"id": "Pendulum-v1", "kwargs": {}},
    ]

    EPOCH_NUM = 10
    NUM_STEP = 100
    
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
        
        
        train(env,EPOCH_NUM, NUM_STEP)
