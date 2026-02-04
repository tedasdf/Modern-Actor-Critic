
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

from A2C.agent import A2Cagent
from helper.base import compute_boostrapping_multi_envs

NUM_ENVS = 4  # number of parallel environments

def train(envs, epoch_num):

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]


    agent = A2Cagent(obs_dim, act_dim, hidden_dim=256, gamma=0.99, lr=3e-4)
    
    obs, infos = envs.reset()  # returns a batch of observations

    
    for _ in range(epoch_num):
        
        ep_rewards = np.zeros(NUM_ENVS) 
        rollout = {
            "obs": [],        
            "actions": [],    
            "rewards": [],
            "log_probs": [],  
            "masks": [],      
            "next_obs": []
        }
        
        for step in range(2):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            actions, log_prob, entropy = agent.select_action(obs_tensor)
          
            next_obs, rew, term, trunc, info = envs.step(actions.detach().numpy())      
            # terminated/truncated is per env
            dones = term | trunc
            
            rollout["obs"].append(obs_tensor)
            rollout["actions"].append(actions)
            rollout["rewards"].append(torch.tensor(rew, dtype=torch.float32))
            rollout["log_probs"].append(log_prob)
            rollout["masks"].append(torch.tensor(1 - dones, dtype=torch.float32))
            rollout["next_obs"].append(torch.tensor(next_obs, dtype=torch.float32))

            obs = next_obs
            ep_rewards += rew
            if dones.any():
                reset_obs, _ = envs.reset(seed=42) # Or specific indices if supported
                obs[dones] = reset_obs[dones]

        states = torch.stack(rollout["obs"])        # (NUM_STEPS, NUM_ENVS, obs_dim)
        actions = torch.stack(rollout["actions"])   # (NUM_STEPS, NUM_ENVS, act_dim)
    
        targets = compute_boostrapping_multi_envs(agent, rollout)
        
        log_probs = torch.stack(rollout["log_probs"]).view(-1)

        actor_loss, critic_loss = agent.compute_losses(states,targets,log_probs)
        agent.update(actor_loss, critic_loss)
        print(
            f"[Loss] "
            f"actor: {actor_loss:.4f} | "
            f"critic: {critic_loss:.4f}"
        )



def make_single_env():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode="human" shows the game
    obs_shape = env.observation_space.shape
    # Image-based envs: (H, W, C)
    if obs_shape is not None and len(obs_shape) == 3:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=3)

    return env


if __name__ == "__main__":
    env_id = "Pendulum-v1"
    
    envs = gym.vector.SyncVectorEnv([lambda: make_single_env() for _ in range(4)])


    train(envs,10)