
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import gymnasium as gym
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

from helper.base import compute_bootstrapping

def train(env, epoch_num, num_step):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    cfg = OmegaConf.load('AC\config.yaml')
    
    agent = instantiate(
        cfg.agent_continuous,  # or cfg.agent_discrete
        obs_dim=int(obs_dim),
        act_dim=int(act_dim)
    )
    obs, info = env.reset()
    for epoch in range(epoch_num):
       
        ep_reward = 0
        rollout = []
        
        for _ in range(num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, entropy = agent.select_action(obs_tensor)

            action_np = action.detach().cpu().numpy().flatten()
           
            next_obs, rew, term, trunc, info = env.step(action_np)
            done = term or trunc
            
            rollout.append((
                obs_tensor.squeeze(0),
                action,
                torch.tensor(rew, dtype=torch.float32),
                torch.tensor(next_obs, dtype=torch.float32),
                torch.tensor(1 - done, dtype=torch.float32),
                log_prob
            ))

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()

        states, actions, rewards, next_obs, masks, log_probs = zip(*rollout)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_obs = torch.stack(next_obs)
        masks = torch.stack(masks)
        log_probs = torch.stack(log_probs)

        targets = compute_bootstrapping(agent, next_obs, masks, rewards)

        actor_loss, critic_loss = agent.compute_losses(states, targets, log_probs)
        actor_loss, critic_loss = agent.update(actor_loss, critic_loss)
        print(
            f"[Loss] "
            f"actor: {actor_loss:.4f} | "
            f"critic: {critic_loss:.4f}"
        )

if __name__ == "__main__":

    cfg = OmegaConf.load('A3C\config.yaml')


    EPOCH_NUM = 10
    NUM_STEP = 100
    

    print(f"\n--- Starting Environment: {cfg.env_id} ---")
    
    env = gym.make(
        cfg.env_id, 
        render_mode="human",
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
