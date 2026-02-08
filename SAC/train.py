

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import time, torch
from omegaconf import OmegaConf
from helper.base import replay_to_tensor
import numpy as np
from hydra.utils import instantiate
import wandb

def train(env, cfg, epoch_num, num_step):

    wandb.init(
        project="SAC_experiment",  # your project name
        config=OmegaConf.to_container(cfg, resolve=True)
    )
   
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    
    agent = instantiate(
        cfg.agent_continuous,  # or cfg.agent_discrete
        obs_dim=int(obs_dim),
        act_dim=int(act_dim)
    )
    obs, info = env.reset()
    for epoch in range(epoch_num):
       
        ep_reward = 0
        
        for _ in range(num_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.select_action(obs_tensor)
            
            action = action.detach().cpu().numpy().flatten()
 
            action = np.clip(action * 2.0, env.action_space.low, env.action_space.high)

            next_obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            
            agent.replay_buffer.store(obs, action, rew, next_obs, 1- done)

            obs = next_obs
            ep_reward += rew
            if done:
                obs, info = env.reset()
        
        if agent.replay_buffer.check_length():

            states, actions,rewards, next_states, masks = agent.replay_buffer.sample()
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, masks_tensor = replay_to_tensor(states, actions, rewards, next_states, masks)

            v_loss = agent.update_v(states_tensor)
            q_loss = agent.update_q(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, masks_tensor)
            policy_loss = agent.update_policy(states_tensor)

            # manually step
            agent.v_opt.zero_grad()
            agent.q_opt.zero_grad()
            agent.act_opt.zero_grad()

            q_loss.backward()
            v_loss.backward()
            policy_loss.backward()

            agent.q_opt.step()
            agent.v_opt.step()
            agent.act_opt.step()

            wandb.log({
                "epoch": epoch + 1,
                "reward": ep_reward,
                "V_loss": v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss,
                "Q_loss": q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss,
                "Policy_loss": policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
            })

            print(f"Epoch {epoch+1}/{epoch_num} | Reward: {ep_reward:.2f} | "
                  f"V Loss: {v_loss:.4f} | Q Loss: {q_loss:.4f} | Policy Loss: {policy_loss:.4f}")
    env.close()
    wandb.finish()  # mark run complete

if __name__ == "__main__":
   
    
    cfg = OmegaConf.load('SAC\config.yaml')
    EPOCH_NUM = cfg.num_epoch
    NUM_STEP = cfg.num_step
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--act_lr", type=float)
    parser.add_argument("--state_lr", type=float)
    parser.add_argument("--state_act_lr", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Override config.yaml values
    if args.hidden_dim: cfg.agent_continuous.hidden_dim = args.hidden_dim
    if args.act_lr: cfg.agent_continuous.act_lr = args.act_lr
    if args.state_lr: cfg.agent_continuous.state_lr = args.state_lr
    if args.state_act_lr: cfg.agent_continuous.state_act_lr = args.state_act_lr
    if args.alpha: cfg.agent_continuous.alpha = args.alpha
    if args.batch_size: cfg.agent_continuous.batch_size = args.batch_size
    if args.seed: cfg.seed = args.seed

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

    train(env,cfg,EPOCH_NUM, NUM_STEP)

         
        