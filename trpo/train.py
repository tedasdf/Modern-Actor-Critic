import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import torch

from helper.base import GAE_compute
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

def train(env, cfg, num_epoch, num_steps):

    wandb.init(
        project="TRPO_experiment",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

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
        states, actions, old_log_probs, rewards, masks, _ , _ , _ = agent.rollout.retrieve()
        targets, advantages = GAE_compute(agent, states, rewards, masks)
        
        # Update actor
        actual_improve, expected_improve, success = agent.actor_update(states, actions, advantages, old_log_probs)

        # Update critic
        for _ in range(cfg.vf_iter):
            vf_loss = agent.critic_update(states, targets)

        agent.rollout.clear()
        wandb.log({
            "epoch": epoch + 1,
            "reward": ep_reward,
            "actor_actual_improve": actual_improve,
            "actor_expected_improve": expected_improve,
            "actor_step_success": success,
            "VF_loss": vf_loss
        })
        print(f"Epoch {epoch+1}/{num_epoch} | Reward: {ep_reward:.2f} | "
                  f"Critic Loss: {vf_loss:.4f} =")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    cfg = OmegaConf.load('TRPO\config.yaml')

    EPOCH_NUM = cfg.num_epoch
    NUM_STEP = cfg.num_step

    env_id = cfg.env_id
    import argparse

    parser = argparse.ArgumentParser()
    # Policy / network
    parser.add_argument("--hidden_size", type=int)

    # TRPO trust region
    parser.add_argument("--max_kl", type=float)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--cg_iters", type=int)

    # Optimization / rollout
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_step", type=int)
    parser.add_argument("--vf_lr", type=float)
    parser.add_argument("--vf_iters", type=int)

    # Seed
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    # Override config values if arguments are provided
    if args.hidden_size is not None:
        cfg.agent_continuous.hidden_size = args.hidden_size
    if args.max_kl is not None:
        cfg.agent_continuous.max_kl = args.max_kl
    if args.damping is not None:
        cfg.agent_continuous.damping = args.damping
    if args.cg_iters is not None:
        cfg.agent_continuous.cg_iters = args.cg_iters
    if args.lr is not None:
        cfg.agent_continuous.lr = args.lr
    if args.num_step is not None:
        cfg.num_step = args.num_step          # rollout length
    if args.vf_lr is not None:
        cfg.agent_continuous.vf_lr = args.vf_lr
    if args.vf_iters is not None:
        cfg.vf_iter = args.vf_iters           # number of value function updates
    if args.seed is not None:
        cfg.seed = args.seed


   
    print(f"\n--- Starting Environment: {env_id} ---")
    
    env = gym.make(
        env_id, 
        render_mode="human"
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

    train(env, cfg, EPOCH_NUM, NUM_STEP)