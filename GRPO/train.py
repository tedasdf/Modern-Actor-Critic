
import torch, wandb
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, RecordEpisodeStatistics
from omegaconf import OmegaConf
from hydra.utils import instantiate
from helper.base import GAE_compute

NUM_ENVS = 4

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_single_env():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs_shape = env.observation_space.shape
    if obs_shape is not None and len(obs_shape) == 3:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=3)
    env = RecordEpisodeStatistics(env)
    return env

def train(envs, cfg):
    set_seed(cfg.seed)

    wandb.init(
        project="RL_experiment",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    obs, infos = envs.reset()  # [num_envs, obs_dim]
    obs_dim = obs.shape[1]
    act_dim = envs.single_action_space.shape[0]

    agent = instantiate(cfg.agent_continuous, obs_dim=obs_dim, act_dim=act_dim)

    minibatch_size = cfg.minibatch_size

    ep_rewards = np.zeros(envs.num_envs)

    for epoch in range(cfg.num_epoch):
        for step in range(int(cfg.num_step)):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, mu, std = agent.select_action(obs_tensor)

            # scale actions to env range
            env_action = envs.single_action_space.low + \
                         (action.detach().numpy() + 1) * 0.5 * \
                         (envs.single_action_space.high - envs.single_action_space.low)

            next_obs, reward, terminated, truncated, info = envs.step(env_action)
            done = np.logical_or(terminated, truncated)

            ep_rewards += reward

            # Store rollout
            masks = 1.0 - done.astype(float)
            agent.rollout.store(obs, action.detach(), reward, masks, log_prob.detach(), mu.detach(), std.detach())

            obs = next_obs

            # Reset episodes that finished
            if done.any():
                for i, d in enumerate(done):
                    if d:
                        ep_rewards[i] = 0.0

        # Retrieve rollout and compute advantages
        states, actions, log_probs_old, rewards, masks, mu_old, std_old, _ = agent.rollout.retrieve()
        targets, advantages = GAE_compute(agent, states, rewards, masks)

        batch_size = states.size(0)
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_advantages = advantages[mb_idx]

            # Update actor (GRPO)
            actor_loss = agent.actor_update(mb_states, mb_actions, mb_advantages)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # sweep parameters
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None) 
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=float, default=None)
    parser.add_argument("--num_envs", type=float, default=None)
    parser.add_argument("--entropy_weight", type=float, default=None)
    

    args = parser.parse_args()

    env_id = "Pendulum-v1"
    
    envs = gym.vector.SyncVectorEnv([lambda: make_single_env() for _ in range(4)])
    cfg = OmegaConf.load('A2C\config.yaml')

    if args.actor_lr is not None:
        cfg.agent_continuous.actor_lr = args.actor_lr
    if args.critic_lr is not None:
        cfg.agent_continuous.critic_lr = args.critic_lr
    if args.n_steps is not None:
        cfg.num_step = args.n_steps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.entropy_weight is not None:
        cfg.agent_continuous.entropy_weight = args.entropy_weight
    if args.seed is not None:
        cfg.seed = args.seed
        
    epoch_num = cfg.num_epoch
    step_num = int(cfg.num_step)
    print(step_num)


    train(envs, cfg, epoch_num, step_num)