
import torch, wandb
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, RecordEpisodeStatistics
from omegaconf import OmegaConf
from A2C.agent import A2Cagent
from helper.base import compute_boostrapping_multi_envs
from hydra.utils import instantiate

NUM_ENVS = 4  # number of parallel environments

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(envs, cfg, epoch_num, step_num):
    set_seed(cfg.seed)

    wandb.init(
        project="RL_experiment",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]


    agent = instantiate(
        cfg.agent_continuous,
        obs_dim=obs_dim,
        act_dim=act_dim
    )
   
    obs, infos = envs.reset()  # returns a batch of observations

    
    for _ in range(epoch_num):
        
        ep_rewards = np.zeros(NUM_ENVS) 
        rollout = {
            "obs": [],        
            "actions": [],    
            "rewards": [],
            "log_probs": [],  
            "masks": [],      
            "next_obs": [],
            "entropies": []
        }
        
        for step in range(step_num):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            actions, log_prob, entropy = agent.select_action(obs_tensor)
            env_actions = 2.0 * actions.detach().numpy()
            next_obs, rew, term, trunc, info = envs.step(env_actions)      
            # terminated/truncated is per env
            
            dones = term | trunc
            
            rollout["obs"].append(obs_tensor)
            rollout["actions"].append(actions)
            rollout["rewards"].append(torch.tensor(rew, dtype=torch.float32))
            rollout["log_probs"].append(log_prob)
            rollout["masks"].append(torch.tensor(1 - dones, dtype=torch.float32))
            rollout["next_obs"].append(torch.tensor(next_obs, dtype=torch.float32))
            rollout["entropies"].append(entropy)

            obs = next_obs
            ep_rewards += rew
            
        states = torch.stack(rollout["obs"])        # (NUM_STEPS, NUM_ENVS, obs_dim)
        actions = torch.stack(rollout["actions"])   # (NUM_STEPS, NUM_ENVS, act_dim)
        entropy = torch.stack(rollout["entropies"]).mean()

        targets = compute_boostrapping_multi_envs(agent, rollout)
        
        log_probs = torch.stack(rollout["log_probs"])  # (T, N)

        actor_loss, critic_loss = agent.compute_losses(states,targets,log_probs)
        agent.update(actor_loss, critic_loss,entropy)
        print(
            f"[Loss] "
            f"actor: {actor_loss:.4f} | "
            f"critic: {critic_loss:.4f}"
        )


        wandb.log({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_return": ep_rewards.mean(),
        })



def make_single_env():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    obs_shape = env.observation_space.shape
    if obs_shape is not None and len(obs_shape) == 3:
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=3)

    env = RecordEpisodeStatistics(env)
    return env



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # sweep parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    env_id = "Pendulum-v1"
    
    envs = gym.vector.SyncVectorEnv([lambda: make_single_env() for _ in range(4)])
    cfg = OmegaConf.load('A2C\config.yaml')

    if args.lr is not None:
        cfg.agent_continuous.lr = args.lr
    if args.hidden_dim is not None:
        cfg.agent_continuous.hidden_dim = args.hidden_dim

    epoch_num = cfg.num_epoch
    step_num = cfg.num_step


    train(envs, cfg, epoch_num, step_num)