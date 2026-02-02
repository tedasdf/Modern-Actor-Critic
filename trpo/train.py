import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import time, torch
import numpy as np
from trpo.agent import TRPOagent
from trpo.utility import GAE_compute

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

        observation, info = env.reset()
        


        for i in range(200):
            action = env.action_space.sample() 
            observation, reward, terminated, truncated, info = env.step(action)
            
            
            # --- SEE THE FRAME/STATE DATA ---
            if is_image:
                # If it were an image, we print the mean pixel intensity
                print(f"Step {i:3}: Image mean brightness: {np.mean(observation):.2f}", end="\r")
            else:
                # For Pendulum/Lander, we print the raw vector
                # np.array2string makes it pretty for the terminal
                obs_str = np.array2string(observation, precision=3, suppress_small=True)
                print(f"Step {i:3}: State {obs_str}", end="\r")

            if terminated or truncated:
                observation, info = env.reset()
            
            time.sleep(0.02) # Slow it down slightly to see the numbers


        agent = TRPOagent()

        for epoch in range(EPOCH_NUM):
            obs, info = env.reset()
            ep_reward = 0
            rollout = []
            for _ in range(NUM_STEP):
                # use the observation to seleect act
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, log_prob = agent.select_action(obs_tensor)

                # env step
                next_obs, rew, term, trunc, info = env.step(action.item())
                done = term or trunc
                agent.memory_update((obs, action, reward, 1 - done, log_prob))

                obs = next_obs
                ep_reward += reward
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
        
        env.close()