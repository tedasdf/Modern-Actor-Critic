
import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import GaussianActor, Critic
import torch


class A2Cagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, actor_lr, critic_lr, actor_weight, critic_weight, entropy_weight):
        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        # Single optimizer for both actor and critic
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )


        self.gamma = gamma
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

    def select_action(self, obs):
        action, log_prob, entropy = self.actor.sample(obs)
        
        return action, log_prob, entropy
    
    def compute_losses(self, states, targets, log_probs):
        T, N = states.shape[:2]
        batch_size = T * N

        states = states.view(batch_size, -1)
        log_probs = log_probs.view(-1)
        targets = targets.view(-1)

        values = self.critic(states).squeeze(-1)

        # Advantages
        advantages = targets - values.detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std()

        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, targets.detach())

        # PG variance proxy (diagnostic only)
        pg_variance = (log_probs.detach() * advantages.detach()).std(unbiased=False)

        return (
            actor_loss,
            critic_loss,
            adv_mean.item(),
            adv_std.item(),
            pg_variance.item()
        )

    def update(self, actor_loss, critic_loss, entropy):

        # Actor update
        self.actor_optimizer.zero_grad()
        actor_total_loss = (
            self.actor_weight * actor_loss
            - self.entropy_weight * entropy
        )
        actor_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Critic update
        self.critic_optimizer.zero_grad()
        critic_total_loss = self.critic_weight * critic_loss
        critic_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

