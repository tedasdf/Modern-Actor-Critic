import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import GuassianActor, Critic

class ACagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, lr):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        self.gamma = gamma

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

    def select_action(self, obs):
        action, log_prob, entropy = self.actor.sample(obs)
        return action, log_prob, entropy

    def compute_losses(self, states, targets, log_probs):
        # Critic loss
        values = self.critic(states).squeeze(-1)
        
        advantages = targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
       
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, targets)
        
        return actor_loss, critic_loss

    def update(self, actor_loss, critic_loss , actor_weight=1.0, critic_weight=0.5):

        total_loss = actor_weight * actor_loss + critic_weight * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
