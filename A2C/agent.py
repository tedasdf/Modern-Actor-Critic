
import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import GuassianActor, Critic



class A2Cagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, lr):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma

    def select_action(self, obs):
        action, log_prob, entropy = self.actor.sample(obs)
        
        return action, log_prob, entropy
    
    def compute_losses(self, states, targets, log_probs):
        # Flatten batch
        batch_size = states.shape[0] * states.shape[1]  # NUM_STEPS * NUM_ENVS
        states = states.view(batch_size, -1)
        
        log_probs = log_probs.view(-1)
        targets = targets.view(-1)
        targets = (targets - targets.mean()) / (targets.std() + 1e-8)

        # Critic loss
        values = self.critic(states).squeeze(-1)
        advantages = targets - values.detach()
  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, targets, reduction='mean')

        return actor_loss, critic_loss

    

    def update(self, actor_loss, critic_loss , actor_weight=1.0, critic_weight=0.5):

        total_loss = actor_weight * actor_loss + critic_weight * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

