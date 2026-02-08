
import torch.optim as optim
import torch.nn.functional as F
from helper.neuralnet import GuassianActor, Critic
import torch


class A2Cagent():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, lr, actor_weight, critic_weight, entropy_weight):
        self.actor = GuassianActor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight


    def select_action(self, obs):
        action, log_prob, entropy = self.actor.sample(obs)
        
        return action, log_prob, entropy
    
    def compute_losses(self, states, targets, log_probs):
        # Flatten batch
        batch_size = states.shape[0] * states.shape[1]  # NUM_STEPS * NUM_ENVS
        states = states.view(batch_size, -1)
        
        log_probs = log_probs.view(-1)
        targets = targets.view(-1)

        # Critic loss
        values = self.critic(states).squeeze(-1)
        advantages = targets - values.detach()
  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss                                                                                       
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, targets)

        return actor_loss, critic_loss

    

    def update(self, actor_loss, critic_loss, entropy):

        total_loss = (
            self.actor_weight * actor_loss
            + self.critic_weight * critic_loss
            - self.entropy_weight * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
        )
        self.optimizer.step()


        return actor_loss.item(), critic_loss.item()

