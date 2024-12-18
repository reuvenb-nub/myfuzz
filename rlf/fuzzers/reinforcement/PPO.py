import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import os

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print("Device set to: CPU")

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init**2).to(device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std**2).to(device)
        else:
            print("Warning: set_action_std() is called on a discrete policy.")

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag_embed(self.action_var.expand_as(action_mean))
            dist = MultivariateNormal(action_mean, cov_mat)
            action = action.reshape(-1, self.action_dim) if self.action_dim == 1 else action
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        self.action_std = action_std_init if has_continuous_action_space else None
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("Warning: set_action_std() is called on a discrete policy.")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.cpu().numpy() if self.has_continuous_action_space else action.item()

    def update(self):
        # Check if buffer is empty
        rewards = self.buffer.rewards
        if len(rewards) == 0:
            return
        
        # Compute discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Convert to tensor with matching first dimension
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        
        # Normalize rewards if more than one sample
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Collect buffer data
        states = torch.stack(self.buffer.states[:len(discounted_rewards)]).detach().to(device)
        actions = torch.stack(self.buffer.actions[:len(discounted_rewards)]).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs[:len(discounted_rewards)]).detach().to(device)
        
        # Ensure state values match rewards length
        state_values = torch.stack(self.buffer.state_values[:len(discounted_rewards)]).detach().to(device)

        # Compute advantages
        advantages = discounted_rewards - state_values.squeeze()

        # PPO update loop
        for _ in range(self.K_epochs):
            # Evaluate current policy
            logprobs, current_state_values, dist_entropy = self.policy.evaluate(states, actions)
            current_state_values = current_state_values.squeeze()

            # Compute ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(current_state_values, discounted_rewards) - 0.01 * dist_entropy

            # Gradient update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear the buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), f'{checkpoint_path}/ppo_eval_net.pth')

    def load(self, checkpoint_path):
        if 'drqn_eval_net.pth' in os.listdir(f'{checkpoint_path}'):
            self.policy_old.load_state_dict(torch.load(f'{checkpoint_path}/ppo_eval_net.pth', map_location=device))
            self.policy.load_state_dict(torch.load(f'{checkpoint_path}/ppo_eval_net.pth', map_location=device))
