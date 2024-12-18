import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy


# Actor and Critic Networks
class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(net_width, action_dim)
        self.log_std_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        x = self.actor(state)
        mu = torch.tanh(self.mu_head(x))  # Ensure actions are within a range
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def get_dist(self, state):
        mu, std = self.forward(state)
        return torch.distributions.Normal(mu, std)

    def deterministic_act(self, state):
        mu, _ = self.forward(state)
        return mu


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
        )

    def forward(self, state):
        return self.critic(state)


# PPO Agent
class PPO_agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # Initialize Actor
        self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Initialize Critic
        self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Trajectory storage
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, self.action_dim), dtype=np.float32)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            if deterministic:
                action = self.actor.deterministic_act(state)
                return action.cpu().numpy()[0], None
            else:
                dist = self.actor.get_dist(state)
                action = dist.sample().to(torch.float64)  # Ensure the action is in higher precision
                print(action.cpu().numpy()[0])
                logprob_a = dist.log_prob(action).cpu().numpy().flatten()
                return action.cpu().numpy()[0], logprob_a

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay

        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.dvc)

        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)
            deltas = r + self.gamma * vs_ * (~done) - vs
            deltas = deltas.cpu().flatten().numpy()

            adv = [0]
            for delta, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                adv_val = delta + self.gamma * self.lambd * adv[-1] * (~mask)
                adv.append(adv_val)
            adv.reverse()
            adv = torch.tensor(adv[:-1]).unsqueeze(1).float().to(self.dvc)
            td_target = adv + vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        perm = np.arange(s.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(self.dvc)

        s, a, td_target, adv, logprob_a = \
            s[perm], a[perm], td_target[perm], adv[perm], logprob_a[perm]

        for _ in range(self.K_epochs):
            dist = self.actor.get_dist(s)
            dist_entropy = dist.entropy().sum(1, keepdim=True)
            logprob_a_now = dist.log_prob(a).sum(1, keepdim=True)
            ratio = torch.exp(logprob_a_now - logprob_a)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv
            a_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            c_loss = (self.critic(s) - td_target).pow(2).mean()
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            self.critic_optimizer.step()

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw


# Custom Environment
class CustomContinuousEnv:
    def __init__(self):
        self.state_dim = 1
        self.action_dim = 1
        self.target_action = 0.5
        self.state = np.random.uniform(0, 1, size=(self.state_dim,))
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        self.state = np.random.uniform(0, 1, size=(self.state_dim,))
        self.current_step = 0
        return self.state

    def step(self, action):
        action = np.clip(action, 0, 1)
        reward = -((action - self.target_action) ** 2).sum()
        self.state = np.random.uniform(0, 1, size=(self.state_dim,))
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.state, reward, done, {}

    def render(self):
        print(f"State: {self.state}")

    def close(self):
        pass


# Main Training Loop
env = CustomContinuousEnv()
ppo_params = {
    "state_dim": env.state_dim,
    "action_dim": env.action_dim,
    "net_width": 64,
    "dvc": "cuda" if torch.cuda.is_available() else "cpu",
    "a_lr": 1e-4,
    "c_lr": 1e-3,
    "T_horizon": 64,
    "gamma": 0.99,
    "lambd": 0.95,
    "K_epochs": 10,
    "clip_rate": 0.2,
    "entropy_coef": 0.01,
    "entropy_coef_decay": 0.995,
}

agent = PPO_agent(**ppo_params)
index = 0
for episode in range(1000):
    state = env.reset()
    for t in range(env.max_steps):
        action, logprob = agent.select_action(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        agent.put_data(state, action, reward, next_state, logprob, done, False, index)
        state = next_state
        index += 1
        if done:
            break
        if t % 64 == 0:
            index = 0
    agent.train()
    if episode % 50 == 0:
        print(f"Episode {episode} completed.")
