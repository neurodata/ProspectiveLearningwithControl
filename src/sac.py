import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# -------------------------
# Time-embedding
# -------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        t = t.unsqueeze(-1)
        freqs = self.freqs.to(t.device)
        sin = torch.sin(freqs * t)
        cos = torch.cos(freqs * t)

        return torch.cat([sin, cos], dim=-1)

# -------------------------
# Neural network modules
# -------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, t_dim=50, include_time=True):
        super(QNetwork, self).__init__()
        if include_time:
            self.fc1 = nn.Linear(state_dim + t_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.include_time = include_time
        self.time_embed = TimeEmbedding(t_dim)
        self.state_dim = state_dim

    def forward(self, state, time):
        state = F.one_hot(state, num_classes=self.state_dim).float()
        if self.include_time:
            t = self.time_embed(time)
            x = torch.cat([state, t], dim=-1)
        else:
            x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # shape: (batch, action_dim)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, t_dim=50, include_time=True):
        super(PolicyNetwork, self).__init__()
        if include_time:
            self.fc1 = nn.Linear(state_dim + t_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.include_time = include_time
        self.time_embed = TimeEmbedding(t_dim)
        self.state_dim = state_dim 

    def forward(self, state, time):
        state = F.one_hot(state, num_classes=self.state_dim).float()
        if self.include_time:
            t = self.time_embed(time)
            x = torch.cat([state, t], dim=-1)
        else:
            x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

# -------------------------
# Replay buffer
# -------------------------

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, time, reward, next_state):
        self.buffer.append((state, action, time, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, time, reward, next_state = map(np.array, zip(*batch))
        return (
            torch.LongTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(time),
            torch.FloatTensor(reward),
            torch.LongTensor(next_state)
        )

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Discrete SAC Learner
# -------------------------

class SACLearner:
    def __init__(self, include_time=False, state_dim=7, action_dim=3, gamma=0.99, tau=0.005,
                 alpha=0.2, lr=3e-4, buffer_capacity=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.q1 = QNetwork(state_dim, action_dim, include_time=include_time)
        self.q2 = QNetwork(state_dim, action_dim, include_time=include_time)
        self.q1_target = QNetwork(state_dim, action_dim, include_time=include_time)
        self.q2_target = QNetwork(state_dim, action_dim, include_time=include_time)
        self.policy = PolicyNetwork(state_dim, action_dim, include_time=include_time)

        # Copy parameters to target
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, time, eval_mode=False):
        state = torch.LongTensor([state])
        time = torch.FloatTensor([time])
        probs, _ = self.policy(state, time)
        if eval_mode:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.multinomial(probs, 1).item()
        return action

    def update(self, batch_size=64):
        if len(self.replay_buffer) < 200:
            return

        state, action, time, reward, next_state = self.replay_buffer.sample(batch_size)
        next_time = time + 1  # assuming time increments by 1
        
        # Policy evaluation: Q update
        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_state, next_time)
            q1_next = self.q1_target(next_state, next_time)
            q2_next = self.q2_target(next_state, next_time)
            min_q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = reward + self.gamma * v_next

        q1_pred = self.q1(state, time).gather(1, action.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(state, time).gather(1, action.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy improvement
        probs, log_probs = self.policy(state, time)
        q1_vals = self.q1(state, time)
        q2_vals = self.q2(state, time)
        min_q_vals = torch.min(q1_vals, q2_vals)

        policy_loss = (probs * (self.alpha * log_probs - min_q_vals)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update targets
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

