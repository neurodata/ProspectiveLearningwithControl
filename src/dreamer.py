import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
import numpy as np
import random
from collections import deque

# -------------------------
# Time-embedding (same pattern as sac.py)
# -------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim=50):
        super().__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2).float())
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        freqs = self.freqs.to(t.device)
        sin = torch.sin(freqs * t)
        cos = torch.cos(freqs * t)
        return torch.cat([sin, cos], dim=-1)

# -------------------------
# Categorical with uniform mixing (DreamerV3)
# -------------------------

def categorical_unimix(logits, unimix=0.01):
    """Apply uniform mixing to prevent categorical collapse."""
    probs = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    probs = (1 - unimix) * probs + unimix * uniform
    return D.OneHotCategoricalStraightThrough(probs=probs)

# -------------------------
# Symlog transforms (DreamerV3)
# -------------------------

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# -------------------------
# RSSM (Recurrent State-Space Model)
# -------------------------

class RSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=128, z_dim=8, z_classes=8):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.z_classes = z_classes
        self.z_flat_dim = z_dim * z_classes
        self.action_dim = action_dim

        # Sequence model: projects (z_flat + action) before GRU
        self.pre_gru = nn.Linear(self.z_flat_dim + action_dim, h_dim)
        self.gru = nn.GRUCell(h_dim, h_dim)

        # Prior: h -> z_prior logits
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, self.z_flat_dim)
        )

        # Posterior: (h, obs) -> z_posterior logits
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, self.z_flat_dim)
        )

    def initial_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, self.z_classes, device=device)
        return h, z

    def observe_step(self, prev_h, prev_z, action_oh, obs):
        """Single step with real observation (posterior inference)."""
        z_flat = prev_z.reshape(prev_z.shape[0], -1)
        gru_in = self.pre_gru(torch.cat([z_flat, action_oh], dim=-1))
        h = self.gru(F.silu(gru_in), prev_h)

        # Prior
        prior_logits = self.prior_net(h).reshape(-1, self.z_dim, self.z_classes)

        # Posterior
        post_input = torch.cat([h, obs], dim=-1)
        posterior_logits = self.posterior_net(post_input).reshape(-1, self.z_dim, self.z_classes)
        posterior_dist = categorical_unimix(posterior_logits)
        z = posterior_dist.rsample()

        return h, z, prior_logits, posterior_logits

    def imagine_step(self, prev_h, prev_z, action_oh):
        """Single step without observation (imagination via prior)."""
        z_flat = prev_z.reshape(prev_z.shape[0], -1)
        gru_in = self.pre_gru(torch.cat([z_flat, action_oh], dim=-1))
        h = self.gru(F.silu(gru_in), prev_h)

        prior_logits = self.prior_net(h).reshape(-1, self.z_dim, self.z_classes)
        prior_dist = categorical_unimix(prior_logits)
        z = prior_dist.rsample()

        return h, z, prior_logits

    def get_latent(self, h, z):
        z_flat = z.reshape(z.shape[0], -1)
        return torch.cat([h, z_flat], dim=-1)

# -------------------------
# World Model
# -------------------------

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=128, z_dim=8, z_classes=8):
        super().__init__()
        self.rssm = RSSM(obs_dim, action_dim, h_dim, z_dim, z_classes)
        latent_dim = h_dim + z_dim * z_classes

        # Observation decoder
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, obs_dim)
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

# -------------------------
# Actor
# -------------------------

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, latent):
        logits = self.net(latent)
        return categorical_unimix(logits, unimix=0.01)

# -------------------------
# Critic
# -------------------------

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent):
        return self.net(latent).squeeze(-1)

# -------------------------
# Sequence Replay Buffer
# -------------------------

class SequenceReplayBuffer:
    def __init__(self, capacity=100000):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.capacity = capacity
        self.size = 0

    def push(self, obs, action, reward):
        if self.size < self.capacity:
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
        else:
            idx = self.size % self.capacity
            self.obs[idx] = obs
            self.actions[idx] = action
            self.rewards[idx] = reward
        self.size += 1

    def sample_sequences(self, batch_size, seq_len):
        effective = min(self.size, self.capacity)
        max_start = effective - seq_len
        if max_start <= 0:
            return None
        starts = np.random.randint(0, max_start, size=batch_size)

        obs_seqs = []
        act_seqs = []
        rew_seqs = []
        for s in starts:
            obs_seqs.append(np.stack(self.obs[s:s + seq_len]))
            act_seqs.append(np.array(self.actions[s:s + seq_len]))
            rew_seqs.append(np.array(self.rewards[s:s + seq_len]))

        return {
            'obs': torch.FloatTensor(np.array(obs_seqs)).permute(1, 0, 2),      # (seq, batch, obs_dim)
            'actions': torch.LongTensor(np.array(act_seqs)).permute(1, 0),       # (seq, batch)
            'rewards': torch.FloatTensor(np.array(rew_seqs)).permute(1, 0),      # (seq, batch)
        }

# -------------------------
# DreamerV3 Learner
# -------------------------

class DreamerV3Learner:
    def __init__(self, include_time=True, state_dim=7, action_dim=3,
                 gamma=0.99, lam=0.95, h_dim=128, z_dim=8, z_classes=8,
                 lr_world=1e-4, lr_actor=3e-5, lr_critic=3e-5,
                 seq_len=16, imagination_horizon=8,
                 buffer_capacity=100000, learning_starts=500,
                 entropy_coeff=3e-4, tau=0.02):

        self.include_time = include_time
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.seq_len = seq_len
        self.imagination_horizon = imagination_horizon
        self.learning_starts = learning_starts
        self.entropy_coeff = entropy_coeff
        self.tau = tau

        self.obs_dim = state_dim + 50 if include_time else state_dim
        latent_dim = h_dim + z_dim * z_classes

        self.world_model = WorldModel(self.obs_dim, action_dim, h_dim, z_dim, z_classes)
        self.actor = Actor(latent_dim, action_dim)
        self.critic = Critic(latent_dim)
        self.target_critic = Critic(latent_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=lr_world)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = SequenceReplayBuffer(buffer_capacity)

        self.time_embed = TimeEmbedding(50)

        # RSSM states for online interaction
        self._h = None
        self._z = None
        self._prev_action = 1  # default: stay (action index 1)

        # Separate RSSM states for evaluation
        self._eval_h = None
        self._eval_z = None
        self._eval_prev_action = 1

        self._step_count = 0

    def encode_obs(self, state, time):
        """Convert scalar state + time to observation vector."""
        s = F.one_hot(torch.LongTensor([state]), self.state_dim).float()
        if self.include_time:
            t_emb = self.time_embed(torch.FloatTensor([time]))
            return torch.cat([s, t_emb], dim=-1)
        return s

    def select_action(self, state, time, eval_mode=False):
        with torch.no_grad():
            obs = self.encode_obs(state, time)

            if eval_mode:
                # Use separate eval RSSM state
                if self._eval_h is None:
                    self._eval_h, self._eval_z = self.world_model.rssm.initial_state(1)
                action_oh = F.one_hot(torch.LongTensor([self._eval_prev_action]), self.action_dim).float()
                self._eval_h, self._eval_z, _, _ = self.world_model.rssm.observe_step(
                    self._eval_h, self._eval_z, action_oh, obs)
                latent = self.world_model.rssm.get_latent(self._eval_h, self._eval_z)
                dist = self.actor(latent)
                action = dist.probs.argmax(dim=-1).item()
                self._eval_prev_action = action
            else:
                # Use training RSSM state
                if self._h is None:
                    self._h, self._z = self.world_model.rssm.initial_state(1)
                action_oh = F.one_hot(torch.LongTensor([self._prev_action]), self.action_dim).float()
                self._h, self._z, _, _ = self.world_model.rssm.observe_step(
                    self._h, self._z, action_oh, obs)
                latent = self.world_model.rssm.get_latent(self._h, self._z)
                dist = self.actor(latent)
                action = dist.sample().argmax(dim=-1).item()
                self._prev_action = action

            return action

    def reset_eval_state(self):
        """Reset eval RSSM state before each evaluation rollout."""
        self._eval_h = None
        self._eval_z = None
        self._eval_prev_action = 1

    def update(self, batch_size=16):
        self._step_count += 1
        if self.replay_buffer.size < self.learning_starts:
            return

        data = self.replay_buffer.sample_sequences(batch_size, self.seq_len)
        if data is None:
            return

        # Phase 1: Train world model
        posteriors_h, posteriors_z = self._train_world_model(data)

        # Phase 2: Train actor-critic on imagined trajectories
        self._train_actor_critic(posteriors_h, posteriors_z)

        # Soft update target critic
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _train_world_model(self, data):
        obs = data['obs']          # (seq, batch, obs_dim)
        actions = data['actions']  # (seq, batch)
        rewards = data['rewards']  # (seq, batch)

        seq_len, batch = obs.shape[:2]
        h, z = self.world_model.rssm.initial_state(batch)

        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        all_h = []
        all_z = []

        for t in range(seq_len):
            action_oh = F.one_hot(actions[t], self.action_dim).float()
            h, z, prior_logits, posterior_logits = \
                self.world_model.rssm.observe_step(h, z, action_oh, obs[t])

            latent = self.world_model.rssm.get_latent(h, z)

            # Reconstruction loss
            obs_pred = self.world_model.obs_decoder(latent)
            recon_loss = recon_loss + F.mse_loss(obs_pred, obs[t])

            # Reward prediction loss (symlog transform for scale invariance)
            reward_pred = self.world_model.reward_head(latent).squeeze(-1)
            reward_loss = reward_loss + F.mse_loss(reward_pred, symlog(rewards[t]))

            # KL divergence with free nats
            # DreamerV3: 0.5 * max(KL_dyn, free) + 0.1 * max(KL_rep, free)
            prior_logits_r = prior_logits.reshape(batch * self.world_model.rssm.z_dim, -1)
            post_logits_r = posterior_logits.reshape(batch * self.world_model.rssm.z_dim, -1)

            prior_dist = D.Categorical(logits=prior_logits_r)
            post_dist = D.Categorical(logits=post_logits_r)

            kl_dyn = D.kl_divergence(
                D.Categorical(logits=post_logits_r.detach()),
                prior_dist
            ).reshape(batch, -1).sum(-1).mean()

            kl_rep = D.kl_divergence(
                post_dist,
                D.Categorical(logits=prior_logits_r.detach())
            ).reshape(batch, -1).sum(-1).mean()

            kl_loss = kl_loss + 0.5 * torch.clamp(kl_dyn, min=1.0) + 0.1 * torch.clamp(kl_rep, min=1.0)

            all_h.append(h.detach())
            all_z.append(z.detach())

        total_loss = (recon_loss + reward_loss + kl_loss) / seq_len

        self.world_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000.0)
        self.world_optimizer.step()

        return all_h, all_z

    def _imagine(self, start_h, start_z):
        """Imagine trajectories through the world model using the actor."""
        h, z = start_h, start_z
        imagined_latents = []
        imagined_rewards = []
        imagined_log_probs = []
        imagined_entropies = []

        for _ in range(self.imagination_horizon):
            latent = self.world_model.rssm.get_latent(h, z)
            imagined_latents.append(latent)

            action_dist = self.actor(latent)
            action = action_dist.rsample()  # straight-through one-hot
            imagined_log_probs.append(action_dist.log_prob(action))
            imagined_entropies.append(action_dist.entropy())

            h, z, _ = self.world_model.rssm.imagine_step(h, z, action)
            next_latent = self.world_model.rssm.get_latent(h, z)
            reward_pred = self.world_model.reward_head(next_latent).squeeze(-1)
            imagined_rewards.append(symexp(reward_pred))

        final_latent = self.world_model.rssm.get_latent(h, z)
        return imagined_latents, imagined_rewards, imagined_log_probs, imagined_entropies, final_latent

    def _compute_lambda_returns(self, imagined_rewards, imagined_latents, final_latent, batch):
        """Compute lambda-returns from imagined trajectories (all detached)."""
        H = self.imagination_horizon
        with torch.no_grad():
            values = [self.target_critic(lat.detach()) for lat in imagined_latents]
            final_value = self.target_critic(final_latent.detach())

        returns = torch.zeros(H, batch)
        last_return = final_value

        for t in reversed(range(H)):
            reward = imagined_rewards[t].detach()
            value = values[t]
            next_value = values[t + 1] if t + 1 < H else final_value
            td_error = reward + self.gamma * next_value - value
            last_return = value + td_error + self.gamma * self.lam * (last_return - next_value)
            returns[t] = last_return

        return returns.detach()

    def _train_actor_critic(self, all_h, all_z):
        """Train actor and critic on imagined trajectories from posterior states."""
        seq_len = len(all_h)
        batch = all_h[0].shape[0]

        t_idx = np.random.randint(0, seq_len)
        start_h = all_h[t_idx]  # already detached from world model graph
        start_z = all_z[t_idx]

        # --- Critic update: imagine with no actor gradients ---
        with torch.no_grad():
            im_latents, im_rewards, _, _, im_final = self._imagine(start_h, start_z)
        returns = self._compute_lambda_returns(im_rewards, im_latents, im_final, batch)

        critic_loss = 0.0
        H = self.imagination_horizon
        for t in range(H):
            v = self.critic(im_latents[t])  # critic gets gradients
            critic_loss = critic_loss + F.mse_loss(v, returns[t])
        critic_loss = critic_loss / H

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # --- Actor update: re-imagine with actor gradients ---
        im_latents2, im_rewards2, im_log_probs, im_entropies, im_final2 = \
            self._imagine(start_h, start_z)
        returns2 = self._compute_lambda_returns(im_rewards2, im_latents2, im_final2, batch)

        actor_loss = 0.0
        for t in range(H):
            with torch.no_grad():
                advantage = returns2[t] - self.critic(im_latents2[t].detach())
            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            actor_loss = actor_loss - (im_log_probs[t] * advantage).mean()
            actor_loss = actor_loss - self.entropy_coeff * im_entropies[t].mean()
        actor_loss = actor_loss / H

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
