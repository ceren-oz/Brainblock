"""
Soft Actor-Critic (SAC) for Discrete Action Spaces
=====================================================
Based on "Soft Actor-Critic for Discrete Action Settings" (Christodoulou, 2019).

Key adaptations for BrainBlock:
  - Discrete action space with action masking (invalid actions → -inf before softmax)
  - Separate actor / twin-critic networks
  - Automatic entropy tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional
import copy


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity,), dtype=np.int64)
        self.rewards  = np.zeros((capacity,), dtype=np.float32)
        self.dones    = np.zeros((capacity,), dtype=np.float32)
        self.masks    = np.zeros((capacity, 320), dtype=bool)       # valid-action masks
        self.next_masks = np.zeros((capacity, 320), dtype=bool)

    def add(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.obs[self.ptr]        = obs
        self.next_obs[self.ptr]   = next_obs
        self.actions[self.ptr]    = action
        self.rewards[self.ptr]    = reward
        self.dones[self.ptr]      = float(done)
        self.masks[self.ptr]      = mask
        self.next_masks[self.ptr] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.LongTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
            torch.BoolTensor(self.masks[idx]).to(self.device),
            torch.BoolTensor(self.next_masks[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int,
         activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    layers = []
    dims = (in_dim,) + hidden
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), activation()]
    layers.append(nn.Linear(dims[-1], out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Outputs a probability distribution over all 320 actions.
    Invalid actions are masked out (set to -inf before softmax).
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(obs)
        if mask is not None:
            # Where the entire row is masked, fall back to uniform to avoid NaN
            all_masked = ~mask.any(dim=-1, keepdim=True)  # (B,1)
            safe_mask  = mask | all_masked.expand_as(mask)
            logits = logits.masked_fill(~safe_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        # Small eps to avoid log(0)
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

    def get_action(self, obs: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False) -> torch.Tensor:
        probs, _ = self.forward(obs, mask)
        if deterministic:
            return probs.argmax(dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()


class Critic(nn.Module):
    """Twin Q-networks (Q1 and Q2) mapping obs → Q-values for all actions."""
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(obs_dim, hidden, n_actions)
        self.q2 = _mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs), self.q2(obs)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Discrete Soft Actor-Critic with:
      - twin Q-networks + target networks
      - automatic entropy coefficient tuning
      - valid-action masking
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        target_entropy_ratio: float = 0.98,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        update_every: int = 1,
        warmup_steps: int = 1_000,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.n_actions = n_actions
        self._total_steps = 0

        # Networks
        self.actor  = Actor(obs_dim, n_actions, hidden).to(self.device)
        self.critic = Critic(obs_dim, n_actions, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Entropy
        self.auto_entropy = auto_entropy
        if auto_entropy:
            # Target: roughly uniform over valid actions
            self.target_entropy = -np.log(1.0 / n_actions) * target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask: np.ndarray,
                      deterministic: bool = False) -> int:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        action = self.actor.get_action(obs_t, mask_t, deterministic)
        return int(action.item())

    def store(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.add(obs, action, reward, next_obs, done, mask, next_mask)
        self._total_steps += 1

    def update(self) -> Optional[dict]:
        if len(self.buffer) < self.warmup_steps:
            return None
        if self._total_steps % self.update_every != 0:
            return None

        (obs, actions, rewards, next_obs, dones,
         masks, next_masks) = self.buffer.sample(self.batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs, next_masks)
            q1_next, q2_next = self.critic_target(next_obs)
            min_q_next = torch.min(q1_next, q2_next)
            # Expected value under next policy (over all actions)
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + (1.0 - dones) * self.gamma * v_next

        q1, q2 = self.critic(obs)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ---- Actor update ----
        probs, log_probs = self.actor(obs, masks)
        with torch.no_grad():
            q1_pi, q2_pi = self.critic(obs)
            min_q_pi = torch.min(q1_pi, q2_pi)

        # Policy gradient: maximise E[Q] - α*H
        actor_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ---- Alpha (entropy) update ----
        alpha_loss = None
        if self.auto_entropy:
            # H(π) ≈ -Σ π log π
            entropy = -(probs * log_probs).sum(dim=1).mean().detach()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft update target ----
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss.item() if alpha_loss is not None else 0.0,
        }

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.auto_entropy and ckpt["log_alpha"] is not None:
            self.log_alpha.data = ckpt["log_alpha"].data
            self.alpha = self.log_alpha.exp().item()
