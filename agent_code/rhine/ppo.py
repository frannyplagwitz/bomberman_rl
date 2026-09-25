"""Rollout buffer, GAE, and PPO clipped-objective update.

select_action() is a standalone function (no PPOAgent instance needed) so
callbacks.py can use it in eval-only mode without building an optimizer/buffer.
PPOAgent wraps the buffer + optimizer + update() used only when self.train
is True (train.py).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from .model import ActorCriticMLP, masked_logits


def select_action(model: ActorCriticMLP, features: np.ndarray, mask: np.ndarray, deterministic: bool = False):
    """Returns (action_idx, log_prob, value). Used both for live decisions (callbacks.act)
    and, via PPOAgent, is not re-used for training (train.py reuses the values act() already
    computed for the same step instead of recomputing them).
    """
    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(state_t)
        logits = masked_logits(logits, mask_t)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, state, mask, action, log_prob, value, reward, done):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def mark_last_done(self):
        if self.dones:
            self.dones[-1] = True

    def clear(self):
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


class PPOAgent:
    def __init__(self, model: ActorCriticMLP, ppo_config):
        self.model = model
        self.config = ppo_config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
        self.buffer = RolloutBuffer()

    def value_of(self, features: np.ndarray) -> float:
        state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(state_t)
        return float(value.item())

    def store(self, features, mask, action, log_prob, value, reward, done):
        self.buffer.add(features, mask, action, log_prob, value, reward, done)

    def ready_to_update(self) -> bool:
        return len(self.buffer) >= self.config.rollout_steps

    def update(self, last_value: float) -> dict:
        n = len(self.buffer)
        if n == 0:
            return {}

        advantages, returns = compute_gae(
            rewards=self.buffer.rewards,
            values=self.buffer.values,
            dones=self.buffer.dones,
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        states = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32)
        masks = torch.as_tensor(np.array(self.buffer.masks), dtype=torch.bool)
        actions = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.long)
        old_log_probs = torch.as_tensor(np.array(self.buffer.log_probs), dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        idxs = np.arange(n)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_minibatches = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.config.minibatch_size):
                mb_idx = idxs[start:start + self.config.minibatch_size]
                mb_states = states[mb_idx]
                mb_masks = masks[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                if self.config.normalize_advantage and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                logits, values = self.model(mb_states)
                logits = masked_logits(logits, mb_masks)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                n_minibatches += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_minibatches, 1),
            "value_loss": total_value_loss / max(n_minibatches, 1),
            "entropy": total_entropy / max(n_minibatches, 1),
            "n_transitions": n,
        }
