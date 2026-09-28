"""Rollout buffer, GAE and PPO clipped-objective update.

select_action() is standalone so inference needs no optimizer or buffer;
PPOAgent holds the buffer, optimizer and update() used during training.
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
    """Returns (action_idx, log_prob, value) for a single masked decision."""
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


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """EV = 1 - Var(returns - values) / Var(returns); NaN when returns have
    near-zero variance.
    """
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_returns)


class RunningMeanStd:
    """Running mean/variance over sequential batches (Chan et al. parallel
    Welford update). Persistent state, unlike per-minibatch advantage
    normalization, so it is saved in checkpoints. count starts at a small
    epsilon to avoid division by zero on the first update.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def normalize(self, x, epsilon: float = 1e-8):
        """(x - mean) / (std + epsilon). Applied to both sides of the value
        MSE, so the mean cancels and only the loss scale changes; values,
        bootstrap and GAE need no denormalization.
        """
        return (x - self.mean) / (self.std + epsilon)

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, state: dict):
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


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
    def __init__(self, model: ActorCriticMLP, ppo_config, steps_already_trained: int = 0):
        self.model = model
        self.config = ppo_config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
        self.buffer = RolloutBuffer()
        # Carries decay progress across PPOAgent instances, which are
        # re-created every training burst.
        self.steps_already_trained = steps_already_trained
        self.steps_processed = 0
        # Only used when config.normalize_returns=True.
        self.return_rms = RunningMeanStd()

    def value_of(self, features: np.ndarray) -> float:
        state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(state_t)
        return float(value.item())

    def store(self, features, mask, action, log_prob, value, reward, done):
        self.buffer.add(features, mask, action, log_prob, value, reward, done)
        self.steps_processed += 1

    def ready_to_update(self) -> bool:
        return len(self.buffer) >= self.config.rollout_steps

    def _decayed_entropy_coef_and_lr(self):
        """Linearly interpolates (entropy_coef, learning_rate) toward their
        *_end values as progress = (steps_already_trained + steps_processed)
        / total_timesteps goes from 0 to 1 (clamped). Returns the unchanged
        values when decay is off.
        """
        if not self.config.enable_entropy_lr_decay:
            return self.config.entropy_coef, self.config.learning_rate
        total = self.steps_already_trained + self.steps_processed
        progress = min(1.0, total / self.config.total_timesteps) if self.config.total_timesteps > 0 else 1.0
        entropy_coef = self.config.entropy_coef + (self.config.entropy_coef_end - self.config.entropy_coef) * progress
        learning_rate = self.config.learning_rate + (self.config.learning_rate_end - self.config.learning_rate) * progress
        return entropy_coef, learning_rate

    def update(self, last_value: float) -> dict:
        n = len(self.buffer)
        if n == 0:
            return {}

        entropy_coef, learning_rate = self._decayed_entropy_coef_and_lr()
        if self.config.enable_entropy_lr_decay:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

        advantages, returns = compute_gae(
            rewards=self.buffer.rewards,
            values=self.buffer.values,
            dones=self.buffer.dones,
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        # Whole-batch diagnostic of the value function that collected this
        # rollout, computed from the stored pre-update predictions.
        batch_explained_variance = explained_variance(
            np.array(self.buffer.values, dtype=np.float32), returns
        )

        # Updated once per call from the full batch of raw returns; everything
        # except value_loss stays on the raw scale.
        if self.config.normalize_returns:
            self.return_rms.update(returns)

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
        total_grad_norm = 0.0
        min_policy_loss, max_policy_loss = float("inf"), float("-inf")
        min_value_loss, max_value_loss = float("inf"), float("-inf")
        min_entropy, max_entropy = float("inf"), float("-inf")
        min_grad_norm, max_grad_norm_seen = float("inf"), float("-inf")
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

                if self.config.normalize_returns:
                    # Same running stats on both sides; only value_loss's
                    # scale changes, not the value head's target.
                    value_loss = F.mse_loss(
                        self.return_rms.normalize(values),
                        self.return_rms.normalize(mb_returns),
                    )
                else:
                    value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.config.value_loss_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # Pre-clipping total norm, for monitoring.
                grad_norm = float(clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).item())
                self.optimizer.step()

                pl = float(policy_loss.item())
                vl = float(value_loss.item())
                ent = float(entropy.item())

                total_policy_loss += pl
                total_value_loss += vl
                total_entropy += ent
                total_grad_norm += grad_norm
                min_policy_loss, max_policy_loss = min(min_policy_loss, pl), max(max_policy_loss, pl)
                min_value_loss, max_value_loss = min(min_value_loss, vl), max(max_value_loss, vl)
                min_entropy, max_entropy = min(min_entropy, ent), max(max_entropy, ent)
                min_grad_norm, max_grad_norm_seen = min(min_grad_norm, grad_norm), max(max_grad_norm_seen, grad_norm)
                n_minibatches += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_minibatches, 1),
            "policy_loss_min": min_policy_loss,
            "policy_loss_max": max_policy_loss,
            "value_loss": total_value_loss / max(n_minibatches, 1),
            "value_loss_min": min_value_loss,
            "value_loss_max": max_value_loss,
            "entropy": total_entropy / max(n_minibatches, 1),
            "entropy_min": min_entropy,
            "entropy_max": max_entropy,
            "gradient_norm": total_grad_norm / max(n_minibatches, 1),
            "gradient_norm_min": min_grad_norm,
            "gradient_norm_max": max_grad_norm_seen,
            "explained_variance": batch_explained_variance,
            "n_transitions": n,
            "entropy_coef_used": entropy_coef,
            "learning_rate_used": self.optimizer.param_groups[0]["lr"],
        }
