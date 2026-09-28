"""Tests for PPO monitoring metrics: explained_variance and gradient_norm
computed in PPOAgent.update(), and scripts/common.py::training_loss_stats()
aggregating mean/min/max across a burst's updates.
"""
import dataclasses

import numpy as np
import pytest

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent, explained_variance
import agent_code.rhine.ppo as ppo_module
from agent_code.rhine.scripts import common


def _make_agent(ppo_config):
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    return PPOAgent(model, ppo_config), n_features


# --- explained_variance() as a standalone function, known inputs -----------------

def test_explained_variance_perfect_prediction_is_one():
    returns = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = returns.copy()
    assert explained_variance(values, returns) == pytest.approx(1.0, abs=1e-6)


def test_explained_variance_constant_prediction_is_zero():
    returns = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.zeros_like(returns)
    assert explained_variance(values, returns) == pytest.approx(0.0, abs=1e-6)


def test_explained_variance_worse_than_constant_is_negative():
    returns = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.array([10.0, -8.0, 15.0, -6.0], dtype=np.float32)  # Far off.
    assert explained_variance(values, returns) < 0.0


def test_explained_variance_zero_variance_returns_is_nan():
    returns = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.isnan(explained_variance(values, returns))


# --- explained_variance wired into PPOAgent.update() ------------------------------
# All transitions use done=True, so returns equal rewards exactly and tests can
# choose rewards/values directly.

def test_update_explained_variance_perfect_when_values_equal_rewards():
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=6, minibatch_size=6, update_epochs=1)
    agent, n_features = _make_agent(base)
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    features = np.zeros(n_features, dtype=np.float32)
    for i in range(6):
        r = float(i + 1)
        agent.store(features, mask, 0, 0.0, value=r, reward=r, done=True)
    result = agent.update(last_value=0.0)
    assert result["explained_variance"] == pytest.approx(1.0, abs=1e-5)


def test_update_explained_variance_zero_when_values_constant():
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=6, minibatch_size=6, update_epochs=1)
    agent, n_features = _make_agent(base)
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    features = np.zeros(n_features, dtype=np.float32)
    for i in range(6):
        agent.store(features, mask, 0, 0.0, value=0.0, reward=float(i + 1), done=True)
    result = agent.update(last_value=0.0)
    assert result["explained_variance"] == pytest.approx(0.0, abs=1e-5)


# --- gradient_norm: correctly captured, not just discarded ------------------------

def test_gradient_norm_matches_live_clip_grad_norm_return_values(monkeypatch):
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=8, minibatch_size=4, update_epochs=2)
    agent, n_features = _make_agent(base)
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    rng = np.random.default_rng(0)
    for i in range(8):
        features = rng.normal(size=n_features).astype(np.float32)
        agent.store(features, mask, i % cfg.MODEL_CONFIG.n_actions, 0.0, value=0.0, reward=float(i), done=(i == 7))

    recorded = []
    original = ppo_module.clip_grad_norm_

    def spy(*args, **kwargs):
        norm = original(*args, **kwargs)
        recorded.append(float(norm.item()))
        return norm

    monkeypatch.setattr(ppo_module, "clip_grad_norm_", spy)
    result = agent.update(last_value=0.0)

    # One clip call per minibatch per epoch.
    assert len(recorded) == 4
    assert result["gradient_norm"] == pytest.approx(float(np.mean(recorded)), abs=1e-6)
    assert result["gradient_norm_min"] == pytest.approx(float(np.min(recorded)), abs=1e-6)
    assert result["gradient_norm_max"] == pytest.approx(float(np.max(recorded)), abs=1e-6)
    assert result["gradient_norm_min"] <= result["gradient_norm"] <= result["gradient_norm_max"]


def test_update_return_dict_has_all_expected_monitoring_keys():
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=4, minibatch_size=4, update_epochs=1)
    agent, n_features = _make_agent(base)
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    features = np.zeros(n_features, dtype=np.float32)
    for i in range(4):
        agent.store(features, mask, 0, 0.0, value=float(i), reward=float(i), done=True)
    result = agent.update(last_value=0.0)

    for key in (
        "policy_loss", "policy_loss_min", "policy_loss_max",
        "value_loss", "value_loss_min", "value_loss_max",
        "entropy", "entropy_min", "entropy_max",
        "gradient_norm", "gradient_norm_min", "gradient_norm_max",
        "explained_variance",
    ):
        assert key in result
    assert result["policy_loss_min"] <= result["policy_loss"] <= result["policy_loss_max"]
    assert result["value_loss_min"] <= result["value_loss"] <= result["value_loss_max"]
    assert result["entropy_min"] <= result["entropy"] <= result["entropy_max"]


# --- scripts/common.py::training_loss_stats() min/max aggregation across a burst --
# Uses lightweight stand-ins for the world/agent chain that
# common.training_loss_means() reads, isolating the aggregation arithmetic.

class _FakeSelf:
    def __init__(self, loss_history):
        self.loss_history = loss_history


class _FakeRunner:
    def __init__(self, fake_self):
        self.fake_self = fake_self


class _FakeBackend:
    def __init__(self, fake_self):
        self.runner = _FakeRunner(fake_self)


class _FakeAgent:
    def __init__(self, fake_self):
        self.backend = _FakeBackend(fake_self)


class _FakeWorld:
    def __init__(self, loss_history):
        self.agents = [_FakeAgent(_FakeSelf(loss_history))]


def _fake_update_entry(policy_loss, value_loss, entropy, grad_norm, ev, spread=1.0):
    return {
        "policy_loss": policy_loss, "policy_loss_min": policy_loss - spread, "policy_loss_max": policy_loss + spread,
        "value_loss": value_loss, "value_loss_min": value_loss - spread, "value_loss_max": value_loss + spread,
        "entropy": entropy, "entropy_min": entropy - spread, "entropy_max": entropy + spread,
        "gradient_norm": grad_norm, "gradient_norm_min": grad_norm - spread, "gradient_norm_max": grad_norm + spread,
        "explained_variance": ev,
    }


def test_training_loss_stats_none_when_no_history():
    assert common.training_loss_stats(_FakeWorld([])) is None


def test_training_loss_stats_mean_is_exact_for_equal_sized_updates():
    history = [
        _fake_update_entry(policy_loss=1.0, value_loss=2.0, entropy=0.5, grad_norm=3.0, ev=0.1, spread=0.5),
        _fake_update_entry(policy_loss=3.0, value_loss=4.0, entropy=0.7, grad_norm=5.0, ev=0.3, spread=0.5),
    ]
    stats = common.training_loss_stats(_FakeWorld(history))
    assert stats["policy_loss_mean"] == pytest.approx(2.0)
    assert stats["value_loss_mean"] == pytest.approx(3.0)
    assert stats["entropy_mean"] == pytest.approx(0.6)
    assert stats["gradient_norm_mean"] == pytest.approx(4.0)
    assert stats["explained_variance_mean"] == pytest.approx(0.2)


def test_training_loss_stats_min_max_recovers_true_burst_extremes():
    # The burst-level min must come from update #1's own min, not from
    # comparing the two updates' means.
    history = [
        _fake_update_entry(policy_loss=1.0, value_loss=2.0, entropy=0.5, grad_norm=3.0, ev=0.1, spread=0.5),
        _fake_update_entry(policy_loss=3.0, value_loss=4.0, entropy=0.7, grad_norm=5.0, ev=0.3, spread=0.5),
    ]
    stats = common.training_loss_stats(_FakeWorld(history))
    assert stats["policy_loss_min"] == pytest.approx(0.5)
    assert stats["policy_loss_max"] == pytest.approx(3.5)
    assert stats["value_loss_min"] == pytest.approx(1.5)
    assert stats["value_loss_max"] == pytest.approx(4.5)
    assert stats["gradient_norm_min"] == pytest.approx(2.5)
    assert stats["gradient_norm_max"] == pytest.approx(5.5)
    assert stats["explained_variance_min"] == pytest.approx(0.1)
    assert stats["explained_variance_max"] == pytest.approx(0.3)


def test_training_loss_stats_ignores_nan_explained_variance_entries():
    # An update with ~zero return variance reports EV=NaN; burst statistics
    # must skip it rather than propagate NaN.
    history = [
        _fake_update_entry(policy_loss=1.0, value_loss=2.0, entropy=0.5, grad_norm=3.0, ev=float("nan")),
        _fake_update_entry(policy_loss=1.0, value_loss=2.0, entropy=0.5, grad_norm=3.0, ev=0.4),
    ]
    stats = common.training_loss_stats(_FakeWorld(history))
    assert stats["explained_variance_mean"] == pytest.approx(0.4)
    assert stats["explained_variance_min"] == pytest.approx(0.4)
    assert stats["explained_variance_max"] == pytest.approx(0.4)
