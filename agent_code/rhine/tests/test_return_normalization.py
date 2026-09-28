"""Tests for return normalization: RunningMeanStd, its use in PPOAgent.update()
(value_loss only; GAE, advantages, policy_loss, explained_variance and
value_of() stay on the raw scale), and checkpoint persistence of its stats.
"""
import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent, RunningMeanStd


def _make_agent(ppo_config):
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    return PPOAgent(model, ppo_config), n_features


def _store_transitions(agent, n_features, rewards, values):
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    features = np.zeros(n_features, dtype=np.float32)
    for r, v in zip(rewards, values):
        agent.store(features, mask, 0, 0.0, value=v, reward=r, done=True)


# --- RunningMeanStd numeric correctness -------------------------------------------

def test_running_mean_std_matches_numpy_over_sequential_batches():
    rng = np.random.default_rng(0)
    batches = [rng.normal(loc=5.0, scale=2.0, size=500).astype(np.float32) for _ in range(4)]
    rms = RunningMeanStd()
    for b in batches:
        rms.update(b)

    all_data = np.concatenate(batches)
    # The epsilon initial count is negligible against this many samples.
    assert rms.mean == pytest.approx(float(np.mean(all_data)), abs=1e-3)
    assert rms.std == pytest.approx(float(np.std(all_data)), rel=1e-3)


def test_running_mean_std_single_batch_matches_numpy_closely():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=-3.0, scale=0.7, size=1000).astype(np.float32)
    rms = RunningMeanStd()
    rms.update(data)
    assert rms.mean == pytest.approx(float(np.mean(data)), abs=1e-3)
    assert rms.std == pytest.approx(float(np.std(data)), rel=1e-3)


def test_running_mean_std_accumulates_across_updates_not_overwrites():
    rms = RunningMeanStd()
    rms.update(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    count_after_first = rms.count
    rms.update(np.array([1.0, 1.0], dtype=np.float32))
    assert rms.count == pytest.approx(count_after_first + 2)


def test_running_mean_std_state_dict_round_trip():
    rms = RunningMeanStd()
    rms.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    state = rms.state_dict()

    restored = RunningMeanStd()
    restored.load_state_dict(state)
    assert restored.mean == pytest.approx(rms.mean)
    assert restored.var == pytest.approx(rms.var)
    assert restored.count == pytest.approx(rms.count)


# --- The core mathematical property update()'s value_loss path relies on ---------

def test_normalizing_both_sides_of_mse_rescales_by_inverse_variance():
    """Normalizing both MSE sides with the same stats divides the loss by
    std**2 exactly, since the shared mean cancels.
    """
    rms = RunningMeanStd()
    rms.update(np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32))

    a = torch.tensor([12.0, 18.0, 33.0, 41.0, 48.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    raw_mse = F.mse_loss(a, b)
    normalized_mse = F.mse_loss(rms.normalize(a), rms.normalize(b))
    expected = raw_mse / (rms.std ** 2)
    assert float(normalized_mse) == pytest.approx(float(expected), rel=1e-4)


def test_normalize_zero_std_guarded_by_epsilon():
    rms = RunningMeanStd()
    rms.update(np.array([5.0, 5.0, 5.0], dtype=np.float32))  # Zero variance.
    # normalize() must not divide by zero.
    result = rms.normalize(torch.tensor([5.0, 6.0]))
    assert torch.isfinite(result).all()


# --- PPOAgent.update() gating: disabled path is a true no-op ----------------------

def test_normalize_returns_disabled_leaves_return_rms_untouched():
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=6, minibatch_size=6, update_epochs=1,
                                normalize_returns=False)
    agent, n_features = _make_agent(base)
    _store_transitions(agent, n_features, rewards=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    agent.update(last_value=0.0)

    assert agent.return_rms.mean == 0.0
    assert agent.return_rms.var == 1.0
    assert agent.return_rms.count == pytest.approx(1e-4)


def test_normalize_returns_enabled_updates_return_rms():
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=6, minibatch_size=6, update_epochs=1,
                                normalize_returns=True)
    agent, n_features = _make_agent(base)
    _store_transitions(agent, n_features, rewards=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    agent.update(last_value=0.0)

    assert agent.return_rms.count > 1e-4 + 1e-9
    assert agent.return_rms.mean != 0.0  # Rewards are strictly positive.


def test_normalize_returns_default_is_true():
    # Matches the configuration of every delivered Task 2/3 checkpoint.
    assert cfg.PPO_CONFIG.normalize_returns is True


# --- explained_variance/GAE stay on the raw scale regardless of normalize_returns -

@pytest.mark.parametrize("normalize_returns", [False, True])
def test_explained_variance_unaffected_by_normalize_returns(normalize_returns):
    """With value == reward under done=True, EV is exactly 1.0 unless return
    normalization leaks into its computation.
    """
    base = dataclasses.replace(cfg.PPO_CONFIG, rollout_steps=6, minibatch_size=6, update_epochs=1,
                                normalize_returns=normalize_returns)
    agent, n_features = _make_agent(base)
    _store_transitions(agent, n_features, rewards=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = agent.update(last_value=0.0)
    assert result["explained_variance"] == pytest.approx(1.0, abs=1e-5)


def test_value_of_output_unaffected_by_normalize_returns_flag():
    """value_of() is independent of the normalize_returns flag (two agents with
    identical untrained weights).
    """
    base_off = dataclasses.replace(cfg.PPO_CONFIG, normalize_returns=False)
    base_on = dataclasses.replace(cfg.PPO_CONFIG, normalize_returns=True)
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    agent_off = PPOAgent(model, base_off)
    agent_on = PPOAgent(model, base_on)

    features = np.random.default_rng(0).normal(size=n_features).astype(np.float32)
    assert agent_off.value_of(features) == pytest.approx(agent_on.value_of(features))


# --- Checkpoint save/restore, through train.py's actual functions -----------------

class _StubLogger:
    def info(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass


def _make_fake_self():
    n_features = cfg.n_features_active()
    fake_self = type("FakeSelf", (), {})()
    fake_self.model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    fake_self.logger = _StubLogger()
    fake_self._loaded_checkpoint = None
    return fake_self


def test_setup_training_restores_return_rms_from_checkpoint(monkeypatch):
    monkeypatch.delenv("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_REWARD_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE", raising=False)
    monkeypatch.setenv("PPO_AGENT_PPO_OVERRIDE", json.dumps({"normalize_returns": True}))

    fake_self = _make_fake_self()
    fake_self._loaded_checkpoint = {
        "model_state_dict": fake_self.model.state_dict(),
        "optimizer_state_dict": torch.optim.Adam(fake_self.model.parameters()).state_dict(),
        "return_rms_state": {"mean": 7.5, "var": 3.2, "count": 12345.0},
    }

    train_module.setup_training(fake_self)

    assert fake_self.ppo.return_rms.mean == pytest.approx(7.5)
    assert fake_self.ppo.return_rms.var == pytest.approx(3.2)
    assert fake_self.ppo.return_rms.count == pytest.approx(12345.0)


def test_setup_training_backward_compatible_with_checkpoint_missing_return_rms_key(monkeypatch):
    """Checkpoints without "return_rms_state" load without error and leave
    return_rms at its fresh default.
    """
    monkeypatch.delenv("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_REWARD_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_PPO_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE", raising=False)

    fake_self = _make_fake_self()
    fake_self._loaded_checkpoint = {
        "model_state_dict": fake_self.model.state_dict(),
        "optimizer_state_dict": torch.optim.Adam(fake_self.model.parameters()).state_dict(),
        # No "return_rms_state" key: old-format checkpoint.
    }

    train_module.setup_training(fake_self)  # Must not raise.

    assert fake_self.ppo.return_rms.mean == 0.0
    assert fake_self.ppo.return_rms.var == 1.0
    assert fake_self.ppo.return_rms.count == pytest.approx(1e-4)


def test_save_checkpoint_includes_return_rms_state(tmp_path, monkeypatch):
    monkeypatch.delenv("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_REWARD_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_PPO_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE", raising=False)
    scratch_path = tmp_path / "scratch_checkpoint.pt"
    monkeypatch.setenv("PPO_AGENT_SAVE_CHECKPOINT", str(scratch_path))

    fake_self = _make_fake_self()
    train_module.setup_training(fake_self)
    fake_self.ppo.return_rms.update(np.array([100.0, 200.0, 300.0], dtype=np.float32))

    train_module._save_checkpoint(fake_self)

    saved = torch.load(scratch_path, map_location="cpu")
    assert "return_rms_state" in saved
    assert saved["return_rms_state"]["mean"] == pytest.approx(fake_self.ppo.return_rms.mean)
    assert saved["return_rms_state"]["count"] == pytest.approx(fake_self.ppo.return_rms.count)


def test_save_then_restore_round_trip_via_train_functions(tmp_path, monkeypatch):
    """Non-default return_rms state survives a save via train._save_checkpoint()
    and a load via train.setup_training() into a fresh agent.
    """
    monkeypatch.delenv("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_REWARD_OVERRIDE", raising=False)
    monkeypatch.delenv("PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE", raising=False)
    scratch_path = tmp_path / "scratch_checkpoint.pt"
    monkeypatch.setenv("PPO_AGENT_SAVE_CHECKPOINT", str(scratch_path))
    monkeypatch.setenv("PPO_AGENT_PPO_OVERRIDE", json.dumps({"normalize_returns": True}))

    burst1 = _make_fake_self()
    train_module.setup_training(burst1)
    burst1.ppo.return_rms.update(np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32))
    train_module._save_checkpoint(burst1)
    saved_mean, saved_var, saved_count = (
        burst1.ppo.return_rms.mean, burst1.ppo.return_rms.var, burst1.ppo.return_rms.count,
    )

    # A fresh burst loads the checkpoint back before training resumes.
    burst2 = _make_fake_self()
    burst2.model.load_state_dict(torch.load(scratch_path, map_location="cpu")["model_state_dict"])
    burst2._loaded_checkpoint = torch.load(scratch_path, map_location="cpu")
    train_module.setup_training(burst2)

    assert burst2.ppo.return_rms.mean == pytest.approx(saved_mean)
    assert burst2.ppo.return_rms.var == pytest.approx(saved_var)
    assert burst2.ppo.return_rms.count == pytest.approx(saved_count)
