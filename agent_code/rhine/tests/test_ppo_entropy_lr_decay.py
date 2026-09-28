import dataclasses

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent


def _make_agent(ppo_config, steps_already_trained=0):
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    return PPOAgent(model, ppo_config, steps_already_trained=steps_already_trained), n_features


def _store_dummy_steps(agent, n_features, count):
    mask = np.ones(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    features = np.zeros(n_features, dtype=np.float32)
    for _ in range(count):
        agent.store(features, mask, 0, 0.0, 0.0, 0.0, done=True)


def test_decay_disabled_leaves_entropy_coef_and_lr_unchanged():
    base = dataclasses.replace(
        cfg.PPO_CONFIG, rollout_steps=4, enable_entropy_lr_decay=False,
        entropy_coef=0.01, entropy_coef_end=0.001, learning_rate=3e-4, learning_rate_end=3e-5,
        total_timesteps=8,
    )
    agent, n_features = _make_agent(base)
    initial_lr = agent.optimizer.param_groups[0]["lr"]
    assert initial_lr == 3e-4

    for _ in range(2):  # Well past total_timesteps if decay were on.
        _store_dummy_steps(agent, n_features, 4)
        result = agent.update(last_value=0.0)
        assert result["entropy_coef_used"] == base.entropy_coef
        assert result["learning_rate_used"] == base.learning_rate
        assert agent.optimizer.param_groups[0]["lr"] == base.learning_rate


def test_decay_enabled_linear_interpolation_at_50_and_100_percent():
    base = dataclasses.replace(
        cfg.PPO_CONFIG, rollout_steps=4, enable_entropy_lr_decay=True,
        entropy_coef=0.01, entropy_coef_end=0.001, learning_rate=3e-4, learning_rate_end=3e-5,
        total_timesteps=8,
    )
    agent, n_features = _make_agent(base)

    # First update lands at the midpoint of the decay schedule.
    _store_dummy_steps(agent, n_features, 4)
    result = agent.update(last_value=0.0)
    expected_entropy_mid = (base.entropy_coef + base.entropy_coef_end) / 2
    expected_lr_mid = (base.learning_rate + base.learning_rate_end) / 2
    assert abs(result["entropy_coef_used"] - expected_entropy_mid) < 1e-12
    assert abs(result["learning_rate_used"] - expected_lr_mid) < 1e-12
    assert abs(agent.optimizer.param_groups[0]["lr"] - expected_lr_mid) < 1e-12

    # Second update reaches exactly the end values.
    _store_dummy_steps(agent, n_features, 4)
    result = agent.update(last_value=0.0)
    assert abs(result["entropy_coef_used"] - base.entropy_coef_end) < 1e-12
    assert abs(result["learning_rate_used"] - base.learning_rate_end) < 1e-12


def test_steps_already_trained_offset_is_applied():
    base = dataclasses.replace(
        cfg.PPO_CONFIG, rollout_steps=4, enable_entropy_lr_decay=True,
        entropy_coef=0.01, entropy_coef_end=0.001, learning_rate=3e-4, learning_rate_end=3e-5,
        total_timesteps=8,
    )
    # Steps from a prior burst count toward progress, so decay is complete immediately.
    agent, n_features = _make_agent(base, steps_already_trained=4)
    _store_dummy_steps(agent, n_features, 4)
    result = agent.update(last_value=0.0)
    assert abs(result["entropy_coef_used"] - base.entropy_coef_end) < 1e-12
    assert abs(result["learning_rate_used"] - base.learning_rate_end) < 1e-12


def test_progress_beyond_total_timesteps_is_clamped_to_end_values():
    base = dataclasses.replace(
        cfg.PPO_CONFIG, rollout_steps=4, enable_entropy_lr_decay=True,
        entropy_coef=0.01, entropy_coef_end=0.001, learning_rate=3e-4, learning_rate_end=3e-5,
        total_timesteps=2,  # Fewer than the steps actually stored.
    )
    agent, n_features = _make_agent(base)
    _store_dummy_steps(agent, n_features, 4)
    result = agent.update(last_value=0.0)
    assert abs(result["entropy_coef_used"] - base.entropy_coef_end) < 1e-12
    assert abs(result["learning_rate_used"] - base.learning_rate_end) < 1e-12
