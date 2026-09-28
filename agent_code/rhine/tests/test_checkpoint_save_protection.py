import pytest

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent
from agent_code.rhine.train import _save_checkpoint


def test_unset_save_checkpoint_env_var_raises_without_touching_canonical_path(monkeypatch, tmp_path):
    monkeypatch.delenv("PPO_AGENT_SAVE_CHECKPOINT", raising=False)
    canonical = tmp_path / "task3_stage_a.pt"
    monkeypatch.setattr(cfg, "STAGE_A_CHECKPOINT", canonical)

    # No model/ppo attributes: reading them before the env-var check would
    # raise AttributeError instead of the intended RuntimeError.
    fake_self = object()

    with pytest.raises(RuntimeError, match="PPO_AGENT_SAVE_CHECKPOINT"):
        _save_checkpoint(fake_self)
    assert not canonical.exists()


def test_explicit_save_checkpoint_env_var_writes_to_that_path(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch.pt"
    monkeypatch.setenv("PPO_AGENT_SAVE_CHECKPOINT", str(scratch))
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path)

    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    ppo = PPOAgent(model, cfg.PPO_CONFIG)

    class FakeSelf:
        pass

    fake_self = FakeSelf()
    fake_self.model = model
    fake_self.ppo = ppo

    _save_checkpoint(fake_self)
    assert scratch.is_file()
