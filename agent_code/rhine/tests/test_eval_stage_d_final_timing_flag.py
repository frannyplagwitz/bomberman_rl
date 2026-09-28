"""Covers eval_stage_d_final.py's --disable-think-time-limit flag: it defaults
to False (real per-step budget) and is persisted in the output's
meta.disable_think_time_limit. n_rounds=0 exercises real argument parsing
without running any game loop.
"""
import gzip
import pickle
import sys

import torch

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent
from agent_code.rhine.scripts import eval_stage_d_final


def _make_stub_checkpoint(tmp_path):
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    ppo = PPOAgent(model, cfg.PPO_CONFIG)
    path = tmp_path / "stub.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "return_rms_state": ppo.return_rms.state_dict(),
        },
        path,
    )
    return path


def _run_and_load_meta(tmp_path, monkeypatch, extra_args):
    # main() changes these config flags without restoring them; register the
    # current values with monkeypatch so they are restored at teardown.
    monkeypatch.setattr(cfg, "ENABLE_OSCILLATION_BREAKER", cfg.ENABLE_OSCILLATION_BREAKER)
    monkeypatch.setattr(cfg, "ENABLE_NO_BOMB_WHEN_BOARD_CLEARED", cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED)
    monkeypatch.setattr(cfg, "ENABLE_DEADLOCK_BOMB", cfg.ENABLE_DEADLOCK_BOMB)

    checkpoint = _make_stub_checkpoint(tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "eval_stage_d_final.py",
        "--checkpoint", str(checkpoint),
        "--group", "B",
        "--out-dir", str(out_dir),
        "--n-rounds", "0",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    eval_stage_d_final.main()
    [pkl_path] = list(out_dir.glob("*.pkl.gz"))
    with gzip.open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return payload["meta"]


def test_disable_think_time_limit_defaults_to_false(tmp_path, monkeypatch):
    meta = _run_and_load_meta(tmp_path, monkeypatch, extra_args=[])
    assert meta["disable_think_time_limit"] is False


def test_disable_think_time_limit_explicit_flag_sets_true(tmp_path, monkeypatch):
    meta = _run_and_load_meta(tmp_path, monkeypatch, extra_args=["--disable-think-time-limit"])
    assert meta["disable_think_time_limit"] is True


def test_no_disable_think_time_limit_flag_sets_false(tmp_path, monkeypatch):
    meta = _run_and_load_meta(tmp_path, monkeypatch, extra_args=["--no-disable-think-time-limit"])
    assert meta["disable_think_time_limit"] is False
