"""Effective runtime configuration of a submission run with no CLI args and no
env vars: breaker on, deadlock/no-bomb-when-cleared off, 31-dim features,
masked argmax, weights from cfg.DEPLOYMENT_CHECKPOINT. Every test goes
through callbacks.setup() with all PPO_AGENT_* env vars cleared, like
`python main.py play`. cfg.STAGE_A_CHECKPOINT is never read on this path.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from agent_code.rhine import callbacks
from agent_code.rhine import config as cfg

ENV_VARS = [
    "PPO_AGENT_INIT_CHECKPOINT",
    "PPO_AGENT_SAVE_CHECKPOINT",
    "PPO_AGENT_ENABLE_OSCILLATION_BREAKER",
    "PPO_AGENT_ROLLOUT_STEPS_OVERRIDE",
    "PPO_AGENT_REWARD_OVERRIDE",
]

RUNTIME_FILES = [
    "config.py", "state_processing.py", "features.py", "action_mask.py",
    "model.py", "ppo.py", "callbacks.py",
]


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def info(self, msg, *a, **k):
        self.infos.append(msg)

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)


def _clear_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _make_agent_self():
    return SimpleNamespace(train=False, logger=_RecordingLogger())


def test_zero_config_switches(monkeypatch):
    _clear_env(monkeypatch)

    agent_self = _make_agent_self()
    callbacks.setup(agent_self)

    assert cfg.ENABLE_OSCILLATION_BREAKER is True
    assert cfg.ENABLE_DEADLOCK_BOMB is False
    assert cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED is False
    assert cfg.n_features_active() == 31
    assert agent_self.deterministic is True


def test_zero_config_loads_task4_seed0_checkpoint(monkeypatch):
    _clear_env(monkeypatch)

    agent_self = _make_agent_self()
    callbacks.setup(agent_self)

    expected = torch.load(cfg.DEPLOYMENT_CHECKPOINT, map_location="cpu")["model_state_dict"]
    actual = agent_self.model.state_dict()
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        assert torch.equal(actual[key], expected[key]), f"tensor mismatch at {key}"
    assert not any("randomly initialized" in w for w in agent_self.logger.warnings)


def test_zero_config_loads_task4_seed0_checkpoint_independent_of_cwd(monkeypatch, tmp_path):
    """cfg.DEPLOYMENT_CHECKPOINT is resolved from config.py's location, not
    the process cwd.
    """
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    agent_self = _make_agent_self()
    callbacks.setup(agent_self)

    expected = torch.load(cfg.DEPLOYMENT_CHECKPOINT, map_location="cpu")["model_state_dict"]
    actual = agent_self.model.state_dict()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), f"tensor mismatch at {key}"


def test_zero_config_falls_back_to_random_init_when_deployment_checkpoint_absent(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setattr(cfg, "DEPLOYMENT_CHECKPOINT", tmp_path / "does_not_exist.pt")

    agent_self = _make_agent_self()
    callbacks.setup(agent_self)

    assert any("randomly initialized" in w for w in agent_self.logger.warnings)


# --- No hardcoded absolute paths in the 7 runtime files ---

_ABS_PATH_RE = re.compile(r"""["'](/[^"'\n]*/[^"'\n]*)["']""")


def test_runtime_files_contain_no_hardcoded_absolute_paths():
    agent_dir = Path(__file__).resolve().parent.parent
    offenders = []
    for name in RUNTIME_FILES:
        text = (agent_dir / name).read_text()
        for match in _ABS_PATH_RE.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    assert not offenders, f"hardcoded absolute path literal(s) found: {offenders}"
