import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.ppo import PPOAgent


@pytest.fixture(autouse=True)
def _stage_a_checkpoint_stub(tmp_path_factory, monkeypatch):
    """Provides a throwaway, dimensionally current checkpoint as
    cfg.STAGE_A_CHECKPOINT, since train=False worlds without an explicit
    PPO_AGENT_INIT_CHECKPOINT load it and the file on disk has an older input
    dimension. Only loading is exercised, never the weights.

    Uses tmp_path_factory rather than a test's own tmp_path, because some
    tests assert that their tmp_path stays empty.
    """
    n_features = cfg.n_features_active()
    model = ActorCriticMLP(n_features=n_features, n_actions=cfg.MODEL_CONFIG.n_actions)
    ppo = PPOAgent(model, cfg.PPO_CONFIG)
    stub_path = tmp_path_factory.mktemp("stage_a_checkpoint_stub") / "stage_a_checkpoint_stub.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "return_rms_state": ppo.return_rms.state_dict(),
        },
        stub_path,
    )
    monkeypatch.setattr(cfg, "STAGE_A_CHECKPOINT", stub_path)
