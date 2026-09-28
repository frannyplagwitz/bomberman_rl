"""setup() + act(): extractor -> adapter -> model -> mask -> action.

Checkpoint selection (main.py has no CLI way to pass per-agent args): if
PPO_AGENT_INIT_CHECKPOINT is set, load it regardless of train mode (Stage C
warm start uses this). Otherwise: training -> fresh random init (Stage A);
not training -> try cfg.DEPLOYMENT_CHECKPOINT, else fall back to a fresh
random model with a warning (so `python main.py play --agents rhine` works
even before any training has happened).
"""
import os
from collections import deque
from pathlib import Path

import torch

from . import config as cfg
from .action_mask import apply_oscillation_breaker, mask_from_semantic, should_force_deadlock_bomb
from .features import features_from_semantic
from .model import ActorCriticMLP
from .ppo import select_action
from .state_processing import classify_invalid_action, extract_semantic_state, is_confined_to_small_range


def _load_checkpoint_or_raise(path: Path, expected_n_features: int) -> dict:
    """Loads a checkpoint and fails with an explicit message when its input-layer
    dimension doesn't match the currently configured feature vector size (most
    likely cause: cfg.ENABLE_STALL_HISTORY_FEATURE flipped since training).
    Strict state_dict loading would otherwise raise a bare tensor-shape diff.
    No partial warm-start across a feature-dimension change is implemented.
    """
    checkpoint = torch.load(path, map_location="cpu")
    saved_n_features = checkpoint["model_state_dict"]["trunk.0.weight"].shape[1]
    if saved_n_features != expected_n_features:
        raise RuntimeError(
            f"Checkpoint '{path}' was trained with n_features={saved_n_features}, but the "
            f"current config expects n_features={expected_n_features} "
            f"(cfg.ENABLE_STALL_HISTORY_FEATURE={cfg.ENABLE_STALL_HISTORY_FEATURE}). Partial "
            "warm-start across a feature-dimension change is not implemented; use a checkpoint "
            "trained under the same feature dimension, or retrain from scratch. A pre-Task-4 "
            "28-dim checkpoint needs the code as of git tag 'stage-d-final' (or an earlier stage "
            "tag) to load, not this working tree."
        )
    return checkpoint


def setup(self):
    # Lets `python main.py play` enable config.ENABLE_OSCILLATION_BREAKER
    # without touching main.py; only flips it on for this process, the
    # config default is untouched.
    if os.environ.get("PPO_AGENT_ENABLE_OSCILLATION_BREAKER"):
        cfg.ENABLE_OSCILLATION_BREAKER = True

    if cfg.ENABLE_DEADLOCK_BOMB and not cfg.ENABLE_OSCILLATION_BREAKER:
        self.logger.warning(
            "cfg.ENABLE_DEADLOCK_BOMB is on but cfg.ENABLE_OSCILLATION_BREAKER is off -- "
            "the deadlock-bomb override has no effect without the breaker's confinement trigger."
        )

    n_features = cfg.n_features_active()
    self.model = ActorCriticMLP(
        n_features=n_features,
        n_actions=cfg.MODEL_CONFIG.n_actions,
        hidden_sizes=cfg.MODEL_CONFIG.hidden_sizes,
    )
    # Populated unconditionally (train and eval) so the optional
    # stall-history feature works identically at inference time. Independent
    # of train.py's own recent_positions_v2 (reward-side, training-only).
    self.recent_positions_for_feature = deque(maxlen=4)
    # Separate from recent_positions_for_feature so the oscillation breaker
    # stays independently toggleable via config.ENABLE_OSCILLATION_BREAKER.
    # Only populated in act() when not self.train (see below). maxlen matches
    # apply_oscillation_breaker()'s window_size=7.
    self.recent_positions_for_breaker = deque(maxlen=7)
    # Consecutive-intervention counter carried across act() calls; see
    # apply_oscillation_breaker()'s docstring for the cap it's checked against.
    self.oscillation_breaker_intervention_streak = 0

    # Tracks own-cause vs contested-tile invalid actions (see
    # classify_invalid_action()) by comparing consecutive act() calls'
    # game_state. setup() runs once per Agent lifetime, not per round, so
    # these are reset on step==1 inside act() itself instead.
    self.invalid_action_own_cause_count = 0
    self.invalid_action_contested_tile_count = 0
    self._prev_game_state_for_invalid_check = None
    self._prev_action_for_invalid_check = None

    # Set by act() on every call (train and eval alike) so external callers
    # (e.g. an evaluation script's per-step recording) can read whether this
    # step's action was overridden by config.ENABLE_DEADLOCK_BOMB.
    self.last_deadlock_bomb_triggered = False

    self._loaded_checkpoint = None
    checkpoint_override = os.environ.get("PPO_AGENT_INIT_CHECKPOINT")
    if checkpoint_override:
        path = Path(checkpoint_override)
        self.logger.info(f"Loading model from checkpoint override: {path}")
        checkpoint = _load_checkpoint_or_raise(path, n_features)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._loaded_checkpoint = checkpoint
    elif self.train:
        self.logger.info("Training mode, no checkpoint override: initializing model from scratch.")
    else:
        if cfg.DEPLOYMENT_CHECKPOINT.is_file():
            self.logger.info(f"Loading default deployment checkpoint: {cfg.DEPLOYMENT_CHECKPOINT}")
            checkpoint = _load_checkpoint_or_raise(cfg.DEPLOYMENT_CHECKPOINT, n_features)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._loaded_checkpoint = checkpoint
        else:
            self.logger.warning("No checkpoint found and not training: using a randomly initialized model.")

    self.model.train(mode=self.train)
    self.deterministic = not self.train


def act(self, game_state: dict) -> str:
    if game_state is None:
        return "WAIT"

    if game_state.get("step") == 1:
        self.invalid_action_own_cause_count = 0
        self.invalid_action_contested_tile_count = 0
        self._prev_game_state_for_invalid_check = None
        self._prev_action_for_invalid_check = None
    elif self._prev_game_state_for_invalid_check is not None:
        outcome = classify_invalid_action(
            self._prev_game_state_for_invalid_check, self._prev_action_for_invalid_check, game_state
        )
        if outcome == "own_cause":
            self.invalid_action_own_cause_count += 1
        elif outcome == "contested_tile":
            self.invalid_action_contested_tile_count += 1

    semantic = extract_semantic_state(game_state)

    stall_history = None
    if cfg.ENABLE_STALL_HISTORY_FEATURE:
        self.recent_positions_for_feature.append(semantic.self_pos)
        stall_history = is_confined_to_small_range(self.recent_positions_for_feature)

    features = features_from_semantic(semantic, stall_history=stall_history)
    mask = mask_from_semantic(semantic)

    # `not self.train` guarantees this never reaches a training rollout even
    # if the flag were left on by mistake: train.py computes its own mask
    # directly and never calls apply_oscillation_breaker().
    self.last_deadlock_bomb_triggered = False
    if cfg.ENABLE_OSCILLATION_BREAKER and not self.train:
        self.recent_positions_for_breaker.append(semantic.self_pos)
        mask, self.oscillation_breaker_intervention_streak = apply_oscillation_breaker(
            mask, self.recent_positions_for_breaker, self.oscillation_breaker_intervention_streak
        )
        if cfg.ENABLE_DEADLOCK_BOMB and should_force_deadlock_bomb(
            semantic, mask, self.recent_positions_for_breaker
        ):
            self.last_deadlock_bomb_triggered = True

    action_idx, log_prob, value = select_action(self.model, features, mask, deterministic=self.deterministic)
    if self.last_deadlock_bomb_triggered:
        action_idx = cfg.ACTIONS.index("BOMB")
    self.last_log_prob = log_prob
    self.last_value = value

    chosen_action = cfg.ACTIONS[action_idx]
    self._prev_game_state_for_invalid_check = game_state
    self._prev_action_for_invalid_check = chosen_action
    return chosen_action
