"""Centralized configuration for the PPO agent.

All tunable numbers used across the agent modules live here rather than
being hard-coded at each call site.
"""
from dataclasses import dataclass
from pathlib import Path

import settings as s

AGENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = AGENT_DIR / "models"

# Stage A training save path; not read at inference time -- inference loads
# DEPLOYMENT_CHECKPOINT instead.
STAGE_A_CHECKPOINT = MODELS_DIR / "task3_stage_a.pt"
STAGE_C_CHECKPOINT = MODELS_DIR / "task2_stage_c.pt"  # unused, never read anywhere.

# Checkpoint loaded by setup() when not training and no
# PPO_AGENT_INIT_CHECKPOINT override is set.
DEPLOYMENT_CHECKPOINT = MODELS_DIR / "task4_seed0.pt"

# Fixed action order, used everywhere feature/mask/model index <-> action string.
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]

# Distance normalized by (ROWS-1)+(COLS-1), i.e. 32 on the fixed 17x17 board.
# Geometric board-diameter bound only (used as a "genuinely far" sanity bound
# in tests) -- NOT the distance-feature normalization divisor; see
# DISTANCE_FEATURE_NORM below for that.
MAX_BOARD_DISTANCE = (s.ROWS - 1) + (s.COLS - 1)

# Normalization divisor (clipped to 1.0 above it) for the three BFS-distance
# features (nearest_reachable_coin_distance, nearest_bombing_position_distance,
# nearest_kill_distance). Single point of definition, referenced from
# features.py only.
DISTANCE_FEATURE_NORM = 16

# Distance cutoff and count-normalization divisor for
# nearest_alive_opponent_distance/opponents_within_3, both 3 by definition.
OPPONENTS_WITHIN_RANGE_THRESHOLD = 3

# BFS step cap and normalization divisor for the reachable_space feature
# (see state_processing.reachable_space_count()).
REACHABLE_SPACE_DEPTH_CAP = 6
REACHABLE_SPACE_NORM = 40

# Bomb blast radius and fuse length, read from the framework's own settings
# so they always match environment.py's rules.
BOMB_POWER = s.BOMB_POWER
BOMB_TIMER = s.BOMB_TIMER
# 4 directions x BOMB_POWER tiles each, upper bound if every tile were a crate.
MAX_CRATES_PER_BOMB = 4 * s.BOMB_POWER

# bombing_target_info() filters candidates to only those with a safe escape
# route. Default off: measurably hurt oscillation with little
# completion-rate benefit. A driver script can flip this on for comparison.
ENABLE_BOMBING_TARGET_SAFETY_FILTER: bool = False

# bombing_target_info() selects the safety-filtered candidate maximizing
# crates_destructible / (distance + 1), instead of simply the nearest one.
# Default on; a driver script can flip this off to reproduce the original
# nearest-distance selection.
ENABLE_BOMBING_TARGET_SCORING_FORMULA: bool = True

# Appends one extra boolean feature (reusing
# state_processing.is_confined_to_small_range) marking whether the agent has
# been confined to a small area recently, since the otherwise stateless
# feature vector cannot condition on how long it has been stuck. Changes the
# input dimension from n_features to n_features+1 -- incompatible with
# checkpoints saved at the base dimension. Default off.
ENABLE_STALL_HISTORY_FEATURE: bool = False

# Evaluation/deployment-only action-mask refinement; see
# action_mask.apply_oscillation_breaker() for the trigger logic and
# callbacks.py's act() for how it's wired in (gated on `not self.train`, so
# it never fires during a training rollout). Default on; a driver script can
# flip this off per run.
#
# Known limitation: validated only against static hazards (no moving
# opponents). The trigger is a pure position-history pattern and cannot
# distinguish looping without reason from a legitimate reversal caused by a
# moving opponent blocking the forward tile. Re-evaluate before enabling
# once opponents move.
ENABLE_OSCILLATION_BREAKER: bool = True

# Removes BOMB from the legal-action mask whenever no crates, no surviving
# opponents, and no collectable coins remain -- see action_mask.py's
# _board_fully_cleared() for the exact trigger check. Does not touch
# movement/WAIT legality or any other mask logic. Default off.
ENABLE_NO_BOMB_WHEN_BOARD_CLEARED: bool = False

# Evaluation-only: when the oscillation breaker's own confinement trigger
# holds, no crates/collectable coins remain, and the current tile is itself
# a kill target (nearest_kill_distance == 0), overrides the model's chosen
# action to BOMB -- see action_mask.should_force_deadlock_bomb(). Only takes
# effect when ENABLE_OSCILLATION_BREAKER is also on and not self.train; does
# not bypass the mask (BOMB must already be legal). Default off.
ENABLE_DEADLOCK_BOMB: bool = False


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping weights and toggles for training."""
    COIN_REWARD: float = 1.0
    STEP_COST: float = -0.01
    PROGRESS_WEIGHT: float = 0.10  # Task 1 only; not read by Task 2's reward module.
    CRATE_DESTROYED_REWARD: float = 0.1
    SELF_KILL_PENALTY: float = -2.0
    # Kept distinct from SELF_KILL_PENALTY: being killed by an opponent also
    # hands them the kill reward, widening the score gap beyond a plain
    # self-kill (see rewards.py's death-cause split). Decoupled from the
    # displayed/reported score, which reads settings.py directly.
    TRAINING_KILLED_OPPONENT_REWARD: float = 3.0
    TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY: float = -3.0
    BOMBING_PROGRESS_SHAPING_WEIGHT: float = 0.10
    # True: unified progress shaping toward the current coin/bombing target.
    ENABLE_BOMBING_PROGRESS_SHAPING: bool = True
    # Penalizes staying at a valid bombing position (BOMB legal, distance 0)
    # without pressing BOMB or genuinely moving away, once train.py's
    # stall_counter exceeds this many consecutive steps.
    STALL_THRESHOLD: int = 10
    STALL_PENALTY: float = -0.05
    # False: superseded by ENABLE_STALL_PENALTY_V2 -- this mechanism was
    # found structurally ineffective.
    ENABLE_STALL_PENALTY: bool = False
    # Triggers on being confined to <=2 distinct tiles within the last 4
    # bomb_available=True steps, independent of nearest_bombing_distance and
    # independent of the STALL_PENALTY mechanism above. Default on.
    STALL_PENALTY_V2: float = -0.05
    ENABLE_STALL_PENALTY_V2: bool = True
    # Higher CRATE_DESTROYED_REWARD when no reachable coin exists at the
    # time of the crate-destroying transition. Default off.
    CRATE_DESTROYED_REWARD_NO_COIN: float = 0.3
    ENABLE_CRATE_NO_COIN_BONUS: bool = False
    # Applied on a BOMB placement that destroys zero crates and has no
    # reachable opponent in its blast radius either -- see rewards.py.
    WASTEFUL_BOMB_PENALTY: float = -0.2
    ENABLE_WASTEFUL_BOMB_PENALTY: bool = True


@dataclass(frozen=True)
class PPOConfig:
    """Standard PPO defaults (Stable-Baselines3 / original PPO paper conventions)."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    # ~10 rounds' worth of data per PPO update; larger rollouts reduce
    # per-update sample variance without real multi-environment
    # parallelism. Total compute for a fixed total-timesteps budget is
    # unaffected -- this only changes how work is chunked into updates.
    rollout_steps: int = 4096
    minibatch_size: int = 64
    update_epochs: int = 10
    normalize_advantage: bool = True
    # Optional linear decay of entropy_coef and learning_rate over training
    # progress (steps_so_far / total_timesteps). Default off so existing
    # configs are unaffected.
    enable_entropy_lr_decay: bool = False
    entropy_coef_end: float = 0.001
    learning_rate_end: float = 3e-5
    # Only consulted when enable_entropy_lr_decay=True; normally overridden
    # per-run to match that run's actual --total-timesteps.
    total_timesteps: int = 500_000
    # Running-mean/std normalization of the value-loss target only; GAE,
    # advantages, policy_loss, explained_variance, and the bootstrap value all
    # keep the raw scale (see ppo.py PPOAgent.update()). Default on. The
    # running statistics are checkpointed so they survive PPOAgent
    # re-creation across training bursts.
    normalize_returns: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """2 layers x 64 units; actor/critic share a trunk with two heads."""
    hidden_sizes: tuple = (64, 64)
    n_features: int = 31  # base/canonical dim; see n_features_active() for the effective dim.
    n_actions: int = 6


REWARD_CONFIG = RewardConfig()
PPO_CONFIG = PPOConfig()
MODEL_CONFIG = ModelConfig()


def n_features_active() -> int:
    """Effective feature-vector / model input dimension, accounting for the
    optional stall-history feature. Read at call time (not baked into
    MODEL_CONFIG at import time) so ENABLE_STALL_HISTORY_FEATURE can be
    flipped per training run without editing this file's on-disk default.
    """
    return MODEL_CONFIG.n_features + (1 if ENABLE_STALL_HISTORY_FEATURE else 0)
