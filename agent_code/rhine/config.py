"""Centralized configuration for the PPO agent.

All tunable numbers used across the agent modules live here rather than
being hard-coded at each call site.
"""
from dataclasses import dataclass
from pathlib import Path

import settings as s

AGENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = AGENT_DIR / "models"

# Stage A training save path. Kept separate from DEPLOYMENT_CHECKPOINT so a
# from-scratch training run never overwrites the submission weights.
STAGE_A_CHECKPOINT = MODELS_DIR / "task3_stage_a.pt"
STAGE_C_CHECKPOINT = MODELS_DIR / "task2_stage_c.pt"  # unused, never read anywhere.

# Checkpoint loaded at inference when no PPO_AGENT_INIT_CHECKPOINT override is set.
DEPLOYMENT_CHECKPOINT = MODELS_DIR / "task4_seed0.pt"

# Fixed action order, used everywhere feature/mask/model index <-> action string.
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]

# Geometric board-diameter bound; not the distance-feature divisor (see
# DISTANCE_FEATURE_NORM).
MAX_BOARD_DISTANCE = (s.ROWS - 1) + (s.COLS - 1)

# Normalization divisor (clipped to 1.0) for the BFS-distance features
# nearest_reachable_coin_distance, nearest_bombing_position_distance and
# nearest_kill_distance.
DISTANCE_FEATURE_NORM = 16

# Distance cutoff and count-normalization divisor for
# nearest_alive_opponent_distance/opponents_within_3.
OPPONENTS_WITHIN_RANGE_THRESHOLD = 3

# BFS step cap and normalization divisor for the reachable_space feature
# (see state_processing.reachable_space_count()).
REACHABLE_SPACE_DEPTH_CAP = 6
REACHABLE_SPACE_NORM = 40

# Bomb blast radius and fuse length, read from the framework's own settings
# so they always match environment.py's rules.
BOMB_POWER = s.BOMB_POWER
BOMB_TIMER = s.BOMB_TIMER
# Upper bound: every tile in all four blast arms is a crate.
MAX_CRATES_PER_BOMB = 4 * s.BOMB_POWER

# Restricts bombing_target_info() candidates to those with a safe escape
# route. Default off: it hurt oscillation with little completion-rate benefit.
ENABLE_BOMBING_TARGET_SAFETY_FILTER: bool = False

# bombing_target_info() picks the candidate maximizing
# crates_destructible / (distance + 1) instead of the nearest one.
ENABLE_BOMBING_TARGET_SCORING_FORMULA: bool = True

# Appends one boolean feature marking recent confinement to a small area
# (state_processing.is_confined_to_small_range). Increases the input
# dimension by one, so it is incompatible with base-dimension checkpoints.
ENABLE_STALL_HISTORY_FEATURE: bool = False

# Evaluation-only mask refinement (action_mask.apply_oscillation_breaker());
# never active during training rollouts.
#
# Known limitation: the trigger is a pure position-history pattern and cannot
# tell aimless looping from a legitimate reversal forced by a moving opponent.
ENABLE_OSCILLATION_BREAKER: bool = True

# Removes BOMB from the mask once no crates, surviving opponents or
# collectable coins remain (action_mask._board_fully_cleared()).
ENABLE_NO_BOMB_WHEN_BOARD_CLEARED: bool = False

# Evaluation-only: when the oscillation breaker's confinement trigger holds,
# no crates/coins remain and the current tile is a kill target, overrides the
# chosen action with BOMB (action_mask.should_force_deadlock_bomb()). Requires
# ENABLE_OSCILLATION_BREAKER; never bypasses the mask.
ENABLE_DEADLOCK_BOMB: bool = False


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping weights and toggles for training."""
    COIN_REWARD: float = 1.0
    STEP_COST: float = -0.01
    PROGRESS_WEIGHT: float = 0.10  # Task 1 only; not read by Task 2's reward module.
    CRATE_DESTROYED_REWARD: float = 0.1
    SELF_KILL_PENALTY: float = -2.0
# Kept separate from SELF_KILL_PENALTY: being killed by an opponent also hands
# them the kill reward. Training-only; reported scores use settings.py.
    TRAINING_KILLED_OPPONENT_REWARD: float = 3.0
    TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY: float = -3.0
    BOMBING_PROGRESS_SHAPING_WEIGHT: float = 0.10
# Unified progress shaping toward the current coin/bombing target.
    ENABLE_BOMBING_PROGRESS_SHAPING: bool = True
# Penalizes idling at a valid bombing position without bombing once
# train.py's stall_counter exceeds STALL_THRESHOLD.
    STALL_THRESHOLD: int = 10
    STALL_PENALTY: float = -0.05
# V1 stall penalty; superseded by ENABLE_STALL_PENALTY_V2.
    ENABLE_STALL_PENALTY: bool = False
# Penalizes confinement to a few tiles while a bomb is available,
# independent of nearest_bombing_distance and of the V1 mechanism.
    STALL_PENALTY_V2: float = -0.05
    ENABLE_STALL_PENALTY_V2: bool = True
# Higher crate reward when no reachable coin exists at destruction time.
    CRATE_DESTROYED_REWARD_NO_COIN: float = 0.3
    ENABLE_CRATE_NO_COIN_BONUS: bool = False
# Applied to a BOMB that destroys no crate and threatens no reachable
# opponent (see rewards.py).
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
# Larger rollouts reduce per-update sample variance; total compute for a
# fixed timestep budget is unchanged.
    rollout_steps: int = 4096
    minibatch_size: int = 64
    update_epochs: int = 10
    normalize_advantage: bool = True
# Optional linear decay of entropy_coef and learning_rate over training
# progress (steps_so_far / total_timesteps).
    enable_entropy_lr_decay: bool = False
    entropy_coef_end: float = 0.001
    learning_rate_end: float = 3e-5
# Only used when enable_entropy_lr_decay=True; normally overridden per run.
    total_timesteps: int = 500_000
# Running mean/std normalization of the value-loss target only; GAE,
# advantages and bootstrap values stay on the raw scale. The running
# statistics are saved in checkpoints.
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
    """Effective model input dimension, including the optional stall-history
    feature. Read at call time so ENABLE_STALL_HISTORY_FEATURE can be flipped
    per run.
    """
    return MODEL_CONFIG.n_features + (1 if ENABLE_STALL_HISTORY_FEATURE else 0)
