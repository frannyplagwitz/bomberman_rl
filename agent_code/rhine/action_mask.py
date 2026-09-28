"""6-action legality mask.

Movement: blocked tiles (wall/crate/bomb) are illegal; open tiles are legal
unless moving there leaves no safe escape from known bombs/explosions.
BOMB: legal only if available and a safe escape from its own blast exists,
given currently known bombs/explosions.
WAIT: illegal if staying is guaranteed lethal and another action is legal;
kept as a fallback when nothing else is legal.
"""
import numpy as np

from . import config as cfg
from .state_processing import (
    DIRECTIONS,
    SAFETY_HORIZON,
    SemanticState,
    _has_sufficient_escape_directions,
    _opponent_distance_maps,
    exists_safe_path,
    extract_semantic_state,
    has_safe_escape_after_bombing,
    is_confined_to_small_range,
    neighbor_tile,
)



def _board_fully_cleared(semantic: SemanticState) -> bool:
    """Trigger for config.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED: no crates,
    surviving opponents or collectable coins remain anywhere on the board.
    """
    return (
        not np.any(semantic.field_arr == 1)
        and not semantic.opponents
        and semantic.coins_remaining == 0
    )


def mask_from_semantic(semantic: SemanticState) -> np.ndarray:
    mask = np.zeros(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    neighbor_by_direction = {}

    for d in DIRECTIONS:
        if not semantic.can_move[d]:
            continue
        # The move resolves within this action, so safety starts at offset 0.
        neighbor = neighbor_tile(semantic.self_pos, d)
        neighbor_by_direction[d] = neighbor
        mask[cfg.ACTIONS.index(d)] = exists_safe_path(
            neighbor, 0, semantic.field_arr, semantic.blocked, semantic.danger_offsets, SAFETY_HORIZON
        )

    # When standing in a blast zone with opponents around, keep only the
    # directions that retain enough independent escape routes given nearby
    # opponents (_has_sufficient_escape_directions()), unless none do.
    redundancy_trigger = semantic.current_tile_in_danger and bool(semantic.opponents)
    if redundancy_trigger:
        robust_by_direction = {
            d: _has_sufficient_escape_directions(
                neighbor, semantic.field_arr, semantic.blocked, semantic.danger_offsets, semantic.opponents,
                start_offset=0,
            )
            for d, neighbor in neighbor_by_direction.items()
            if mask[cfg.ACTIONS.index(d)]
        }
        if any(robust_by_direction.values()):
            for d, is_robust in robust_by_direction.items():
                if not is_robust:
                    mask[cfg.ACTIONS.index(d)] = False

    # Among remaining legal directions, prefer landing tiles no opponent can
    # reach next step, avoiding a same-tick collision that turns the move into
    # INVALID_ACTION. Only applies when an uncontested alternative exists.
    if semantic.current_tile_in_danger and semantic.opponents:
        currently_legal_dirs = [d for d in neighbor_by_direction if mask[cfg.ACTIONS.index(d)]]
        if len(currently_legal_dirs) > 1:
            opponent_dist_maps = _opponent_distance_maps(semantic.field_arr, semantic.blocked, semantic.opponents)
            uncontested = [
                d for d in currently_legal_dirs
                if not any(dmap.get(neighbor_by_direction[d], float("inf")) <= 1 for dmap in opponent_dist_maps.values())
            ]
            if uncontested:
                for d in currently_legal_dirs:
                    if d not in uncontested:
                        mask[cfg.ACTIONS.index(d)] = False

    if semantic.bomb_available:
        mask[cfg.ACTIONS.index("BOMB")] = has_safe_escape_after_bombing(
            semantic.self_pos, semantic.field_arr, semantic.blocked, semantic.danger_offsets, cfg.BOMB_POWER,
            semantic.opponents,
        )
    else:
        mask[cfg.ACTIONS.index("BOMB")] = False

    if cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED and _board_fully_cleared(semantic):
        mask[cfg.ACTIONS.index("BOMB")] = False

    # WAIT keeps the agent on self_pos through the next tick (offset 1); the
    # offset-0 check covers the current instant, which exists_safe_path skips.
    wait_safe = (
        0 not in semantic.danger_offsets.get(semantic.self_pos, ())
        and exists_safe_path(
            semantic.self_pos, 1, semantic.field_arr, semantic.blocked, semantic.danger_offsets, SAFETY_HORIZON
        )
    )
    other_action_legal = bool(mask[:5].any())  # UP/DOWN/LEFT/RIGHT/BOMB, i.e. everything but WAIT
    mask[cfg.ACTIONS.index("WAIT")] = wait_safe or not other_action_legal
    return mask


def compute_action_mask(game_state: dict) -> np.ndarray:
    return mask_from_semantic(extract_semantic_state(game_state))


# Evaluation-only refinement layered on an already-computed mask
# (config.ENABLE_OSCILLATION_BREAKER, and never during training).
#
# Known limitation: cannot tell aimless looping from a legitimate reversal
# forced by a moving opponent blocking the forward tile.
# Kept below the confinement window length so a legitimate in-place wait
# breaks confinement before the cap binds.
OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS = 6


def apply_oscillation_breaker(
    mask: np.ndarray,
    position_history,
    intervention_streak: int = 0,
) -> "tuple[np.ndarray, int]":
    """Masks out stepping back to the previous tile while the agent is
    confined to a small range (is_confined_to_small_range), unless that
    leaves no other movement direction (a genuine dead end).

    Args:
        mask: already-computed legality mask.
        position_history: recent positions, oldest first.
        intervention_streak: consecutive prior calls that filtered a retreat;
            at OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS the breaker
            stops intervening so it cannot suppress a legitimate hold forever.

    Returns:
        (mask, updated_streak); the streak resets to 0 once not confined.
    """
    if not is_confined_to_small_range(position_history, window_size=7):
        return mask, 0
    positions = list(position_history)
    if len(positions) < 2:
        return mask, 0
    current, previous = positions[-1], positions[-2]
    if previous == current:
        # No movement last step, so there is no tile to retreat to.
        return mask, intervention_streak

    direction_to_previous = next(
        (d for d in DIRECTIONS if neighbor_tile(current, d) == previous), None
    )
    if direction_to_previous is None:
        # Non-adjacent history (e.g. not reset across rounds).
        return mask, intervention_streak

    idx_back = cfg.ACTIONS.index(direction_to_previous)
    if not mask[idx_back]:
        return mask, intervention_streak

    candidate_mask = mask.copy()
    candidate_mask[idx_back] = False
    # Only movement counts as a way out: WAIT is almost always legal via the
    # mask's fallback and would otherwise defeat the dead-end guard.
    movement_indices = [cfg.ACTIONS.index(d) for d in DIRECTIONS]
    if not candidate_mask[movement_indices].any():
        # Genuine dead end.
        return mask, intervention_streak

    if intervention_streak >= OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS:
        return mask, intervention_streak

    return candidate_mask, intervention_streak + 1


def should_force_deadlock_bomb(
    semantic: SemanticState,
    mask: np.ndarray,
    position_history,
) -> bool:
    """Whether to override the chosen action with BOMB
    (config.ENABLE_DEADLOCK_BOMB). All of the following must hold:
    - the breaker's confinement trigger holds (whether or not it masked
      anything this step);
    - no crates and no collectable coins remain on the board;
    - the current tile is itself a kill target;
    - BOMB is legal in `mask` (the mask is never bypassed).

    The caller applies this after action selection; the mask is not modified.
    """
    if not is_confined_to_small_range(position_history, window_size=7):
        return False
    if np.any(semantic.field_arr == 1):
        return False
    if semantic.coins_remaining != 0:
        return False
    if not (semantic.has_kill_target and semantic.nearest_kill_distance == 0):
        return False
    return bool(mask[cfg.ACTIONS.index("BOMB")])
