"""6-action mask.

Movement: blocked tiles (wall/crate/bomb) are illegal; open tiles are legal
unless moving there would strand the agent with no safe escape before a
known bomb/explosion reaches it.
BOMB: illegal if unavailable; otherwise legal only if a safe escape route
exists from the current position before this bomb's own blast (checked
against currently-known bombs/explosions, not bombs placed later).
WAIT: illegal if staying at the current tile guarantees a hit and at least
one other action is legal. Kept as a fallback when no action would survive.
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
    """Trigger check for config.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED: no crates,
    no surviving opponents, and no collectable coins remain anywhere on the
    board (not just out of reach).
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
        # Moving happens as part of resolving this action, so safety is
        # checked at offset 0, not one step later.
        neighbor = neighbor_tile(semantic.self_pos, d)
        neighbor_by_direction[d] = neighbor
        mask[cfg.ACTIONS.index(d)] = exists_safe_path(
            neighbor, 0, semantic.field_arr, semantic.blocked, semantic.danger_offsets, SAFETY_HORIZON
        )

    # If self_pos is within an active bomb's blast reach, re-check each
    # already-legal direction for >=N+1 independent, conflict-disjoint
    # escape routes (see _has_sufficient_escape_directions()), N being the
    # number of opponents whose current BFS distance would let them reach
    # some tile on a candidate path in time. A direction failing this
    # stricter check is masked out only if another legal direction passes.
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

    # Tie-break among remaining legal movement directions: prefer whichever
    # landing tile no opponent could also reach next step (real
    # current-position BFS distance <=1) -- guards against a same-tick
    # collision race turning an already-chosen legal move into
    # INVALID_ACTION. Only filters when a genuinely uncontested alternative
    # exists and there's an actual choice to make (>=2 legal directions) --
    # a preference among already-legal options, not a legality change.
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

    # WAIT means staying at self_pos through the next tick (offset 1), not
    # landing on a new tile the way movement/BOMB do. The offset-0 danger
    # check is kept explicit because exists_safe_path(self_pos, 1, ...) alone
    # would skip verifying that the current instant isn't already lethal.
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


# Oscillation breaker: opt-in, evaluation/deployment-only refinement layered
# on top of an already-computed mask. Gated behind
# config.ENABLE_OSCILLATION_BREAKER and `not self.train`, so it never
# affects a training rollout.
#
# Known limitation: validated only against static (opponent-free) hazards;
# cannot distinguish looping without reason from a legitimate reversal
# caused by a moving opponent blocking the forward tile.
OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS = 6  # one less than the
# confinement window (7 steps, see is_confined_to_small_range call below), so
# a multi-step in-place wait (e.g. for the agent's own bomb to explode) has
# already broken confinement again well before this cap could ever bind.


def apply_oscillation_breaker(
    mask: np.ndarray,
    position_history,
    intervention_streak: int = 0,
) -> "tuple[np.ndarray, int]":
    """Mask out reversing to the previous tile when the agent is confined to
    <=2 tiles over the last 7 steps (see is_confined_to_small_range), unless
    doing so would leave no legal action at all (retreating is the only way
    out -- a genuine dead end).

    Unlike an earlier version of this function, this does not require the
    direction opposite the retreat to be legal (a straight-through passage);
    any other still-legal action -- including a side branch perpendicular to
    the direction the agent arrived from -- is enough to filter the retreat.

    `intervention_streak` is the count of consecutive prior calls (while
    confinement held) that actually filtered a retreat; once it reaches
    OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS this call stops
    intervening (returns the mask unchanged) even though confinement still
    holds, so it can't indefinitely suppress a legitimate reason to hold
    position. Returns (mask, updated_streak); the caller carries the updated
    streak into its next call and resets it to 0 itself once this function
    reports a non-confined state.
    """
    if not is_confined_to_small_range(position_history, window_size=7):
        return mask, 0
    positions = list(position_history)
    if len(positions) < 2:
        return mask, 0
    current, previous = positions[-1], positions[-2]
    if previous == current:
        # Last action was WAIT/BOMB (no actual movement) -- no "previous
        # tile" to retreat from this step.
        return mask, intervention_streak

    direction_to_previous = next(
        (d for d in DIRECTIONS if neighbor_tile(current, d) == previous), None
    )
    if direction_to_previous is None:
        # `previous` isn't actually adjacent to `current` (e.g. history not
        # reset across rounds); nothing sensible to mask in that case.
        return mask, intervention_streak

    idx_back = cfg.ACTIONS.index(direction_to_previous)
    if not mask[idx_back]:
        return mask, intervention_streak

    candidate_mask = mask.copy()
    candidate_mask[idx_back] = False
    # "Way out" means an alternative *movement* direction, not BOMB/WAIT
    # (WAIT in particular is legal almost everywhere via mask_from_semantic's
    # fallback, which would otherwise defeat this dead-end guard entirely).
    movement_indices = [cfg.ACTIONS.index(d) for d in DIRECTIONS]
    if not candidate_mask[movement_indices].any():
        # Retreating is the only legal movement direction -- a genuine dead end.
        return mask, intervention_streak

    if intervention_streak >= OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS:
        return mask, intervention_streak

    return candidate_mask, intervention_streak + 1


def should_force_deadlock_bomb(
    semantic: SemanticState,
    mask: np.ndarray,
    position_history,
) -> bool:
    """config.ENABLE_DEADLOCK_BOMB's trigger: does the current step warrant
    overriding the model's chosen action to BOMB? All of the following must
    hold:
    - the oscillation breaker's own confinement trigger holds (same
      is_confined_to_small_range check, window_size=7, apply_oscillation_
      breaker() uses -- independent of whether it actually masked anything
      this step);
    - no crates remain anywhere on the board;
    - no collectable coins remain anywhere on the board;
    - the current tile is itself a kill target (has_kill_target and
      nearest_kill_distance == 0);
    - BOMB is still legal in `mask` (this never bypasses the mask).

    Does not touch the mask itself and does not affect apply_oscillation_
    breaker()'s own retreat-masking behavior -- a caller applies this after
    action selection, only for this one step. A two-tile confinement where
    neither tile is a kill target never satisfies this (has_kill_target
    fails), so that deadlock is left exactly as apply_oscillation_breaker()
    already handles it.
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
