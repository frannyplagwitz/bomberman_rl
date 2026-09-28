"""1D adapter: 31-dim feature vector.

1 can_move_up, 2 can_move_down, 3 can_move_left, 4 can_move_right,
5 has_reachable_coin, 6 nearest_reachable_coin_distance (normalized by
cfg.DISTANCE_FEATURE_NORM),
7-10 coin_path_up/down/left/right,
11 bomb_available,
12 has_bombing_target, 13 nearest_bombing_position_distance (normalized by
cfg.DISTANCE_FEATURE_NORM),
14-17 bombing_path_up/down/left/right,
18 crates_destructible_at_target (normalized by /12),
19 current_tile_in_danger, 20 nearest_threat_timer (normalized by /4).

21 coin_contested.

22 has_kill_target, 23 nearest_kill_distance (normalized by
cfg.DISTANCE_FEATURE_NORM), 24-27 kill_direction_up/down/left/right, 28
expected_kill_value_at_target (the MAX escape-difficulty score among the
target's covered opponents, already bounded to [0, 1] -- no further
normalization -- see state_processing.kill_target_info()). Parallel to, but
scored independently of, the bombing-target group above: a kill target is
defined purely by whether the tile's blast covers >=1 opponent's CURRENT
position, a snapshot judgment that does not predict whether that opponent
would move out of the blast before it detonates, and is therefore not a
real kill probability.

Opponent proximity is exposed as two undirected features (#29/#30, no
direction bits), independent of has_kill_target/expected_kill_value.

29 nearest_alive_opponent_distance (normalized by cfg.DISTANCE_FEATURE_NORM;
1.0 when no opponent is reachable this way, including no opponents at all),
30 opponents_within_3 (count of alive opponents at that same distance metric
<= cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD, normalized by the same threshold;
0 when none), 31 reachable_space (local mobility around self_pos ignoring
bombs/danger, normalized by cfg.REACHABLE_SPACE_NORM -- see
state_processing.reachable_space_count()).

32 stall_history (optional, config.ENABLE_STALL_HISTORY_FEATURE, default off
-- NOT part of the canonical 31-dim vector, always appended after every
other feature regardless of the base dimension): boolean, reuses
state_processing.is_confined_to_small_range's judgement on the caller-tracked
position history. Passed in by the caller (features_from_semantic doesn't
track history itself, staying a pure function of its inputs), since it
differs by call site (act() vs. train.py's bookkeeping calls) which history
buffer is authoritative -- see callbacks.py/train.py.
"""
from typing import Optional

import numpy as np

from . import config as cfg
from .state_processing import SemanticState, extract_semantic_state


def features_from_semantic(semantic: SemanticState, stall_history: Optional[bool] = None) -> np.ndarray:
    features = np.zeros(cfg.n_features_active(), dtype=np.float32)
    features[0] = float(semantic.can_move["UP"])
    features[1] = float(semantic.can_move["DOWN"])
    features[2] = float(semantic.can_move["LEFT"])
    features[3] = float(semantic.can_move["RIGHT"])
    features[4] = float(semantic.has_reachable_coin)
    # No reachable coin -> both feature 5 and 6 are 0; feature 5
    # disambiguates "distance 0" from "no target".
    # Clipped to 1.0: cfg.DISTANCE_FEATURE_NORM is a normalization cap, not a
    # hard bound -- a real BFS path around walls/crates can exceed it.
    if semantic.has_reachable_coin:
        features[5] = min(semantic.nearest_coin_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    features[6] = float(semantic.coin_path_dirs["UP"])
    features[7] = float(semantic.coin_path_dirs["DOWN"])
    features[8] = float(semantic.coin_path_dirs["LEFT"])
    features[9] = float(semantic.coin_path_dirs["RIGHT"])

    features[10] = float(semantic.bomb_available)
    features[11] = float(semantic.has_bombing_target)
    # Clipped to 1.0 for the same reason as feature 5 above.
    if semantic.has_bombing_target:
        features[12] = min(semantic.nearest_bombing_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    features[13] = float(semantic.bombing_path_dirs["UP"])
    features[14] = float(semantic.bombing_path_dirs["DOWN"])
    features[15] = float(semantic.bombing_path_dirs["LEFT"])
    features[16] = float(semantic.bombing_path_dirs["RIGHT"])
    features[17] = semantic.crates_destructible_at_target / cfg.MAX_CRATES_PER_BOMB

    features[18] = float(semantic.current_tile_in_danger)
    features[19] = semantic.nearest_threat_timer / cfg.BOMB_TIMER

    features[20] = float(semantic.coin_contested)

    features[21] = float(semantic.has_kill_target)
    if semantic.has_kill_target:
        # Clipped to 1.0 for the same reason as feature 5 above.
        features[22] = min(semantic.nearest_kill_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    features[23] = float(semantic.kill_direction_dirs["UP"])
    features[24] = float(semantic.kill_direction_dirs["DOWN"])
    features[25] = float(semantic.kill_direction_dirs["LEFT"])
    features[26] = float(semantic.kill_direction_dirs["RIGHT"])
    features[27] = semantic.expected_kill_value_at_target

    if semantic.nearest_alive_opponent_distance is None:
        features[28] = 1.0
    else:
        features[28] = min(semantic.nearest_alive_opponent_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    features[29] = min(semantic.opponents_within_3 / cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD, 1.0)
    features[30] = min(semantic.reachable_space / cfg.REACHABLE_SPACE_NORM, 1.0)

    if cfg.ENABLE_STALL_HISTORY_FEATURE:
        features[cfg.MODEL_CONFIG.n_features] = float(bool(stall_history))
    return features


def state_to_features(game_state: Optional[dict]) -> Optional[np.ndarray]:
    """Convenience wrapper mirroring the tpl_agent naming convention. Not
    called anywhere in this codebase. Does not track position history, so
    if ENABLE_STALL_HISTORY_FEATURE is on, this always reports
    stall_history=False -- do not use it while that switch is enabled
    without also passing a real history-derived stall_history value.
    """
    if game_state is None:
        return None
    return features_from_semantic(extract_semantic_state(game_state))
