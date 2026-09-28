"""Feature adapter: SemanticState -> 31-dim vector (1-based numbering).

1-4   can_move_up/down/left/right
5     has_reachable_coin
6     nearest_reachable_coin_distance (/ cfg.DISTANCE_FEATURE_NORM)
7-10  coin_path_up/down/left/right
11    bomb_available
12    has_bombing_target
13    nearest_bombing_position_distance (/ cfg.DISTANCE_FEATURE_NORM)
14-17 bombing_path_up/down/left/right
18    crates_destructible_at_target (/ cfg.MAX_CRATES_PER_BOMB)
19    current_tile_in_danger
20    nearest_threat_timer (/ cfg.BOMB_TIMER)
21    coin_contested
22    has_kill_target
23    nearest_kill_distance (/ cfg.DISTANCE_FEATURE_NORM)
24-27 kill_direction_up/down/left/right
28    expected_kill_value_at_target, already in [0, 1] (see
      state_processing.kill_target_info()). A kill target is a tile whose
      blast covers an opponent's current position; it is a snapshot, not a
      kill probability.
29    nearest_alive_opponent_distance (/ cfg.DISTANCE_FEATURE_NORM; 1.0 when
      no opponent is reachable)
30    opponents_within_3 (count within cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD,
      normalized by the same threshold)
31    reachable_space (local mobility ignoring bombs/danger,
      / cfg.REACHABLE_SPACE_NORM; see state_processing.reachable_space_count())
32    stall_history (optional, cfg.ENABLE_STALL_HISTORY_FEATURE): appended
      after the base vector. Supplied by the caller, since the authoritative
      position history differs between act() and train.py.
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
    # has_reachable_coin disambiguates "distance 0" from "no target". Distances are
    # clipped because a BFS path around walls/crates can exceed the divisor.
    if semantic.has_reachable_coin:
        features[5] = min(semantic.nearest_coin_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    features[6] = float(semantic.coin_path_dirs["UP"])
    features[7] = float(semantic.coin_path_dirs["DOWN"])
    features[8] = float(semantic.coin_path_dirs["LEFT"])
    features[9] = float(semantic.coin_path_dirs["RIGHT"])

    features[10] = float(semantic.bomb_available)
    features[11] = float(semantic.has_bombing_target)
    # Clipped like the coin distance above.
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
        # Clipped like the coin distance above.
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
    """Convenience wrapper using the tpl_agent naming; unused in this codebase.
    Tracks no position history, so the optional stall-history feature is
    always False here.
    """
    if game_state is None:
        return None
    return features_from_semantic(extract_semantic_state(game_state))
