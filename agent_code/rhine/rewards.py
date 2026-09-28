"""Reward calculation.

reward = coin_reward + crate_destroyed_reward + bombing_progress_shaping (optional)
         + self_kill_penalty + got_killed_by_opponent_penalty + killed_opponent_reward
         + step_cost + stall_penalty + wasteful_bomb_penalty (optional)

Kill terms: e.GOT_KILLED fires for every death, so a death caused by an
opponent is GOT_KILLED without KILLED_SELF; e.KILLED_OPPONENT is counted
directly. Their magnitudes are training-only and independent of the
reported score.

`stall_penalty` (ENABLE_STALL_PENALTY): applied every step once train.py's
stall_counter (idling at a valid bombing position without reaching a new
tile) exceeds STALL_THRESHOLD, until the agent bombs or moves away.

`crate_no_coin_bonus` (ENABLE_CRATE_NO_COIN_BONUS): uses
CRATE_DESTROYED_REWARD_NO_COIN when old_game_state has no reachable coin.

`stall_penalty_v2` (ENABLE_STALL_PENALTY_V2): independent of stall_penalty;
triggered by train.py's _update_stall_v2 (confinement while a bomb is
available) and passed in as `stall_v2_triggered`.

`wasteful_bomb_penalty` (ENABLE_WASTEFUL_BOMB_PENALTY): applied on
BOMB_DROPPED when the blast destroys no crate and covers no opponent
(bomb_threatens_reachable_opponent()). Defensive/denial placements are not
distinguished.

`bombing_progress_shaping`: potential Phi(s) = -d(s), d being the BFS
distance to the nearest reachable coin, else to the nearest bombing
position. Skipped when:
  - events include COIN_COLLECTED, COIN_FOUND, CRATE_DESTROYED,
    BOMB_DROPPED or BOMB_EXPLODED (target or graph changed); or
  - the agent stands on its own bomb in old_game_state (bfs_distances()
    always treats the start tile as free, so old/new distances would use
    different graphs).

compute_reward() takes the raw transition so train.py stays agnostic of what
a reward design needs.
"""
from typing import List, Optional

import events as e

from . import config as cfg
from .state_processing import (
    _bomb_positions,
    _to_coord,
    bomb_threatens_reachable_opponent,
    crates_in_blast,
    nearest_bombing_position_distance,
    nearest_reachable_coin_distance,
)

_TARGET_SWITCH_EVENTS = (
    e.COIN_COLLECTED,
    e.COIN_FOUND,
    e.CRATE_DESTROYED,
    e.BOMB_DROPPED,
    e.BOMB_EXPLODED,
)


def _target_distance(game_state: dict) -> Optional[int]:
    coin_distance = nearest_reachable_coin_distance(game_state)
    if coin_distance is not None:
        return coin_distance
    return nearest_bombing_position_distance(game_state, cfg.BOMB_POWER)


def compute_reward(
    old_game_state: dict,
    self_action: str,
    new_game_state: Optional[dict],
    events_list: List[str],
    reward_config: Optional[cfg.RewardConfig] = None,
    stall_counter: int = 0,
    stall_v2_triggered: bool = False,
) -> float:
    reward_config = reward_config or cfg.REWARD_CONFIG

    reward = reward_config.STEP_COST
    reward += events_list.count(e.COIN_COLLECTED) * reward_config.COIN_REWARD

    crate_reward = reward_config.CRATE_DESTROYED_REWARD
    if reward_config.ENABLE_CRATE_NO_COIN_BONUS:
        if nearest_reachable_coin_distance(old_game_state) is None:
            crate_reward = reward_config.CRATE_DESTROYED_REWARD_NO_COIN
    reward += events_list.count(e.CRATE_DESTROYED) * crate_reward

    if e.KILLED_SELF in events_list:
        reward += reward_config.SELF_KILL_PENALTY
    # Isolates deaths caused by an opponent from self-kills.
    if e.GOT_KILLED in events_list and e.KILLED_SELF not in events_list:
        reward += reward_config.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY
    reward += events_list.count(e.KILLED_OPPONENT) * reward_config.TRAINING_KILLED_OPPONENT_REWARD
    if reward_config.ENABLE_STALL_PENALTY and stall_counter > reward_config.STALL_THRESHOLD:
        reward += reward_config.STALL_PENALTY
    if reward_config.ENABLE_STALL_PENALTY_V2 and stall_v2_triggered:
        reward += reward_config.STALL_PENALTY_V2

    if reward_config.ENABLE_WASTEFUL_BOMB_PENALTY and e.BOMB_DROPPED in events_list:
        bomb_pos = _to_coord(old_game_state["self"][3])
        if (
            crates_in_blast(old_game_state["field"], bomb_pos, cfg.BOMB_POWER) == 0
            and not bomb_threatens_reachable_opponent(old_game_state, bomb_pos, cfg.BOMB_POWER)
        ):
            reward += reward_config.WASTEFUL_BOMB_PENALTY

    if reward_config.ENABLE_BOMBING_PROGRESS_SHAPING:
        target_switched = any(ev in events_list for ev in _TARGET_SWITCH_EVENTS)
        standing_on_own_bomb = _to_coord(old_game_state["self"][3]) in _bomb_positions(old_game_state)
        # new_game_state is None on the terminal-transition fallback.
        if not target_switched and not standing_on_own_bomb and new_game_state is not None:
            old_distance = _target_distance(old_game_state)
            new_distance = _target_distance(new_game_state)
            if old_distance is not None and new_distance is not None:
                reward += reward_config.BOMBING_PROGRESS_SHAPING_WEIGHT * (old_distance - new_distance)

    return reward
