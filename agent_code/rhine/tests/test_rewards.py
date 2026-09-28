import dataclasses

import numpy as np
import events as e

from agent_code.rhine import config as cfg
from agent_code.rhine.rewards import compute_reward


def make_field(rows_pattern):
    height = len(rows_pattern)
    width = len(rows_pattern[0])
    field = np.zeros((width, height), dtype=int)
    for y, row in enumerate(rows_pattern):
        for x, ch in enumerate(row):
            if ch == "#":
                field[x, y] = -1
            elif ch == "X":
                field[x, y] = 1
    return field


def make_state(field, pos, coins=(), bombs=(), others=()):
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "coins": list(coins),
        "bombs": list(bombs),
        "others": [("opp", 0, True, opp_pos) for opp_pos in others],
        "explosion_map": np.zeros_like(field, dtype=float),
    }


# Explicit historical baseline (shaping off, V1 stall penalty on, V2 off),
# independent of config.py's current defaults.
_PLAIN_BASELINE = dataclasses.replace(
    cfg.REWARD_CONFIG,
    ENABLE_BOMBING_PROGRESS_SHAPING=False, ENABLE_STALL_PENALTY=True, ENABLE_STALL_PENALTY_V2=False,
)
B1 = _PLAIN_BASELINE
B2 = dataclasses.replace(_PLAIN_BASELINE, ENABLE_BOMBING_PROGRESS_SHAPING=True)


def test_step_cost_only_on_plain_move():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (2, 1))
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT], B1)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_coin_collected_reward():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1), coins=[(2, 1)])
    new_state = make_state(field, (2, 1), coins=[])
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT, e.COIN_COLLECTED], B1)
    assert reward == cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.COIN_REWARD


def test_multiple_crates_destroyed_reward_scales_with_count():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    events_list = ["BOMB_EXPLODED", e.CRATE_DESTROYED, e.CRATE_DESTROYED, e.CRATE_DESTROYED]
    reward = compute_reward(old_state, "WAIT", new_state, events_list, B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + 3 * cfg.REWARD_CONFIG.CRATE_DESTROYED_REWARD
    assert reward == expected


def test_self_kill_penalty_applied_once():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "WAIT", None, [e.KILLED_SELF, "GOT_KILLED"], B1)
    assert reward == cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.SELF_KILL_PENALTY


def test_got_killed_by_opponent_penalty_applied_when_not_self_kill():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "WAIT", None, [e.GOT_KILLED], B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY
    assert abs(reward - expected) < 1e-9


def test_got_killed_by_opponent_penalty_not_applied_when_also_self_kill():
    # GOT_KILLED accompanies every death; with KILLED_SELF only the self-kill
    # penalty applies, not the opponent-kill penalty on top.
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "WAIT", None, [e.KILLED_SELF, e.GOT_KILLED], B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.SELF_KILL_PENALTY
    assert abs(reward - expected) < 1e-9


def test_killed_opponent_reward_scales_with_count():
    # One bomb can kill several opponents, each adding its own KILLED_OPPONENT.
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    events_list = ["BOMB_EXPLODED", e.KILLED_OPPONENT, e.KILLED_OPPONENT]
    reward = compute_reward(old_state, "WAIT", new_state, events_list, B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + 2 * cfg.REWARD_CONFIG.TRAINING_KILLED_OPPONENT_REWARD
    assert abs(reward - expected) < 1e-9


def test_stall_penalty_not_applied_at_or_below_threshold():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    reward = compute_reward(
        old_state, "WAIT", new_state, [e.WAITED], B1, stall_counter=cfg.REWARD_CONFIG.STALL_THRESHOLD
    )
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_stall_penalty_applied_above_threshold():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    reward = compute_reward(
        old_state, "WAIT", new_state, [e.WAITED], B1, stall_counter=cfg.REWARD_CONFIG.STALL_THRESHOLD + 1
    )
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.STALL_PENALTY
    assert abs(reward - expected) < 1e-9


def test_stall_penalty_defaults_to_zero_when_unspecified():
    # Calls without stall_counter (terminal fallback) must not apply the penalty.
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "WAIT", new_state, [e.WAITED], B1)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_stall_penalty_disabled_via_toggle():
    # ENABLE_STALL_PENALTY=False suppresses the penalty even far past the threshold.
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    no_stall = dataclasses.replace(_PLAIN_BASELINE, ENABLE_STALL_PENALTY=False)
    reward = compute_reward(
        old_state, "WAIT", new_state, [e.WAITED], no_stall,
        stall_counter=cfg.REWARD_CONFIG.STALL_THRESHOLD + 1,
    )
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_crate_no_coin_bonus_disabled_by_default():
    field = make_field(["########", "#....X.#", "########"])
    old_state = make_state(field, (1, 1))  # No coins; the default config uses the flat reward.
    new_state = make_state(field, (2, 1))
    events_list = [e.MOVED_RIGHT, "BOMB_EXPLODED", e.CRATE_DESTROYED]
    reward = compute_reward(old_state, "RIGHT", new_state, events_list, B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.CRATE_DESTROYED_REWARD
    assert abs(reward - expected) < 1e-9


def test_crate_no_coin_bonus_applies_when_no_coin_reachable():
    field = make_field(["########", "#....X.#", "########"])
    old_state = make_state(field, (1, 1))  # No coins.
    new_state = make_state(field, (2, 1))
    events_list = [e.MOVED_RIGHT, "BOMB_EXPLODED", e.CRATE_DESTROYED]
    bonus_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_CRATE_NO_COIN_BONUS=True)
    reward = compute_reward(old_state, "RIGHT", new_state, events_list, bonus_cfg)
    expected = cfg.REWARD_CONFIG.STEP_COST + bonus_cfg.CRATE_DESTROYED_REWARD_NO_COIN
    assert abs(reward - expected) < 1e-9


def test_crate_no_coin_bonus_not_applied_when_coin_reachable():
    field = make_field(["########", "#....X.#", "########"])
    # The coin lies before the crate, so it is reachable without crossing it.
    old_state = make_state(field, (1, 1), coins=[(3, 1)])
    new_state = make_state(field, (2, 1), coins=[(3, 1)])
    events_list = [e.MOVED_RIGHT, "BOMB_EXPLODED", e.CRATE_DESTROYED]
    bonus_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_CRATE_NO_COIN_BONUS=True)
    reward = compute_reward(old_state, "RIGHT", new_state, events_list, bonus_cfg)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.CRATE_DESTROYED_REWARD
    assert abs(reward - expected) < 1e-9


def test_stall_v2_disabled_by_default():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "WAIT", new_state, [e.WAITED], B1, stall_v2_triggered=True)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_stall_v2_applies_when_enabled_and_triggered():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    v2_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_STALL_PENALTY_V2=True)
    reward = compute_reward(old_state, "WAIT", new_state, [e.WAITED], v2_cfg, stall_v2_triggered=True)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.STALL_PENALTY_V2
    assert abs(reward - expected) < 1e-9


def test_stall_v2_not_applied_when_triggered_false():
    field = make_field(["#####", "#...#", "#####"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    v2_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_STALL_PENALTY_V2=True)
    reward = compute_reward(old_state, "WAIT", new_state, [e.WAITED], v2_cfg, stall_v2_triggered=False)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_wasteful_bomb_penalty_enabled_by_default():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    reward = compute_reward(old_state, "BOMB", new_state, [e.BOMB_DROPPED], B1)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.WASTEFUL_BOMB_PENALTY
    assert abs(reward - expected) < 1e-9


def test_wasteful_bomb_penalty_can_be_disabled():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    disabled_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_WASTEFUL_BOMB_PENALTY=False)
    reward = compute_reward(old_state, "BOMB", new_state, [e.BOMB_DROPPED], disabled_cfg)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_wasteful_bomb_penalty_not_applied_when_crate_in_blast():
    field = make_field(["#######", "#..X..#", "#######"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    penalty_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_WASTEFUL_BOMB_PENALTY=True)
    reward = compute_reward(old_state, "BOMB", new_state, [e.BOMB_DROPPED], penalty_cfg)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_wasteful_bomb_penalty_applied_when_no_crate_and_no_opponent_value():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1))
    penalty_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_WASTEFUL_BOMB_PENALTY=True)
    reward = compute_reward(old_state, "BOMB", new_state, [e.BOMB_DROPPED], penalty_cfg)
    expected = cfg.REWARD_CONFIG.STEP_COST + penalty_cfg.WASTEFUL_BOMB_PENALTY
    assert abs(reward - expected) < 1e-9


def test_wasteful_bomb_penalty_not_applied_when_reachable_opponent_in_blast():
    # No crate, but a reachable opponent inside the blast gives the bomb value.
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1), others=[(4, 1)])
    new_state = make_state(field, (1, 1), others=[(4, 1)])
    penalty_cfg = dataclasses.replace(_PLAIN_BASELINE, ENABLE_WASTEFUL_BOMB_PENALTY=True)
    reward = compute_reward(old_state, "BOMB", new_state, [e.BOMB_DROPPED], penalty_cfg)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_b1_never_applies_shaping():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1), coins=[(5, 1)])
    new_state = make_state(field, (2, 1), coins=[(5, 1)])
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT], B1)
    assert reward == cfg.REWARD_CONFIG.STEP_COST


def test_b2_applies_positive_shaping_when_approaching_coin():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (1, 1), coins=[(5, 1)])
    new_state = make_state(field, (2, 1), coins=[(5, 1)])
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT], B2)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.BOMBING_PROGRESS_SHAPING_WEIGHT * 1
    assert abs(reward - expected) < 1e-9


def test_b2_skips_shaping_on_coin_collected_transition():
    field = make_field(["#######", "#.....#", "#######"])
    old_state = make_state(field, (4, 1), coins=[(5, 1)])
    new_state = make_state(field, (5, 1), coins=[])
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT, e.COIN_COLLECTED], B2)
    # COIN_REWARD + STEP_COST only: no progress term after the coin disappears.
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.COIN_REWARD
    assert abs(reward - expected) < 1e-9


def test_b2_skips_shaping_on_crate_destroyed_transition():
    field = make_field(["########", "#....X.#", "########"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (2, 1))
    events_list = [e.MOVED_RIGHT, "BOMB_EXPLODED", e.CRATE_DESTROYED]
    reward = compute_reward(old_state, "RIGHT", new_state, events_list, B2)
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.CRATE_DESTROYED_REWARD
    assert abs(reward - expected) < 1e-9


def test_b2_skips_shaping_on_bomb_dropped_transition():
    # Disable the wasteful-bomb penalty to isolate the progress-shaping skip.
    local_cfg = dataclasses.replace(B2, ENABLE_WASTEFUL_BOMB_PENALTY=False)
    field = make_field(["########", "#....X.#", "########"])
    old_state = make_state(field, (1, 1))
    new_state = make_state(field, (1, 1), bombs=[((1, 1), 3)])
    events_list = [e.BOMB_DROPPED]
    reward = compute_reward(old_state, "BOMB", new_state, events_list, local_cfg)
    assert abs(reward - cfg.REWARD_CONFIG.STEP_COST) < 1e-9


def test_b2_skips_shaping_on_bomb_exploded_transition():
    field = make_field(["########", "#....X.#", "########"])
    old_state = make_state(field, (1, 1), bombs=[((1, 1), 0)])
    new_state = make_state(field, (1, 1))
    events_list = ["BOMB_EXPLODED"]
    reward = compute_reward(old_state, "WAIT", new_state, events_list, B2)
    assert abs(reward - cfg.REWARD_CONFIG.STEP_COST) < 1e-9


def test_b2_skips_shaping_when_standing_on_own_bomb():
    # Ring corridor: once the agent steps off its own bomb, the nearby coin is
    # reachable only via a long detour. The standing-on-own-bomb check must
    # skip shaping for this transition.
    field = make_field([
        "#########",
        "#.......#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.......#",
        "#########",
    ])
    old_state = make_state(field, (4, 1), coins=[(6, 1)], bombs=[((4, 1), 3)])
    new_state = make_state(field, (3, 1), coins=[(6, 1)], bombs=[((4, 1), 2)])
    reward = compute_reward(old_state, "LEFT", new_state, [e.MOVED_LEFT], B2)
    assert abs(reward - cfg.REWARD_CONFIG.STEP_COST) < 1e-9


def test_b2_targets_bombing_position_when_no_coin_reachable():
    # Open room with one crate, where a safe bombing target exists.
    field = make_field(["#####", "#.X.#", "#...#", "#...#", "#####"])
    old_state = make_state(field, (1, 3))
    new_state = make_state(field, (2, 3))
    reward = compute_reward(old_state, "RIGHT", new_state, [e.MOVED_RIGHT], B2)
    # One ordinary step closer to the bombing target.
    expected = cfg.REWARD_CONFIG.STEP_COST + cfg.REWARD_CONFIG.BOMBING_PROGRESS_SHAPING_WEIGHT * 1
    assert abs(reward - expected) < 1e-9


def test_b2_shaping_magnitude_bounded_for_plain_step_with_unchanged_graph():
    # With unchanged obstacles, no target switch and one ordinary step, shaping
    # changes by at most weight * 1. The ring-corridor case would break this
    # bound if the standing-on-own-bomb skip were removed.
    plain_field = make_field(["#######", "#.....#", "#.....#", "#.....#", "#######"])
    plain_old = make_state(plain_field, (1, 1), coins=[(5, 3)])
    plain_new = make_state(plain_field, (2, 1), coins=[(5, 3)])

    ring_field = make_field([
        "#########",
        "#.......#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.......#",
        "#########",
    ])
    ring_old = make_state(ring_field, (4, 1), coins=[(6, 1)], bombs=[((4, 1), 3)])
    ring_new = make_state(ring_field, (3, 1), coins=[(6, 1)], bombs=[((4, 1), 2)])

    cases = [
        (plain_old, plain_new, "RIGHT", [e.MOVED_RIGHT]),
        (ring_old, ring_new, "LEFT", [e.MOVED_LEFT]),
    ]
    for old_state, new_state, action, events_list in cases:
        reward = compute_reward(old_state, action, new_state, events_list, B2)
        shaping = reward - cfg.REWARD_CONFIG.STEP_COST
        assert abs(shaping) <= cfg.REWARD_CONFIG.BOMBING_PROGRESS_SHAPING_WEIGHT * 1 + 1e-9
