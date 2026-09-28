import numpy as np

from agent_code.rhine.state_processing import classify_invalid_action


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


def make_state(field, pos, bomb_available=True, coins=(), bombs=(), others=()):
    return {
        "field": field,
        "self": ("me", 0, bomb_available, pos),
        "coins": list(coins),
        "bombs": list(bombs),
        "others": [("opp", 0, True, opp_pos) for opp_pos in others],
        "explosion_map": np.zeros_like(field, dtype=float),
    }


FIELD = make_field(["#####", "#...#", "#...#", "#...#", "#####"])


def test_successful_move_is_none():
    prev = make_state(FIELD, (2, 2))
    new = make_state(FIELD, (3, 2))
    assert classify_invalid_action(prev, "RIGHT", new) is None


def test_wait_is_never_invalid():
    prev = make_state(FIELD, (2, 2))
    new = make_state(FIELD, (2, 2))
    assert classify_invalid_action(prev, "WAIT", new) is None


def test_own_cause_wall():
    prev = make_state(FIELD, (1, 2))
    new = make_state(FIELD, (1, 2))
    assert classify_invalid_action(prev, "LEFT", new) == "own_cause"


def test_own_cause_bomb_occupied_target():
    prev = make_state(FIELD, (2, 2), bombs=[((3, 2), 3)])
    new = make_state(FIELD, (2, 2), bombs=[((3, 2), 2)])
    assert classify_invalid_action(prev, "RIGHT", new) == "own_cause"


def test_own_cause_opponent_occupied_target():
    prev = make_state(FIELD, (2, 2), others=[(3, 2)])
    new = make_state(FIELD, (2, 2), others=[(3, 2)])
    assert classify_invalid_action(prev, "RIGHT", new) == "own_cause"


def test_own_cause_bomb_unavailable():
    prev = make_state(FIELD, (2, 2), bomb_available=False)
    new = make_state(FIELD, (2, 2), bomb_available=False)
    assert classify_invalid_action(prev, "BOMB", new) == "own_cause"


def test_bomb_available_and_placed_is_none():
    prev = make_state(FIELD, (2, 2), bomb_available=True)
    new = make_state(FIELD, (2, 2), bomb_available=False, bombs=[((2, 2), 4)])
    assert classify_invalid_action(prev, "BOMB", new) is None


def test_contested_tile_target_free_in_prev_state():
    # Target free at decision time, yet the move failed: another agent took it
    # the same tick.
    prev = make_state(FIELD, (2, 2))
    new = make_state(FIELD, (2, 2))
    assert classify_invalid_action(prev, "RIGHT", new) == "contested_tile"
