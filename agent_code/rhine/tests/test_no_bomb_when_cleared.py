"""Tests for config.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED: masks out BOMB once no
crates, surviving opponents or collectable coins remain. Default off;
movement/WAIT legality is never affected.
"""
import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import compute_action_mask

# Open room where BOMB has a real escape, so any BOMB=False below comes from
# this feature.
OPEN_ROOM = ["#####", "#...#", "#...#", "#...#", "#####"]
CENTER_POS = (2, 2)


def _game_state(field, coins=(), others=()):
    return {
        "field": field,
        "self": ("me", 0, True, CENTER_POS),
        "coins": list(coins),
        "others": list(others),
    }


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


def test_default_off_bomb_stays_legal_on_cleared_board():
    assert cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED is False  # Stored default.
    field = make_field(OPEN_ROOM)
    mask = compute_action_mask(_game_state(field))
    assert mask[cfg.ACTIONS.index("BOMB")] == True


def test_enabled_masks_bomb_on_fully_cleared_board():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        field = make_field(OPEN_ROOM)
        mask = compute_action_mask(_game_state(field))
        assert mask[cfg.ACTIONS.index("BOMB")] == False
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False


def test_enabled_movement_and_wait_unaffected():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        field = make_field(OPEN_ROOM)
        baseline = compute_action_mask(_game_state(field))
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False
        reference = compute_action_mask(_game_state(field))
        for direction in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
            assert baseline[cfg.ACTIONS.index(direction)] == reference[cfg.ACTIONS.index(direction)]
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False


def test_enabled_bomb_stays_legal_when_crates_remain():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        field = make_field(["#####", "#.X.#", "#...#", "#...#", "#####"])
        mask = compute_action_mask(_game_state(field))
        assert mask[cfg.ACTIONS.index("BOMB")] == True
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False


def test_enabled_bomb_stays_legal_when_opponent_remains():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        # Opponent far away, so the escape check cannot be what masks BOMB.
        field = make_field([
            "###########",
            "#.........#",
            "#.........#",
            "#.........#",
            "#.........#",
            "###########",
        ])
        game_state = _game_state(field, others=[("opp", 0, True, (9, 4))])
        game_state["self"] = ("me", 0, True, (2, 2))
        mask = compute_action_mask(game_state)
        assert mask[cfg.ACTIONS.index("BOMB")] == True
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False


def test_enabled_bomb_stays_legal_when_coin_remains():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        field = make_field(OPEN_ROOM)
        game_state = _game_state(field, coins=[(1, 1)])
        mask = compute_action_mask(game_state)
        assert mask[cfg.ACTIONS.index("BOMB")] == True
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False


def test_enabled_still_masks_bomb_when_unavailable_regardless_of_clear_state():
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    try:
        field = make_field(OPEN_ROOM)
        game_state = {"field": field, "self": ("me", 0, False, CENTER_POS), "coins": [], "others": []}
        mask = compute_action_mask(game_state)
        assert mask[cfg.ACTIONS.index("BOMB")] == False
    finally:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False
