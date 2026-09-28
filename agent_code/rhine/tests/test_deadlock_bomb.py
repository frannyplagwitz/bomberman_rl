"""Tests for config.ENABLE_DEADLOCK_BOMB: action_mask.should_force_deadlock_bomb()
and its wiring into callbacks.act().
"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.rhine import callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic, should_force_deadlock_bomb
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import extract_semantic_state


def make_field(rows_pattern):
    """'#'=wall, 'X'=crate, '.'=free."""
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


# Open room with self next to an opponent: self_pos is a kill target with a
# safe escape; no crates, no coins.
_ROOM = make_field([
    "#############",
    "#...........#",
    "#...........#",
    "#...........#",
    "#...........#",
    "#...........#",
    "#############",
])
_SELF_POS = (6, 3)
_OPP_POS = (7, 3)


def _all_conditions_met_state():
    game_state = {
        "field": _ROOM, "self": ("me", 0, True, _SELF_POS), "coins": [],
        "others": [("opp", 0, True, _OPP_POS)], "step": 10,
    }
    semantic = extract_semantic_state(game_state)
    mask = mask_from_semantic(semantic)
    return semantic, mask, game_state


def _confined_history():
    return deque([_SELF_POS] * 7, maxlen=7)


# --- 1. should_force_deadlock_bomb(): trigger-condition correctness ---

def test_all_conditions_met_triggers():
    semantic, mask, _ = _all_conditions_met_state()
    assert semantic.has_kill_target is True
    assert semantic.nearest_kill_distance == 0
    assert semantic.coins_remaining == 0
    assert not np.any(semantic.field_arr == 1)
    assert mask[cfg.ACTIONS.index("BOMB")]
    assert should_force_deadlock_bomb(semantic, mask, _confined_history()) is True


def test_no_trigger_when_history_not_confined():
    semantic, mask, _ = _all_conditions_met_state()
    # One entry short of the confinement window.
    short_history = deque([_SELF_POS] * 6, maxlen=7)
    assert should_force_deadlock_bomb(semantic, mask, short_history) is False


def test_no_trigger_when_crates_remain():
    field_with_crate = make_field([
        "#############",
        "#...........#",
        "#...........#",
        "#..X........#",
        "#...........#",
        "#...........#",
        "#############",
    ])
    game_state = {
        "field": field_with_crate, "self": ("me", 0, True, _SELF_POS), "coins": [],
        "others": [("opp", 0, True, _OPP_POS)], "step": 10,
    }
    semantic = extract_semantic_state(game_state)
    mask = mask_from_semantic(semantic)
    assert should_force_deadlock_bomb(semantic, mask, _confined_history()) is False


def test_no_trigger_when_coins_remain():
    game_state = {
        "field": _ROOM, "self": ("me", 0, True, _SELF_POS), "coins": [(2, 2)],
        "others": [("opp", 0, True, _OPP_POS)], "step": 10,
    }
    semantic = extract_semantic_state(game_state)
    mask = mask_from_semantic(semantic)
    assert should_force_deadlock_bomb(semantic, mask, _confined_history()) is False


def test_no_trigger_when_not_at_a_kill_target():
    # Opponent outside self_pos's blast.
    game_state = {
        "field": _ROOM, "self": ("me", 0, True, (2, 3)), "coins": [],
        "others": [("opp", 0, True, _OPP_POS)], "step": 10,
    }
    semantic = extract_semantic_state(game_state)
    mask = mask_from_semantic(semantic)
    assert semantic.nearest_kill_distance != 0
    history = deque([(2, 3)] * 7, maxlen=7)
    assert should_force_deadlock_bomb(semantic, mask, history) is False


def test_no_trigger_when_bomb_not_legal_in_mask():
    semantic, mask, _ = _all_conditions_met_state()
    mask = mask.copy()
    mask[cfg.ACTIONS.index("BOMB")] = False
    assert should_force_deadlock_bomb(semantic, mask, _confined_history()) is False


# --- 2. Wiring into callbacks.act() ---

class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _make_agent_self(train: bool):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    self_obj = SimpleNamespace(train=train, logger=_NullLogger())
    callbacks.setup(self_obj)
    return self_obj


def test_act_overrides_to_bomb_when_all_conditions_met():
    _, _, game_state = _all_conditions_met_state()
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_DEADLOCK_BOMB = True
    try:
        agent_self = _make_agent_self(train=False)
        action = None
        for _ in range(7):
            action = callbacks.act(agent_self, game_state)
        assert action == "BOMB"
        assert agent_self.last_deadlock_bomb_triggered is True
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True
        cfg.ENABLE_DEADLOCK_BOMB = False


def test_act_never_triggers_in_train_mode():
    _, _, game_state = _all_conditions_met_state()
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_DEADLOCK_BOMB = True
    try:
        agent_self = _make_agent_self(train=True)
        for _ in range(7):
            callbacks.act(agent_self, game_state)
        assert agent_self.last_deadlock_bomb_triggered is False
        assert len(agent_self.recent_positions_for_breaker) == 0
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True
        cfg.ENABLE_DEADLOCK_BOMB = False


def test_act_never_triggers_when_breaker_disabled():
    _, _, game_state = _all_conditions_met_state()
    cfg.ENABLE_OSCILLATION_BREAKER = False
    cfg.ENABLE_DEADLOCK_BOMB = True
    try:
        agent_self = _make_agent_self(train=False)
        for _ in range(7):
            callbacks.act(agent_self, game_state)
        assert agent_self.last_deadlock_bomb_triggered is False
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True
        cfg.ENABLE_DEADLOCK_BOMB = False


def test_act_default_off_never_triggers():
    _, _, game_state = _all_conditions_met_state()
    assert cfg.ENABLE_DEADLOCK_BOMB is False  # Stored default.
    cfg.ENABLE_OSCILLATION_BREAKER = True
    try:
        agent_self = _make_agent_self(train=False)
        for _ in range(7):
            callbacks.act(agent_self, game_state)
        assert agent_self.last_deadlock_bomb_triggered is False
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True
