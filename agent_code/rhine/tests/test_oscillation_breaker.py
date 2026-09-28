"""Tests for action_mask.apply_oscillation_breaker() and its wiring into
callbacks.act() via config.ENABLE_OSCILLATION_BREAKER.

Covers the trigger condition, retreat filtering vs dead ends (on concrete
maps), per-call independence, the intervention cap, default-off behavior, and
that the mechanism never reaches the training path (a train=True act() call
and a static source scan of train.py/ppo.py).
"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.rhine import action_mask
from agent_code.rhine import callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import apply_oscillation_breaker, mask_from_semantic
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


def _mask_at(field, pos, bomb_available=True):
    game_state = {"field": field, "self": ("me", 0, bomb_available, pos), "coins": []}
    semantic = extract_semantic_state(game_state)
    return semantic, mask_from_semantic(semantic)


# --- 1. Trigger-condition correctness (is_confined_to_small_range gate) ---

def test_no_effect_when_history_too_short():
    field = make_field(["#########", "#.......#", "#########"])
    _, mask = _mask_at(field, (4, 1))
    # One entry short of the confinement window: must not fire.
    history = [(3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1)]
    result, streak = apply_oscillation_breaker(mask, history)
    assert np.array_equal(result, mask)
    assert streak == 0


def test_no_effect_when_not_confined_to_small_range():
    field = make_field(["#########", "#.......#", "#########"])
    _, mask = _mask_at(field, (4, 1))
    # Full window but three distinct tiles: not confined.
    history = [(2, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1)]
    result, streak = apply_oscillation_breaker(mask, history)
    assert np.array_equal(result, mask)
    assert streak == 0


# --- 2. Retreat filtering vs. dead-end branch, built from concrete map layouts ---

def test_masks_reverse_direction_in_through_passage():
    # Straight corridor: both the retreat and the onward direction are legal.
    field = make_field(["#########", "#.......#", "#########"])
    semantic, mask = _mask_at(field, (4, 1))
    assert mask[cfg.ACTIONS.index("LEFT")] and mask[cfg.ACTIONS.index("RIGHT")]

    # Just stepped RIGHT, bouncing between the same two tiles.
    history = [(4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1)]
    result, streak = apply_oscillation_breaker(mask, history)

    assert result[cfg.ACTIONS.index("LEFT")] == False  # Retreat masked.
    assert result[cfg.ACTIONS.index("RIGHT")] == mask[cfg.ACTIONS.index("RIGHT")]
    assert result[cfg.ACTIONS.index("UP")] == mask[cfg.ACTIONS.index("UP")]
    assert result[cfg.ACTIONS.index("DOWN")] == mask[cfg.ACTIONS.index("DOWN")]
    assert result[cfg.ACTIONS.index("BOMB")] == mask[cfg.ACTIONS.index("BOMB")]
    assert result[cfg.ACTIONS.index("WAIT")] == mask[cfg.ACTIONS.index("WAIT")]
    # Only that one bit changed.
    assert (result != mask).sum() == 1
    assert streak == 1


def test_masks_reverse_direction_toward_a_side_branch():
    # Onward direction walled off but a side branch is legal: the retreat is
    # still filtered, since another movement option remains.
    field = make_field(
        [
            "#####",
            "#..##",
            "##.##",
            "#####",
        ]
    )
    semantic, mask = _mask_at(field, (2, 1))
    assert mask[cfg.ACTIONS.index("LEFT")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Wall.
    assert mask[cfg.ACTIONS.index("DOWN")] == True  # Side branch.

    # Just stepped RIGHT, bouncing between the same two tiles.
    history = [(2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1)]
    result, streak = apply_oscillation_breaker(mask, history)

    assert result[cfg.ACTIONS.index("LEFT")] == False  # Retreat filtered.
    assert result[cfg.ACTIONS.index("DOWN")] == True  # Side branch untouched.
    assert streak == 1


def test_does_not_mask_at_a_genuine_dead_end():
    # Dead-end tile: the retreat is the only legal move and must stay legal.
    field = make_field(["#########", "#.......#", "#########"])
    semantic, mask = _mask_at(field, (7, 1))
    assert mask[cfg.ACTIONS.index("LEFT")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Wall.

    history = [(7, 1), (6, 1), (7, 1), (6, 1), (7, 1), (6, 1), (7, 1)]
    result, streak = apply_oscillation_breaker(mask, history)

    assert np.array_equal(result, mask)
    assert streak == 0


def test_no_effect_when_previous_tile_not_actually_adjacent():
    # Non-adjacent previous entry (e.g. cross-round history): must be a no-op.
    field = make_field(["#########", "#.......#", "#########"])
    _, mask = _mask_at(field, (4, 1))
    history = [(4, 1), (1, 1), (4, 1), (1, 1), (4, 1), (1, 1), (4, 1)]
    result, streak = apply_oscillation_breaker(mask, history)
    assert np.array_equal(result, mask)
    assert streak == 0


def test_no_retreat_filtering_after_wait_or_bomb():
    # No movement last step (WAIT/BOMB): no-op, and the streak is preserved
    # since confinement still holds.
    field = make_field(["#########", "#.......#", "#########"])
    _, mask = _mask_at(field, (4, 1))
    history = [(3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (4, 1)]  # Last action was WAIT/BOMB.
    result, streak = apply_oscillation_breaker(mask, history, intervention_streak=2)
    assert np.array_equal(result, mask)
    assert streak == 2


# --- 3. Per-step independence: no cross-call lock-in state ---

def test_every_call_judged_independently_when_not_confined():
    field = make_field(["#########", "#.......#", "#########"])
    _, mask = _mask_at(field, (4, 1))

    triggering_history = [(4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1)]
    fired, streak = apply_oscillation_breaker(mask, triggering_history)
    assert fired[cfg.ACTIONS.index("LEFT")] == False  # It fired once.
    assert streak == 1

    # A later non-confined history is unaffected by the earlier firing and
    # reports streak 0 regardless of the streak passed in.
    non_confined_history = [(1, 1), (2, 1), (3, 1), (4, 1), (4, 1), (5, 1), (6, 1)]
    _, mask_at_6 = _mask_at(field, (6, 1))
    not_fired, reset_streak = apply_oscillation_breaker(mask_at_6, non_confined_history, intervention_streak=streak)
    assert np.array_equal(not_fired, mask_at_6)
    assert reset_streak == 0


# --- 4. Consecutive-intervention cap ---

def test_stops_intervening_after_max_consecutive_interventions():
    # Under sustained confinement the breaker filters for exactly
    # OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS calls, then stops.
    field = make_field(["#########", "#.......#", "#########"])
    _, mask_left = _mask_at(field, (3, 1))
    _, mask_right = _mask_at(field, (4, 1))
    history = deque([(4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1), (4, 1)], maxlen=7)

    streak = 0
    cap = action_mask.OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS
    for step in range(cap):
        current_mask = mask_left if history[-1] == (3, 1) else mask_right
        result, streak = apply_oscillation_breaker(current_mask, history, intervention_streak=streak)
        assert (result != current_mask).sum() == 1, f"expected an intervention at step {step}"
        assert streak == step + 1
        next_pos = (4, 1) if history[-1] == (3, 1) else (3, 1)
        history.append(next_pos)

    # Cap reached: this call must not intervene, and the streak must stay put.
    current_mask = mask_left if history[-1] == (3, 1) else mask_right
    result, streak = apply_oscillation_breaker(current_mask, history, intervention_streak=streak)
    assert np.array_equal(result, current_mask)
    assert streak == cap


# --- 4. Wiring into callbacks.act(): default-off behavior unchanged, and the
#        training path is provably never touched. ---

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


def _corridor_game_state(pos, step=10):
    field = make_field(["#########", "#.......#", "#########"])
    return {"field": field, "self": ("me", 0, True, pos), "coins": [], "step": step}


def test_default_off_act_behavior_unchanged():
    assert cfg.ENABLE_OSCILLATION_BREAKER is True  # Stored default.
    cfg.ENABLE_OSCILLATION_BREAKER = False
    try:
        agent_self = _make_agent_self(train=False)

        # With the flag off, every action must come from the unmodified mask.
        positions = [(4, 1), (3, 1), (4, 1), (3, 1), (4, 1)]
        for pos in positions:
            game_state = _corridor_game_state(pos)
            semantic = extract_semantic_state(game_state)
            expected_mask = mask_from_semantic(semantic)
            action = callbacks.act(agent_self, game_state)
            assert expected_mask[cfg.ACTIONS.index(action)] == True
        # The breaker's position deque stays empty when off.
        assert len(agent_self.recent_positions_for_breaker) == 0
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True


def test_train_mode_never_populates_breaker_state_even_when_enabled():
    """With the flag on, act() in train=True mode never touches the breaker
    state or applies the refinement.
    """
    cfg.ENABLE_OSCILLATION_BREAKER = True
    try:
        agent_self = _make_agent_self(train=True)
        positions = [(4, 1), (3, 1), (4, 1), (3, 1), (4, 1), (3, 1)]
        for pos in positions:
            callbacks.act(agent_self, _corridor_game_state(pos))
        assert len(agent_self.recent_positions_for_breaker) == 0
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True


def test_eval_mode_applies_breaker_when_enabled():
    """With the flag on, eval mode (train=False) does apply the breaker."""
    cfg.ENABLE_OSCILLATION_BREAKER = True
    try:
        agent_self = _make_agent_self(train=False)
        positions = [(4, 1), (3, 1), (4, 1), (3, 1)]
        for pos in positions:
            callbacks.act(agent_self, _corridor_game_state(pos))
        assert len(agent_self.recent_positions_for_breaker) == 4
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True


# --- 5. Static check: the training path's source never references the breaker. ---

def test_train_py_and_ppo_py_never_reference_the_breaker():
    import agent_code.rhine.train as train_module
    import agent_code.rhine.ppo as ppo_module

    for module in (train_module, ppo_module):
        source = open(module.__file__).read()
        assert "apply_oscillation_breaker" not in source
        assert "recent_positions_for_breaker" not in source


# --- 6. PPO_AGENT_ENABLE_OSCILLATION_BREAKER: enables the breaker via
#        callbacks.setup() alone, for `python main.py play`. ---

def test_env_var_unset_leaves_default_behavior_unchanged():
    common._set_or_clear_env("PPO_AGENT_ENABLE_OSCILLATION_BREAKER", None)
    cfg.ENABLE_OSCILLATION_BREAKER = False
    try:
        _make_agent_self(train=False)
        assert cfg.ENABLE_OSCILLATION_BREAKER is False  # Untouched without the env var.
    finally:
        cfg.ENABLE_OSCILLATION_BREAKER = True


def test_env_var_set_flips_the_config_flag():
    common._set_or_clear_env("PPO_AGENT_ENABLE_OSCILLATION_BREAKER", "1")
    cfg.ENABLE_OSCILLATION_BREAKER = False
    try:
        _make_agent_self(train=False)
        assert cfg.ENABLE_OSCILLATION_BREAKER is True
    finally:
        common._set_or_clear_env("PPO_AGENT_ENABLE_OSCILLATION_BREAKER", None)
        cfg.ENABLE_OSCILLATION_BREAKER = True


def test_env_var_reaches_the_play_code_path_without_setup_training():
    """The play path runs callbacks.setup() without setup_training(); the
    breaker must actually engage once enabled through the env var.
    """
    common._set_or_clear_env("PPO_AGENT_ENABLE_OSCILLATION_BREAKER", "1")
    cfg.ENABLE_OSCILLATION_BREAKER = False
    try:
        agent_self = _make_agent_self(train=False)
        assert cfg.ENABLE_OSCILLATION_BREAKER is True
        positions = [(4, 1), (3, 1), (4, 1), (3, 1)]
        for pos in positions:
            callbacks.act(agent_self, _corridor_game_state(pos))
        assert len(agent_self.recent_positions_for_breaker) == 4
    finally:
        common._set_or_clear_env("PPO_AGENT_ENABLE_OSCILLATION_BREAKER", None)
        cfg.ENABLE_OSCILLATION_BREAKER = True
