from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.state_processing import SemanticState, is_confined_to_small_range
from agent_code.rhine.train import _update_stall_v2

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def _make_semantic(**overrides):
    base = dict(
        self_pos=(5, 5),
        can_move={d: True for d in DIRECTIONS},
        has_reachable_coin=False,
        nearest_coin_distance=None,
        bomb_available=True,
    )
    base.update(overrides)
    return SemanticState(**base)


def test_switch_off_default_dim_and_values_unchanged():
    assert cfg.ENABLE_STALL_HISTORY_FEATURE is False
    semantic = _make_semantic()

    features = features_from_semantic(semantic)
    assert features.shape == (31,)

    # stall_history passed while the switch is off is ignored; the vector is unchanged.
    features_with_ignored_arg = features_from_semantic(semantic, stall_history=True)
    assert features_with_ignored_arg.shape == (31,)
    assert np.array_equal(features, features_with_ignored_arg)


def test_switch_on_adds_32nd_dim(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_STALL_HISTORY_FEATURE", True)
    semantic = _make_semantic()

    features_true = features_from_semantic(semantic, stall_history=True)
    features_false = features_from_semantic(semantic, stall_history=False)

    assert features_true.shape == (32,)
    assert features_false.shape == (32,)
    assert features_true[31] == 1.0
    assert features_false[31] == 0.0
    # Base dims are unaffected by the switch/stall_history value.
    assert np.array_equal(features_true[:31], features_false[:31])


def test_38th_dim_matches_is_confined_to_small_range_three_cases(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_STALL_HISTORY_FEATURE", True)
    semantic = _make_semantic()

    # (a) Window not yet full: always False.
    partial = deque([(1, 1), (2, 2)], maxlen=4)
    expected = is_confined_to_small_range(partial)
    assert expected is False
    features = features_from_semantic(semantic, stall_history=expected)
    assert features[31] == 0.0

    # (b) Full window confined to two tiles: the oscillation shape.
    oscillating = deque([(1, 1), (2, 1), (1, 1), (2, 1)], maxlen=4)
    expected = is_confined_to_small_range(oscillating)
    assert expected is True
    features = features_from_semantic(semantic, stall_history=expected)
    assert features[31] == 1.0

    # (c) Full window after walking out to a new tile.
    moved_on = deque([(2, 1), (1, 1), (2, 1), (3, 1)], maxlen=4)
    expected = is_confined_to_small_range(moved_on)
    assert expected is False
    features = features_from_semantic(semantic, stall_history=expected)
    assert features[31] == 0.0


def test_shared_logic_matches_update_stall_v2_for_same_position_sequence():
    """The stall-history feature uses the same decision rule as
    _update_stall_v2 for the reward: both agree at every step on an identical
    position sequence (bomb always available, so V2's exemption never applies).
    """
    fake_self = SimpleNamespace(recent_positions_v2=deque(maxlen=4))
    sequence = [(5, 5), (6, 5), (5, 5), (6, 5), (9, 9), (5, 5), (6, 5)]

    feature_side_history = deque(maxlen=4)
    for pos in sequence:
        reward_side_result = _update_stall_v2(fake_self, _make_semantic(self_pos=pos, bomb_available=True))
        feature_side_history.append(pos)
        feature_side_result = is_confined_to_small_range(feature_side_history)
        assert reward_side_result == feature_side_result
