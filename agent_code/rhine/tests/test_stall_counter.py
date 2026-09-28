from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.train import _update_stall_counter


def make_fake_self():
    return SimpleNamespace(stall_counter=0, recent_positions=deque(maxlen=3))


def make_semantic(pos, has_bombing_target, nearest_bombing_distance):
    return SimpleNamespace(
        self_pos=pos,
        has_bombing_target=has_bombing_target,
        nearest_bombing_distance=nearest_bombing_distance,
    )


def mask_with_bomb(bomb_legal):
    mask = np.zeros(cfg.MODEL_CONFIG.n_actions, dtype=bool)
    mask[cfg.ACTIONS.index("BOMB")] = bomb_legal
    mask[cfg.ACTIONS.index("WAIT")] = True
    return mask


def test_counter_starts_at_one_on_first_valid_position():
    fake_self = make_fake_self()
    semantic = make_semantic((5, 5), True, 0)
    result = _update_stall_counter(fake_self, semantic, mask_with_bomb(True))
    assert result == 1


def test_counter_increments_when_stuck_at_same_position():
    fake_self = make_fake_self()
    semantic = make_semantic((5, 5), True, 0)
    for expected in range(1, 6):
        result = _update_stall_counter(fake_self, semantic, mask_with_bomb(True))
        assert result == expected


def test_counter_increments_through_two_tile_oscillation():
    # Alternating between two adjacent tiles that are both valid bombing positions.
    fake_self = make_fake_self()
    a = make_semantic((5, 5), True, 0)
    b = make_semantic((6, 5), True, 0)
    sequence = [a, b, a, b, a, b, a, b]
    results = [_update_stall_counter(fake_self, s, mask_with_bomb(True)) for s in sequence]
    # The first steps see new tiles and reset to 1; afterwards both tiles are
    # inside the history window, so the counter increments.
    assert results[0] == 1
    assert results[1] == 1
    assert results[2:] == [2, 3, 4, 5, 6, 7]


def test_counter_resets_when_moving_to_genuinely_new_tile():
    fake_self = make_fake_self()
    a = make_semantic((5, 5), True, 0)
    new_tile = make_semantic((10, 10), True, 0)
    _update_stall_counter(fake_self, a, mask_with_bomb(True))
    _update_stall_counter(fake_self, a, mask_with_bomb(True))
    result = _update_stall_counter(fake_self, new_tile, mask_with_bomb(True))
    assert result == 1


def test_counter_resets_to_zero_when_not_at_valid_bombing_position():
    fake_self = make_fake_self()
    a = make_semantic((5, 5), True, 0)
    _update_stall_counter(fake_self, a, mask_with_bomb(True))
    _update_stall_counter(fake_self, a, mask_with_bomb(True))
    not_bombing_pos = make_semantic((5, 5), True, 3)  # Same tile, no longer a bombing position.
    result = _update_stall_counter(fake_self, not_bombing_pos, mask_with_bomb(True))
    assert result == 0


def test_counter_resets_to_zero_when_bomb_illegal_even_at_valid_position():
    fake_self = make_fake_self()
    semantic = make_semantic((5, 5), True, 0)
    _update_stall_counter(fake_self, semantic, mask_with_bomb(True))
    result = _update_stall_counter(fake_self, semantic, mask_with_bomb(False))
    assert result == 0
