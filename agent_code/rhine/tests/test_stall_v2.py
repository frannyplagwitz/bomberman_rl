from collections import deque
from types import SimpleNamespace

from agent_code.rhine.train import _update_stall_v2


def make_fake_self():
    return SimpleNamespace(recent_positions_v2=deque(maxlen=4))


def make_semantic(pos, bomb_available):
    return SimpleNamespace(self_pos=pos, bomb_available=bomb_available)


def test_partial_window_never_triggers():
    # A partially filled window is never stalled, even when standing still.
    fake_self = make_fake_self()
    positions = [(5, 5), (5, 5), (5, 5)]  # Deliberately the same tile.
    results = [_update_stall_v2(fake_self, make_semantic(p, True)) for p in positions]
    assert results == [False, False, False]


def test_confined_to_two_tiles_over_four_steps_triggers():
    fake_self = make_fake_self()
    sequence = [(5, 5), (6, 5), (5, 5), (6, 5)]  # A,B,A,B
    results = [_update_stall_v2(fake_self, make_semantic(p, True)) for p in sequence]
    # Only the step that fills the window triggers.
    assert results == [False, False, False, True]


def test_bomb_unavailable_step_is_exempt():
    fake_self = make_fake_self()
    a, b = (5, 5), (6, 5)
    # A bomb-unavailable step must not occupy a window slot.
    r1 = _update_stall_v2(fake_self, make_semantic(a, True))
    r2 = _update_stall_v2(fake_self, make_semantic((9, 9), False))  # Ignored step.
    r3 = _update_stall_v2(fake_self, make_semantic(b, True))
    r4 = _update_stall_v2(fake_self, make_semantic(a, True))
    r5 = _update_stall_v2(fake_self, make_semantic(b, True))
    assert [r1, r2, r3, r4, r5] == [False, False, False, False, True]
    assert list(fake_self.recent_positions_v2) == [a, b, a, b]  # Skipped step left no trace.


def test_new_tile_stops_penalty_and_resets():
    fake_self = make_fake_self()
    a, b, c = (5, 5), (6, 5), (9, 9)
    # A new tile stops the penalty immediately; it only re-triggers once that
    # tile has rolled out of the window.
    sequence = [a, b, a, b, a, c, a, b, a, b]
    results = [_update_stall_v2(fake_self, make_semantic(p, True)) for p in sequence]
    assert results == [
        False, False, False, True, True,   # Window fills, then stays stalled.
        False,                             # New tile C: stop immediately.
        False, False, False,               # C still in the window.
        True,                              # C rolled out: re-triggered.
    ]
