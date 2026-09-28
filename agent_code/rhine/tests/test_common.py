"""Tests for scripts/common.py's _time_to_stop_no_early_clear() using minimal
stand-ins for BombeRLeWorld.
"""
import numpy as np

import settings as s
from agent_code.rhine.scripts.common import AGENT_CODE_NAME, _time_to_stop_no_early_clear


class _FakeAgent:
    def __init__(self, code_name, train=False):
        self.code_name = code_name
        self.train = train


class _FakeCoin:
    def __init__(self, collectable):
        self.collectable = collectable


class _FakeArgs:
    def __init__(self, continue_without_training=True):
        self.continue_without_training = continue_without_training


class _FakeWorld:
    def __init__(
        self, active_agents, agents=None, arena=None, coins=None, bombs=None,
        explosions=None, step=0, continue_without_training=True,
    ):
        self.active_agents = active_agents
        self.agents = agents if agents is not None else active_agents
        self.arena = arena if arena is not None else np.zeros((5, 5), dtype=int)
        self.coins = coins if coins is not None else []
        self.bombs = bombs if bombs is not None else []
        self.explosions = explosions if explosions is not None else []
        self.step = step
        self.args = _FakeArgs(continue_without_training)


def _cleared_board():
    return dict(arena=np.zeros((5, 5), dtype=int), coins=[_FakeCoin(collectable=False)], bombs=[], explosions=[])


def test_opponents_all_dead_board_clear_stops_immediately():
    # Only rhine alive and the board cleared: must stop immediately.
    world = _FakeWorld(active_agents=[_FakeAgent(AGENT_CODE_NAME)], **_cleared_board())
    assert _time_to_stop_no_early_clear(world) is True


def test_opponent_still_alive_board_clear_keeps_running():
    # An opponent is still alive when the board clears: must not stop early.
    world = _FakeWorld(
        active_agents=[_FakeAgent(AGENT_CODE_NAME), _FakeAgent("peaceful_agent")],
        **_cleared_board(),
    )
    assert _time_to_stop_no_early_clear(world) is False


def test_opponent_still_alive_board_not_clear_keeps_running():
    # Board not clear, opponent alive.
    world = _FakeWorld(
        active_agents=[_FakeAgent(AGENT_CODE_NAME), _FakeAgent("peaceful_agent")],
        arena=np.array([[1, 0], [0, 0]]),
    )
    assert _time_to_stop_no_early_clear(world) is False


def test_opponents_all_dead_board_not_clear_keeps_running():
    # Opponents dead but the board is not clear: must not stop.
    world = _FakeWorld(active_agents=[_FakeAgent(AGENT_CODE_NAME)], arena=np.array([[1, 0], [0, 0]]))
    assert _time_to_stop_no_early_clear(world) is False


def test_all_agents_dead_stops_regardless_of_opponents():
    # Nobody left alive.
    world = _FakeWorld(active_agents=[])
    assert _time_to_stop_no_early_clear(world) is True


def test_step_limit_stops_regardless_of_opponents():
    # Step cap reached with an opponent still alive.
    world = _FakeWorld(
        active_agents=[_FakeAgent(AGENT_CODE_NAME), _FakeAgent("peaceful_agent")],
        step=s.MAX_STEPS,
    )
    assert _time_to_stop_no_early_clear(world) is True
