"""Tests for the bombing-target-selection metric replacing bomb_efficiency:
diagnose_bombing_cost_benefit.bombing_target_best_by_score() /
is_optimal_bombing_placement(). Uses concrete maps so the expected best set
can be verified by hand from the blast rule.

The distance map is computed from the queried tile, so a tile with its own
crate usually wins unless a nearby tile reaches a much larger,
blast-isolated crate group. All fields disable
cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER, matching the deployed config.
"""
import numpy as np
import pytest

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.diagnose_bombing_cost_benefit import (
    bombing_target_best_by_score,
    is_optimal_bombing_placement,
)
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


def _semantic_at(field, pos):
    game_state = {"field": field, "self": ("me", 0, True, pos), "coins": []}
    return extract_semantic_state(game_state)


@pytest.fixture(autouse=True)
def _disable_safety_filter():
    original = cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER
    cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER = False
    yield
    cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER = original


def _best_set(field, pos):
    semantic = _semantic_at(field, pos)
    return bombing_target_best_by_score(
        semantic.field_arr, pos, semantic.blocked, cfg.BOMB_POWER, semantic.danger_offsets
    )


def _is_optimal(field, pos):
    semantic = _semantic_at(field, pos)
    return is_optimal_bombing_placement(
        semantic.field_arr, pos, semantic.blocked, cfg.BOMB_POWER, semantic.danger_offsets
    )


# A crate cuts the corridor into a dead-end region whose tiles can all hit it.
CORRIDOR_FIELD = make_field(["#############", "#....X.....X#", "#############"])


def test_unique_best_candidate_is_optimal():
    # The queried tile hits the crate at distance 0, so it is the unique best.
    assert _best_set(CORRIDOR_FIELD, (2, 1)) == frozenset({(2, 1)})
    assert _is_optimal(CORRIDOR_FIELD, (2, 1)) is True


def test_non_candidate_tile_with_candidates_elsewhere_is_not_optimal():
    # The queried tile hits no crate, but other candidates exist, so the result
    # is False rather than None.
    best_set = _best_set(CORRIDOR_FIELD, (1, 1))
    assert best_set  # Candidates exist.
    assert (1, 1) not in best_set
    assert _is_optimal(CORRIDOR_FIELD, (1, 1)) is False


# Self hits one crate, but the adjacent corner tile reaches a denser crate
# stack that self's blast cannot (blasts never turn corners), giving the
# corner a strictly better score.
CORNER_FIELD = make_field([
    "#####",
    "#X..#",
    "#...#",
    "#.X.#",
    "#.X.#",
    "#.X.#",
    "#####",
])


def test_valid_candidate_beaten_by_a_denser_nearby_target_is_not_optimal():
    best_set = _best_set(CORNER_FIELD, (1, 2))
    assert best_set == frozenset({(2, 2)})
    assert _is_optimal(CORNER_FIELD, (1, 2)) is False


def test_the_denser_corner_tile_itself_is_optimal():
    assert _is_optimal(CORNER_FIELD, (2, 2)) is True


# Two symmetric corner tiles, each next to its own blast-isolated crate stack,
# tie for best from a center tile that hits no crate itself.
TIE_FIELD = make_field([
    "#######",
    "#######",
    "#.....#",
    "#.X.X.#",
    "#.X.X.#",
    "#.X.X.#",
    "#######",
])


def test_tied_best_by_score_contains_both_symmetric_candidates():
    assert _best_set(TIE_FIELD, (3, 2)) == frozenset({(2, 2), (4, 2)})


def test_center_not_in_the_tied_set_is_not_optimal():
    assert _is_optimal(TIE_FIELD, (3, 2)) is False


def test_either_tied_candidate_individually_counts_as_optimal():
    for pos in [(2, 2), (4, 2)]:
        assert _is_optimal(TIE_FIELD, pos) is True


# No reachable crates: there is no optimal placement to compare against.
NO_CRATE_FIELD = make_field(["#######", "#.....#", "#######"])


def test_no_candidates_anywhere_returns_none_not_false():
    assert _best_set(NO_CRATE_FIELD, (3, 1)) == frozenset()
    assert _is_optimal(NO_CRATE_FIELD, (3, 1)) is None
