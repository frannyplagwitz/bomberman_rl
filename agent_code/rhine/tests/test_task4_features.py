import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.state_processing import SemanticState, extract_semantic_state, reachable_space_count
from agent_code.rhine.scripts.investigate_stage_d_batch4 import _reachable_space

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


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


def others(*positions):
    return [(f"opp{i}", 0, True, p) for i, p in enumerate(positions)]


# --- #29/#30 nearest_alive_opponent_distance / opponents_within_3, via BFS ---

def test_no_opponents_distance_defaults_to_max_and_count_zero():
    field = make_field(["#######", "#.....#", "#######"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": [], "others": []}
    semantic = extract_semantic_state(game_state)
    assert semantic.nearest_alive_opponent_distance is None
    assert semantic.opponents_within_3 == 0

    features = features_from_semantic(semantic)
    assert features[28] == 1.0
    assert features[29] == 0.0


def test_unreachable_opponent_treated_like_no_opponent():
    # Two rooms split by a wall: the opponent has free neighbors, none reachable.
    field = make_field(["#########", "#...#...#", "#########"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": [], "others": others((7, 1))}
    semantic = extract_semantic_state(game_state)
    assert semantic.nearest_alive_opponent_distance is None
    assert semantic.opponents_within_3 == 0

    features = features_from_semantic(semantic)
    assert features[28] == 1.0
    assert features[29] == 0.0


def test_multiple_opponents_nearest_and_within_3_count():
    # Two opponents inside the range threshold, one outside.
    field = make_field(["#############", "#...........#", "#############"])
    game_state = {
        "field": field, "self": ("me", 0, True, (6, 1)), "coins": [],
        "others": others((3, 1), (9, 1), (1, 1)),
    }
    semantic = extract_semantic_state(game_state)
    assert semantic.nearest_alive_opponent_distance == 3
    assert semantic.opponents_within_3 == 2

    features = features_from_semantic(semantic)
    assert features[28] == 3 / cfg.DISTANCE_FEATURE_NORM
    assert features[29] == 2 / cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD


def test_opponent_behind_wall_gap_forces_bfs_detour():
    # The corridors connect only through two side columns, so the true BFS
    # path detours around the wall.
    field = make_field([
        "#########",
        "#.......#",
        "#.#####.#",
        "#.......#",
        "#########",
    ])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": [], "others": others((7, 3))}
    semantic = extract_semantic_state(game_state)
    # Distance comes from the BFS detour, not a straight-line shortcut.
    assert semantic.nearest_alive_opponent_distance == 8


# --- #31 reachable_space, via BFS ---

def test_opponents_block_reachable_space_like_walls():
    # Opponents sit on the only paths outward, capping the count.
    field = make_field(["#############", "#...........#", "#############"])
    game_state = {
        "field": field, "self": ("me", 0, True, (6, 1)), "coins": [],
        "others": others((3, 1), (9, 1), (1, 1)),
    }
    semantic = extract_semantic_state(game_state)
    # Counting stops at the opponents on both sides.
    assert semantic.reachable_space == 5
    features = features_from_semantic(semantic)
    assert features[30] == 5 / cfg.REACHABLE_SPACE_NORM


def test_reachable_space_count_matches_investigate_stage_d_batch4_reference():
    """reachable_space_count() matches the analysis-script reference on
    handwritten maps; runtime code must not import analysis scripts, so the
    check lives in tests.
    """
    cases = [
        # (field, self_pos, opponents)
        (make_field(["#######", "#.....#", "#######"]), (3, 1), []),
        (make_field(["#####", "#.X.#", "#...#", "#####"]), (1, 1), []),
        (make_field(["#############", "#...........#", "#############"]), (6, 1), [(3, 1), (9, 1)]),
        (make_field(["#" * 15] + ["#" + "." * 13 + "#"] + ["#" * 15]), (1, 1), [(10, 1)]),
    ]
    for field, self_pos, opponents in cases:
        assert reachable_space_count(field, self_pos, opponents, depth_cap=6) == _reachable_space(
            field, self_pos, opponents, depth_cap=6
        )


# --- Boundary/1.0-saturation behavior, via synthetic SemanticState ---

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


def test_nearest_alive_opponent_distance_saturates_at_1_0():
    at_cap = features_from_semantic(_make_semantic(nearest_alive_opponent_distance=cfg.DISTANCE_FEATURE_NORM))
    over_cap = features_from_semantic(
        _make_semantic(nearest_alive_opponent_distance=cfg.DISTANCE_FEATURE_NORM + 5)
    )
    under_cap = features_from_semantic(_make_semantic(nearest_alive_opponent_distance=1))
    assert at_cap[28] == 1.0
    assert over_cap[28] == 1.0
    assert under_cap[28] == 1 / cfg.DISTANCE_FEATURE_NORM


def test_opponents_within_3_saturates_at_1_0():
    at_cap = features_from_semantic(_make_semantic(opponents_within_3=cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD))
    # No input can exceed the cap, so only the exact-threshold boundary is checked.
    assert at_cap[29] == 1.0


def test_reachable_space_saturates_at_1_0():
    at_cap = features_from_semantic(_make_semantic(reachable_space=cfg.REACHABLE_SPACE_NORM))
    over_cap = features_from_semantic(_make_semantic(reachable_space=cfg.REACHABLE_SPACE_NORM + 10))
    under_cap = features_from_semantic(_make_semantic(reachable_space=1))
    assert at_cap[30] == 1.0
    assert over_cap[30] == 1.0
    assert under_cap[30] == 1 / cfg.REACHABLE_SPACE_NORM
