import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import state_processing as sp
from agent_code.rhine.features import features_from_semantic


def make_field(rows_pattern):
    """'#'=wall, '.'=free; returns field[x, y] like game_state['field']."""
    height = len(rows_pattern)
    width = len(rows_pattern[0])
    field = np.zeros((width, height), dtype=int)
    for y, row in enumerate(rows_pattern):
        for x, ch in enumerate(row):
            field[x, y] = -1 if ch == "#" else 0
    return field


def test_is_free_returns_plain_python_bool():
    field = make_field(["###", "#.#", "###"])
    assert sp.is_free(field, 1, 1) is True
    assert sp.is_free(field, 0, 1) is False


def test_bfs_distances_simple_corridor():
    field = make_field([
        "#####",
        "#...#",
        "#####",
    ])
    dist = sp.bfs_distances(field, (1, 1))
    assert dist[(1, 1)] == 0
    assert dist[(2, 1)] == 1
    assert dist[(3, 1)] == 2
    assert (0, 1) not in dist  # Wall.


def test_legal_moves_respects_walls():
    field = make_field([
        "#####",
        "#.#.#",
        "#...#",
        "#####",
    ])
    moves = sp.legal_moves(field, (1, 1))
    assert moves["DOWN"] is True
    assert moves["RIGHT"] is False
    assert moves["UP"] is False
    assert moves["LEFT"] is False


def test_nearest_reachable_coin_distance_none_when_blocked():
    field = make_field([
        "#####",
        "#.#.#",
        "#.#.#",
        "#####",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [(3, 1)],  # Separated by a wall column.
    }
    assert sp.nearest_reachable_coin_distance(game_state) is None


def test_nearest_reachable_coin_distance_reachable():
    field = make_field([
        "#####",
        "#...#",
        "#####",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [(3, 1)],
    }
    assert sp.nearest_reachable_coin_distance(game_state) == 2


def test_no_coins_returns_none():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    assert sp.nearest_reachable_coin_distance(game_state) is None


def test_equal_distance_coins_in_different_directions_both_marked():
    # No arbitrary tie-break between equally-near coins.
    field = make_field([
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (2, 2)),
        "coins": [(2, 1), (1, 2)],  # Equally near, in different directions.
    }
    semantic = sp.extract_semantic_state(game_state)
    assert semantic.nearest_coin_distance == 1
    assert semantic.coin_path_dirs["UP"] is True
    assert semantic.coin_path_dirs["LEFT"] is True
    assert semantic.coin_path_dirs["DOWN"] is False
    assert semantic.coin_path_dirs["RIGHT"] is False


def test_multiple_shortest_paths_to_single_coin_both_first_steps_marked():
    # Two equally short paths to one coin: both first steps are marked.
    field = make_field([
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [(3, 3)],
    }
    semantic = sp.extract_semantic_state(game_state)
    assert semantic.nearest_coin_distance == 4
    assert semantic.coin_path_dirs["RIGHT"] is True
    assert semantic.coin_path_dirs["DOWN"] is True
    assert semantic.coin_path_dirs["UP"] is False
    assert semantic.coin_path_dirs["LEFT"] is False


def test_single_shortest_path_only_correct_direction_marked():
    field = make_field([
        "#####",
        "#...#",
        "#####",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [(3, 1)],
    }
    semantic = sp.extract_semantic_state(game_state)
    assert semantic.coin_path_dirs["RIGHT"] is True
    assert semantic.coin_path_dirs["UP"] is False
    assert semantic.coin_path_dirs["DOWN"] is False
    assert semantic.coin_path_dirs["LEFT"] is False


def test_nearest_coin_distance_feature_clipped_when_maze_geometry_exceeds_max_board_distance():
    # A long corridor makes the BFS distance exceed cfg.MAX_BOARD_DISTANCE: the
    # raw distance keeps the true value while the feature is clipped to 1.0.
    width = 44
    field = make_field(["#" * width, "#" + "." * (width - 2) + "#", "#" * width])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": [(width - 2, 1)]}
    semantic = sp.extract_semantic_state(game_state)
    assert semantic.nearest_coin_distance > cfg.MAX_BOARD_DISTANCE
    features = features_from_semantic(semantic)
    assert features[5] == 1.0


def test_has_reachable_coin_false_when_no_coins():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    semantic = sp.extract_semantic_state(game_state)
    assert semantic.has_reachable_coin is False
    assert semantic.nearest_coin_distance is None
    assert all(v is False for v in semantic.coin_path_dirs.values())


def test_has_safe_escape_after_bombing_true_in_open_room():
    # A bomb in the room's center leaves the corners as an escape.
    field = make_field(["#####", "#...#", "#...#", "#...#", "#####"])
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=3)
    assert sp.has_safe_escape_after_bombing(
        (2, 2), field, blocked=frozenset(), danger_offsets=danger_offsets, power=3, opponents=[]
    ) is True


def test_has_safe_escape_after_bombing_false_in_dead_end_corridor():
    # The blast covers the whole corridor, so no escape exists.
    field = make_field(["#####", "#...#", "#####"])
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=3)
    assert sp.has_safe_escape_after_bombing(
        (2, 1), field, blocked=frozenset(), danger_offsets=danger_offsets, power=3, opponents=[]
    ) is False


def test_has_safe_escape_after_bombing_false_with_only_one_safe_direction_and_opponent_nearby():
    # Only one direction leads to safety; a nearby opponent requires two
    # independent directions, so the escape is insufficient.
    field = make_field(["#############", "#.......#...#", "#############"])
    tile = (6, 1)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER, opponents=[tile]
    ) is False


def test_has_safe_escape_after_bombing_true_with_only_one_safe_direction_and_no_opponent_nearby():
    # Without a nearby opponent the single working direction suffices.
    field = make_field(["#############", "#.......#...#", "#############"])
    tile = (6, 1)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER, opponents=[]
    ) is True
    # A distant opponent behaves like no opponent.
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(tile[0] + 3, tile[1])],
    ) is True


def test_has_safe_escape_after_bombing_opponent_proximity_threshold_boundary():
    # Boundary of the opponent-proximity trigger: just inside must require two
    # directions, just outside must not. The outside opponent is unreachable
    # by BFS so only the proximity cutoff is tested.
    field = make_field(["#############", "#.......#...#", "#############"])
    tile = (6, 1)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(tile[0] - 2, tile[1])],
    ) is False
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(tile[0] + 3, tile[1])],
    ) is True


def test_has_safe_escape_after_bombing_true_with_two_independent_safe_directions():
    # Two independent escape directions pass even with a nearby opponent. The
    # opponent sits on a wall cell beside `tile`, so it raises the requirement
    # without being able to seal either escape neighbor.
    width = 19
    middle_row = "#" + "." * (width - 2) + "#"
    field = make_field(["#" * width, middle_row, "#" * width])
    tile = (width // 2, 1)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(tile[0], tile[1] - 1)],
    ) is True


def test_has_safe_escape_after_bombing_opponent_catches_up_on_a_later_leg():
    # An opponent too far from `tile` to matter for the first step can still
    # beat the agent to a later tile of the only escape route; with no
    # conflict-disjoint alternative, the escape must be rejected.
    width = 20
    field = make_field(["#" * width, "#" + "." * (width - 2) + "#", "#" * width])
    tile = (width - 2, 1)  # Wall immediately to the right: LEFT only.
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(11, 1)],
    ) is False
    # Without the opponent the single path suffices, so the rejection above is
    # due to the opponent.
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[],
    ) is True


def test_has_safe_escape_after_bombing_depth2_opponent_prediction_blocks_later_hop():
    """An opponent reachable only through a side passage can reach a later hop
    of the only escape route in time.

    Hand-crafted danger_offsets force a zero-slack schedule (move one tile per
    tick, no waiting) down a 1-wide corridor to its only safe endpoint. The
    escape succeeds without the opponent and must fail with it.
    """
    field = make_field([
        "##########",
        "#........#",
        "######.###",
        "######.###",
        "##########",
    ])
    tile = (8, 1)
    danger_offsets = {
        (7, 1): set(range(2, 12)),
        (6, 1): set(range(3, 12)),
        (5, 1): set(range(4, 12)),
        (4, 1): set(range(5, 12)),
        (3, 1): set(range(6, 12)),
        (2, 1): set(),
    }
    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[],
    ) is True

    assert sp.has_safe_escape_after_bombing(
        tile, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER,
        opponents=[(6, 3)],
    ) is False


# --- bomb_threatens_reachable_opponent(): pure blast-coordinate snapshot ---

def test_bomb_threatens_reachable_opponent_true_for_crate_boxed_opponent():
    # A crate-boxed opponent with no free neighbors is still threatened:
    # crates do not stop the blast.
    field = make_field(["#######", "#.....#", "#######"])
    field[2, 1] = 1  # Crate.
    field[4, 1] = 1  # Crate.
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "others": [("opp", 0, True, (3, 1))]}
    assert sp.bomb_threatens_reachable_opponent(game_state, (1, 1), cfg.BOMB_POWER) is True


def test_bomb_threatens_reachable_opponent_false_outside_blast():
    field = make_field(["#" * 12, "#" + "." * 10 + "#", "#" * 12])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "others": [("opp", 0, True, (9, 1))]}
    assert sp.bomb_threatens_reachable_opponent(game_state, (1, 1), cfg.BOMB_POWER) is False


def test_bomb_threatens_reachable_opponent_false_with_no_opponents():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "others": []}
    assert sp.bomb_threatens_reachable_opponent(game_state, (1, 1), cfg.BOMB_POWER) is False


# --- kill_target_info(): independent snapshot-based kill-value scoring ---

def test_kill_target_info_no_target_when_opponent_out_of_every_reachable_blast():
    # A solid wall row separates the rooms, so no reachable tile's blast covers
    # the opponent.
    field = make_field([
        "#############",
        "#...........#",
        "#############",
        "#...........#",
        "#############",
    ])
    self_pos = (1, 1)
    opp = (11, 3)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    has_target, distance, path_dirs, expected_value = sp.kill_target_info(
        field, self_pos, blocked=frozenset(), power=cfg.BOMB_POWER, danger_offsets=danger_offsets, opponents=[opp]
    )
    assert has_target is False
    assert distance is None
    assert expected_value == 0.0
    assert all(v is False for v in path_dirs.values())


def test_kill_target_info_easy_escape_scores_zero_difficulty():
    # An opponent in a big open room with no walls or chasers nearby has the
    # minimum difficulty.
    width = 11
    row = "#" + "." * (width - 2) + "#"
    field = make_field(["#" * width] + [row] * (width - 2) + ["#" * width])
    self_pos = (5, 7)
    opp = (5, 5)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    has_target, distance, _, expected_value = sp.kill_target_info(
        field, self_pos, blocked=frozenset(), power=cfg.BOMB_POWER, danger_offsets=danger_offsets, opponents=[opp]
    )
    assert has_target is True
    assert distance == 0
    assert expected_value == 0.0


def test_kill_target_info_trapped_opponent_scores_maximum_difficulty():
    # An opponent at the end of a dead-end pocket whose only exit is blocked by
    # self has the maximum difficulty; self still has its own escape.
    field = make_field([
        "###########",
        "#.........#",
        "#.........#",
        "#.........#",
        "#####.#####",
        "#####.#####",
        "#####.#####",
        "###########",
    ])
    self_pos = (5, 3)
    opp = (5, 6)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    assert sp.has_safe_escape_after_bombing(
        self_pos, field, blocked=frozenset(), danger_offsets=danger_offsets, power=cfg.BOMB_POWER, opponents=[opp]
    ) is True
    has_target, distance, _, expected_value = sp.kill_target_info(
        field, self_pos, blocked=frozenset(), power=cfg.BOMB_POWER, danger_offsets=danger_offsets, opponents=[opp]
    )
    assert has_target is True
    assert distance == 0
    assert expected_value == 1.0


def test_kill_target_info_maxes_not_sums_difficulty_across_opponents():
    # Two opponents in the same blast with different difficulties: the kill
    # value is their maximum, not their sum or average.
    field = make_field([
        "#############",
        "#...........#",
        "#...........#",
        "#...........#",
        "#....#.#....#",
        "#.......#...#",
        "#...........#",
        "#.......#...#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#############",
    ])
    self_pos = (6, 6)
    opp_a = (6, 4)
    opp_b = (8, 6)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    has_target, distance, _, expected_value = sp.kill_target_info(
        field, self_pos, blocked=frozenset(), power=cfg.BOMB_POWER, danger_offsets=danger_offsets,
        opponents=[opp_a, opp_b],
    )
    assert has_target is True
    assert distance == 0
    assert abs(expected_value - 0.75) < 1e-9


def test_kill_target_info_distance_and_direction_hint():
    # Same pocket, self starting away from its mouth: checks the BFS distance
    # and that both tied first steps are marked.
    field = make_field([
        "###########",
        "#.........#",
        "#.........#",
        "#.........#",
        "#####.#####",
        "#####.#####",
        "#####.#####",
        "###########",
    ])
    self_pos = (3, 2)
    opp = (5, 6)
    danger_offsets = sp.compute_danger_offsets(field, bombs=[], explosion_map=np.zeros_like(field), power=cfg.BOMB_POWER)
    has_target, distance, path_dirs, expected_value = sp.kill_target_info(
        field, self_pos, blocked=frozenset(), power=cfg.BOMB_POWER, danger_offsets=danger_offsets, opponents=[opp]
    )
    assert has_target is True
    assert distance == 3
    assert path_dirs == {"UP": False, "DOWN": True, "LEFT": False, "RIGHT": True}
    assert expected_value == 1.0
