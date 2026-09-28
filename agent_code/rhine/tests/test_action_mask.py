import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import compute_action_mask


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


def test_bomb_masked_off_when_bomb_unavailable():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, False, (2, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("BOMB")] == False


def test_bomb_masked_off_in_narrow_corridor_with_no_escape():
    # The blast covers the whole dead-end corridor, so no escape exists.
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (2, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("BOMB")] == False


def test_wait_always_allowed():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (2, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("WAIT")] == True


def test_wall_blocked_direction_masked_false():
    field = make_field(["#####", "#.#.#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Wall.
    assert mask[cfg.ACTIONS.index("DOWN")] == True     # Free tile.


def test_open_room_all_movement_allowed_and_bomb_has_escape():
    # From the room's center the blast leaves the corners safe, so BOMB is legal.
    field = make_field(["#####", "#...#", "#...#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (2, 2)), "coins": []}
    mask = compute_action_mask(game_state)
    for direction in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
        assert mask[cfg.ACTIONS.index(direction)] == True
    assert mask[cfg.ACTIONS.index("BOMB")] == True


def test_corner_only_two_moves_and_wait_allowed():
    # From a corner the blast leaves part of the room safe and reachable, so
    # BOMB stays legal.
    field = make_field(["#####", "#...#", "#...#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("UP")] == False
    assert mask[cfg.ACTIONS.index("LEFT")] == False
    assert mask[cfg.ACTIONS.index("DOWN")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == True
    assert mask[cfg.ACTIONS.index("WAIT")] == True
    assert mask[cfg.ACTIONS.index("BOMB")] == True


def test_mask_dtype_and_length():
    field = make_field(["#####", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (2, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask.dtype == np.bool_
    assert len(mask) == 6


def test_crate_blocks_movement():
    field = make_field(["#####", "#.X.#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Crate.
    assert mask[cfg.ACTIONS.index("DOWN")] == True


def test_bomb_occupied_tile_blocks_movement():
    field = make_field(["#####", "#...#", "#...#", "#####"])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [],
        "bombs": [((2, 1), 3)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Bomb on the tile.


def test_movement_into_active_explosion_masked_off():
    field = make_field(["#####", "#...#", "#...#", "#####"])
    explosion_map = np.zeros_like(field, dtype=float)
    explosion_map[2, 1] = 1
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [],
        "bombs": [],
        "explosion_map": explosion_map,
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False  # Live explosion.


def test_movement_into_dead_end_that_will_be_bombed_masked_off():
    # Moving into a dead-end corridor that is about to be fully blasted leads
    # nowhere safe.
    field = make_field(["#####", "#...#", "#####"])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "coins": [],
        "bombs": [((3, 1), 0)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False


def test_movement_into_future_blast_with_time_to_escape_allowed():
    # Moving into a future blast is legal when there is still time to step out
    # of it.
    field = make_field([
        "#########",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, True, (1, 2)),
        "coins": [],
        "bombs": [((4, 2), cfg.BOMB_TIMER)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == True


def test_wait_masked_off_when_staying_is_fatal_but_escape_exists():
    # Staying is fatal this step but an escape exists, so WAIT is masked.
    field = make_field([
        "#########",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, False, (2, 2)),
        "coins": [],
        "bombs": [((4, 2), 0)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("UP")] == True
    assert mask[cfg.ACTIONS.index("WAIT")] == False


def test_wait_masked_off_when_one_more_step_of_staying_would_be_fatal():
    # The current tile is safe now but fatal if the agent stays one more step;
    # UP is a permanently safe escape, so WAIT is masked while UP stays legal.
    field = make_field([
        "#########",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, False, (2, 2)),
        "coins": [],
        "bombs": [((4, 2), 1)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("UP")] == True
    assert mask[cfg.ACTIONS.index("WAIT")] == False


def test_wait_still_allowed_with_at_least_two_steps_of_buffer():
    # Danger two steps ahead alone must not make WAIT illegal.
    field = make_field([
        "#########",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ])
    game_state = {
        "field": field,
        "self": ("me", 0, False, (2, 2)),
        "coins": [],
        "bombs": [((4, 2), 2)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("WAIT")] == True


def test_wait_still_available_alongside_other_legal_actions_in_benign_scenario():
    # With no danger, WAIT stays legal alongside every movement direction.
    field = make_field(["#####", "#...#", "#...#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (2, 2)), "coins": [(1, 1)]}
    mask = compute_action_mask(game_state)
    for direction in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
        assert mask[cfg.ACTIONS.index(direction)] == True


def test_wait_kept_as_fallback_when_no_action_would_be_legal():
    # Doomed with no escape anywhere: WAIT stays as the fallback so at least one
    # action is legal.
    field = make_field(["#####", "#...#", "#####"])
    game_state = {
        "field": field,
        "self": ("me", 0, False, (1, 1)),
        "coins": [],
        "bombs": [((3, 1), 0)],
        "explosion_map": np.zeros_like(field, dtype=float),
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("RIGHT")] == False
    assert mask[cfg.ACTIONS.index("WAIT")] == True


def test_has_bombing_target_reaches_nearest_crate_position():
    # The BFS finds a nearby bombing position that also passes the safety check.
    field = make_field(["#####", "#.X.#", "#...#", "#...#", "#####"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 3)), "coins": []}
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.nearest_bombing_distance == 1
    assert semantic.crates_destructible_at_target == 1
    assert semantic.bombing_path_dirs["RIGHT"] is True


def test_nearest_bombing_distance_feature_clipped_when_maze_geometry_exceeds_max_board_distance():
    # A BFS path around walls can exceed cfg.MAX_BOARD_DISTANCE: the raw
    # distance keeps the true value while the feature is clipped to 1.0.
    width = 45
    row_with_crate = "#" + "." * (width - 4) + "X" + "." + "#"
    field = make_field([
        "#" * width,
        row_with_crate,
        "#" + "." * (width - 2) + "#",
        "#" + "." * (width - 2) + "#",
        "#" * width,
    ])
    game_state = {"field": field, "self": ("me", 0, True, (1, 3)), "coins": []}
    from agent_code.rhine.features import features_from_semantic
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.nearest_bombing_distance > cfg.MAX_BOARD_DISTANCE
    features = features_from_semantic(semantic)
    assert features[12] == 1.0


def test_bombing_target_excluded_when_no_safe_escape(monkeypatch):
    # Every crate-hitting tile lacks a safe escape, so the safety filter
    # (enabled here explicitly) leaves no target at all.
    monkeypatch.setattr(cfg, "ENABLE_BOMBING_TARGET_SAFETY_FILTER", True)
    field = make_field(["########", "#....X.#", "########"])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is False
    assert semantic.nearest_bombing_distance is None
    assert semantic.crates_destructible_at_target == 0
    assert all(v is False for v in semantic.bombing_path_dirs.values())


def test_bombing_target_picks_higher_score_over_nearest():
    # Selection is by crates_destructible / (distance + 1): a farther
    # multi-crate cluster must beat a nearer lone crate. The layout was found
    # by automated search.
    field = make_field([
        "#############",
        "#...........#",
        "#........X..#",
        "#...........#",
        "#..........X#",
        "#...........#",
        "#...........#",
        "#........X..#",
        "#...........#",
        "#.........X.#",
        "#...........#",
        "#......X....#",
        "#############",
    ])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.nearest_bombing_distance == 11
    assert semantic.crates_destructible_at_target == 3
    assert semantic.bombing_path_dirs == {"UP": False, "DOWN": True, "LEFT": False, "RIGHT": True}


def test_bombing_target_scoring_formula_can_be_disabled(monkeypatch):
    # With the scoring formula disabled, selection reverts to the nearest
    # candidate.
    from agent_code.rhine import config as cfg
    from agent_code.rhine.state_processing import extract_semantic_state

    monkeypatch.setattr(cfg, "ENABLE_BOMBING_TARGET_SCORING_FORMULA", False)
    field = make_field([
        "#############",
        "#...........#",
        "#........X..#",
        "#...........#",
        "#..........X#",
        "#...........#",
        "#...........#",
        "#........X..#",
        "#...........#",
        "#.........X.#",
        "#...........#",
        "#......X....#",
        "#############",
    ])
    game_state = {"field": field, "self": ("me", 0, True, (1, 1)), "coins": []}

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.nearest_bombing_distance == 6
    assert semantic.crates_destructible_at_target == 1
    assert semantic.bombing_path_dirs == {"UP": False, "DOWN": True, "LEFT": False, "RIGHT": True}


def test_bombing_target_tie_break_prefers_nearer_distance():
    # Several candidates tie on score; the nearest of them must be selected.
    field = make_field([
        "##########",
        "#.X......#",
        "#........#",
        "#........#",
        "#........#",
        "#..X.X...#",
        "#...X....#",
        "#........#",
        "#........#",
        "##########",
    ])
    game_state = {"field": field, "self": ("me", 0, True, (1, 3)), "coins": []}
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.nearest_bombing_distance == 1
    assert semantic.crates_destructible_at_target == 1
    assert semantic.bombing_path_dirs["RIGHT"] is True


# --- Per-step redundancy re-check ---
#
# Shared field: self stands on its own fresh bomb in a corridor. LEFT leads
# into a branchless stretch; RIGHT has an extra side branch. Both directions
# escape the blast in time, so both pass the basic safety standard.
_REDUNDANCY_FIELD = make_field([
    "###########",
    "#.........#",
    "######.####",
    "###########",
])
_REDUNDANCY_SELF_POS = (5, 1)


def _redundancy_game_state(opponent_pos):
    return {
        "field": _REDUNDANCY_FIELD,
        "self": ("me", 0, False, _REDUNDANCY_SELF_POS),
        "coins": [],
        "bombs": [(_REDUNDANCY_SELF_POS, cfg.BOMB_TIMER)],
        "others": [("opp", 0, True, opponent_pos)] if opponent_pos else [],
    }


def test_redundancy_recheck_masks_non_redundant_direction_when_triggered():
    # An opponent near LEFT's side raises the requirement, which LEFT fails;
    # RIGHT is unreachable for the opponent and stays legal, allowing LEFT's
    # removal.
    mask = compute_action_mask(_redundancy_game_state(opponent_pos=(3, 1)))
    assert mask[cfg.ACTIONS.index("LEFT")] == False
    assert mask[cfg.ACTIONS.index("RIGHT")] == True


def test_redundancy_recheck_inactive_when_opponent_far():
    # A distant opponent must not trigger the re-check (needs a wider field).
    field_far = make_field([
        "#" * 25,
        "#" + "." * 23 + "#",
        "######." + "#" * 18,
        "#" * 25,
    ])
    game_state = {
        "field": field_far,
        "self": ("me", 0, False, _REDUNDANCY_SELF_POS),
        "coins": [],
        "bombs": [(_REDUNDANCY_SELF_POS, cfg.BOMB_TIMER)],
        "others": [("opp", 0, True, (22, 1))],
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("LEFT")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == True


def test_redundancy_recheck_no_opponent_present_is_inactive():
    # No opponents: the re-check never triggers.
    mask = compute_action_mask(_redundancy_game_state(opponent_pos=None))
    assert mask[cfg.ACTIONS.index("LEFT")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == True


def test_redundancy_recheck_never_removes_the_only_remaining_options():
    # Without RIGHT's branch neither direction meets the stricter standard, so
    # nothing is filtered and both stay legal.
    field_no_branch = make_field([
        "###########",
        "#.........#",
        "###########",
        "###########",
    ])
    game_state = {
        "field": field_no_branch,
        "self": ("me", 0, False, _REDUNDANCY_SELF_POS),
        "coins": [],
        "bombs": [(_REDUNDANCY_SELF_POS, cfg.BOMB_TIMER)],
        "others": [("opp", 0, True, (5, 0))],
    }
    mask = compute_action_mask(game_state)
    assert mask[cfg.ACTIONS.index("LEFT")] == True
    assert mask[cfg.ACTIONS.index("RIGHT")] == True


# --- kill_target group: independent of the crate bombing-target group above ---

def test_kill_target_independent_of_crate_bombing_target_and_feeds_features():
    # A crate and an opponent on different arms of the room: the crate and kill
    # target groups are populated independently and reach their feature indices.
    field = make_field(["#####", "#.X.#", "#...#", "#...#", "#####"])
    game_state = {
        "field": field, "self": ("me", 0, True, (1, 2)), "coins": [],
        "others": [("opp", 0, True, (2, 3))],
    }
    from agent_code.rhine.features import features_from_semantic
    from agent_code.rhine.state_processing import extract_semantic_state

    semantic = extract_semantic_state(game_state)
    assert semantic.has_bombing_target is True
    assert semantic.crates_destructible_at_target == 1
    assert semantic.has_kill_target is True
    assert semantic.expected_kill_value_at_target > 0.0

    features = features_from_semantic(semantic)
    assert features.shape == (cfg.MODEL_CONFIG.n_features,)
    assert features[21] == float(semantic.has_kill_target)
    assert features[22] == min(semantic.nearest_kill_distance / cfg.DISTANCE_FEATURE_NORM, 1.0)
    assert features[23] == float(semantic.kill_direction_dirs["UP"])
    assert features[24] == float(semantic.kill_direction_dirs["DOWN"])
    assert features[25] == float(semantic.kill_direction_dirs["LEFT"])
    assert features[26] == float(semantic.kill_direction_dirs["RIGHT"])
    assert features[27] == semantic.expected_kill_value_at_target
