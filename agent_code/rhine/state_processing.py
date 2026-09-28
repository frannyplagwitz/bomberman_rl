"""Semantic state extraction shared by features.py and action_mask.py.

Parses the raw `game_state` into model-independent information: position,
traversability, coin/bombing/kill-target BFS, bomb-danger timing, opponent
proximity and coin competition.
"""
import itertools
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from . import config as cfg

DIRECTIONS: Tuple[str, str, str, str] = ("UP", "DOWN", "LEFT", "RIGHT")
_MOVE_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

_DIRECTION_OFFSETS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

Coord = Tuple[int, int]


@dataclass
class SemanticState:
    self_pos: Coord
    can_move: Dict[str, bool]
    has_reachable_coin: bool
    nearest_coin_distance: Optional[int]
    coin_path_dirs: Dict[str, bool] = field(
        default_factory=lambda: {d: False for d in DIRECTIONS}
    )
    bomb_available: bool = False
    has_bombing_target: bool = False
    nearest_bombing_distance: Optional[int] = None
    bombing_path_dirs: Dict[str, bool] = field(
        default_factory=lambda: {d: False for d in DIRECTIONS}
    )
    crates_destructible_at_target: int = 0
    current_tile_in_danger: bool = False
    nearest_threat_timer: int = 0
    # Coin resource-competition flag.
    coin_contested: bool = False
    # Tile whose blast would cover an opponent's current position (see
    # kill_target_info()); scored independently of the crate bombing target.
    has_kill_target: bool = False
    nearest_kill_distance: Optional[int] = None
    kill_direction_dirs: Dict[str, bool] = field(
        default_factory=lambda: {d: False for d in DIRECTIONS}
    )
    expected_kill_value_at_target: float = 0.0
    # Proximity to reachable opponents (alive_opponent_distances());
    # None/0 when none is reachable.
    nearest_alive_opponent_distance: Optional[int] = None
    opponents_within_3: int = 0
    # Local mobility ignoring bombs/danger (reachable_space_count()).
    reachable_space: int = 0
    # Raw parsed data reused by action_mask.py's escape-route search.
    field_arr: np.ndarray = None
    blocked: FrozenSet[Coord] = frozenset()
    danger_offsets: Dict[Coord, Set[int]] = field(default_factory=dict)
    # Opponents' current positions, used by the escape-route checks.
    opponents: List[Coord] = field(default_factory=list)
    # All remaining coins, not reachability-filtered, so "none left" can be
    # told apart from "none reachable".
    coins_remaining: int = 0


# Shared by train.py's stall-v2 penalty and the optional stall-history feature.
def is_confined_to_small_range(
    position_history, window_size: int = 4, max_distinct: int = 2
) -> bool:
    """True iff the history holds at least `window_size` positions
    (most recent last) and the last `window_size` of them cover at most
    `max_distinct` tiles. A shorter history never counts as confined.
    """
    if len(position_history) < window_size:
        return False
    recent = list(position_history)[-window_size:]
    return len(set(recent)) <= max_distinct


def _to_coord(raw) -> Coord:
    return int(raw[0]), int(raw[1])


def is_free(field_arr: np.ndarray, x: int, y: int, blocked: FrozenSet[Coord] = frozenset()) -> bool:
    """A tile is free if it is open floor (not wall/crate) and not occupied
    by a bomb or opponent (`blocked`), matching environment.py's tile_is_free.
    """
    return bool(field_arr[x, y] == 0) and (x, y) not in blocked


def neighbor_tile(pos: Coord, direction: str) -> Coord:
    dx, dy = _DIRECTION_OFFSETS[direction]
    return pos[0] + dx, pos[1] + dy


def legal_moves(field_arr: np.ndarray, pos: Coord, blocked: FrozenSet[Coord] = frozenset()) -> Dict[str, bool]:
    """can_move_up/down/left/right, shared by mask/feature/pathfinding."""
    return {d: is_free(field_arr, *neighbor_tile(pos, d), blocked=blocked) for d in DIRECTIONS}


def bfs_distances(field_arr: np.ndarray, start: Coord, blocked: FrozenSet[Coord] = frozenset()) -> Dict[Coord, int]:
    """BFS distances over free tiles from `start`."""
    dist: Dict[Coord, int] = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for d in DIRECTIONS:
            nxt = neighbor_tile(current, d)
            if nxt not in dist and is_free(field_arr, *nxt, blocked=blocked):
                dist[nxt] = dist[current] + 1
                queue.append(nxt)
    return dist


def _extract_coins(game_state: dict) -> List[Coord]:
    return [_to_coord(c) for c in game_state["coins"]]


def _bomb_positions(game_state: dict) -> FrozenSet[Coord]:
    return frozenset(_to_coord(pos) for pos, _timer in game_state.get("bombs", []))


def _opponent_positions(game_state: dict) -> FrozenSet[Coord]:
    """Opponents' current positions, treated as temporary obstacles like bombs."""
    return frozenset(_to_coord(o[3]) for o in game_state.get("others", []))


def classify_invalid_action(prev_game_state: dict, action: str, new_game_state: dict) -> Optional[str]:
    """Classifies whether `action`, chosen from `prev_game_state`, failed to
    take effect by `new_game_state`, using only information known at
    decision time.

    Returns:
        None if the action took effect (WAIT never fails).
        "own_cause" if the target tile was already blocked, or BOMB was
            chosen without a bomb available -- a masking failure.
        "contested_tile" if the target was free, so another agent must have
            moved onto it the same tick -- a framework-level race unrelated
            to mask correctness. BOMB never falls into this case.
    """
    if action == "BOMB":
        prev_bomb_available = bool(prev_game_state["self"][2])
        return None if prev_bomb_available else "own_cause"
    if action not in DIRECTIONS:
        return None

    prev_pos = _to_coord(prev_game_state["self"][3])
    new_pos = _to_coord(new_game_state["self"][3])
    if new_pos != prev_pos:
        return None

    target = neighbor_tile(prev_pos, action)
    field_arr = prev_game_state["field"]
    if field_arr[target[0], target[1]] != 0:
        return "own_cause"
    if target in _bomb_positions(prev_game_state):
        return "own_cause"
    if target in _opponent_positions(prev_game_state):
        return "own_cause"
    return "contested_tile"


def nearest_reachable_coin_distance(game_state: dict) -> Optional[int]:
    """Nearest reachable coin distance for rewards.py's potential shaping;
    skips the direction computation only the feature vector needs.
    """
    field_arr = game_state["field"]
    self_pos = _to_coord(game_state["self"][3])
    coins = _extract_coins(game_state)
    if not coins:
        return None
    blocked = _bomb_positions(game_state)
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    reachable = [dist_map[c] for c in coins if c in dist_map]
    return min(reachable) if reachable else None


# --- Blast radius / crate / bombing-position / danger ---

def blast_coords(field_arr: np.ndarray, pos: Coord, power: int) -> List[Coord]:
    """Blast tiles of a bomb at `pos`, matching environment.py's
    Bomb.get_blast_coords: stone walls stop propagation, crates do not.
    """
    x, y = pos
    coords = [pos]
    for dx, dy in _MOVE_DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if field_arr[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def crates_in_blast(field_arr: np.ndarray, pos: Coord, power: int) -> int:
    return sum(1 for (x, y) in blast_coords(field_arr, pos, power) if field_arr[x, y] == 1)


def bombing_target_info(
    field_arr: np.ndarray,
    self_pos: Coord,
    blocked: FrozenSet[Coord],
    power: int,
    danger_offsets: Dict[Coord, Set[int]],
    opponents: List[Coord],
):
    """Finds the best reachable tile from which a bomb would destroy at least
    one crate (self_pos included, at distance 0).

    With cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA, candidates are ranked by
    crates_destructible / (distance + 1) using exact fractions; otherwise by
    distance alone. Remaining ties go to the smaller distance, and every
    first-step direction on a shortest path to any tied tile is marked. With
    cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER, candidates also need a safe
    escape (has_safe_escape_after_bombing(), using `opponents`).

    Returns:
        (has_target, distance, path_dirs, crates_at_target); the crate count
        is the maximum over the tied tiles. No candidate -> (False, None,
        all-False dirs, 0).
    """
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    candidates: Dict[Coord, int] = {}
    for tile, dist in dist_map.items():
        n = crates_in_blast(field_arr, tile, power)
        if n < 1:
            continue
        if cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER and not has_safe_escape_after_bombing(
            tile, field_arr, blocked, danger_offsets, power, opponents
        ):
            continue
        candidates[tile] = n

    if not candidates:
        return False, None, {d: False for d in DIRECTIONS}, 0

    if cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA:
        scores = {t: Fraction(n, dist_map[t] + 1) for t, n in candidates.items()}
        best_score = max(scores.values())
        best_by_score = [t for t in candidates if scores[t] == best_score]
        best_distance = min(dist_map[t] for t in best_by_score)
        tied = [t for t in best_by_score if dist_map[t] == best_distance]
    else:
        # Nearest distance only, ignoring crate count.
        best_distance = min(dist_map[t] for t in candidates)
        tied = [t for t in candidates if dist_map[t] == best_distance]
    crates_at_target = max(candidates[t] for t in tied)

    can_move = legal_moves(field_arr, self_pos, blocked=blocked)
    path_dirs = {d: False for d in DIRECTIONS}
    for target in tied:
        dist_from_target = bfs_distances(field_arr, target, blocked=blocked)
        for d in DIRECTIONS:
            if path_dirs[d] or not can_move[d]:
                continue
            nxt = neighbor_tile(self_pos, d)
            if dist_from_target.get(nxt) == best_distance - 1:
                path_dirs[d] = True

    return True, best_distance, path_dirs, crates_at_target


def nearest_bombing_position_distance(game_state: dict, power: int) -> Optional[int]:
    """Bombing-target distance for rewards.py. Delegates to
    bombing_target_info() so the reward potential and the features always
    refer to the same target.
    """
    field_arr = game_state["field"]
    self_pos = _to_coord(game_state["self"][3])
    bombs = game_state.get("bombs", [])
    explosion_map = game_state.get("explosion_map")
    if explosion_map is None:
        explosion_map = np.zeros_like(field_arr)
    blocked = _bomb_positions(game_state)
    danger_offsets = compute_danger_offsets(field_arr, bombs, explosion_map, power)
    opponents = list(_opponent_positions(game_state))
    _, distance, _, _ = bombing_target_info(field_arr, self_pos, blocked, power, danger_offsets, opponents)
    return distance


def bomb_threatens_reachable_opponent(game_state: dict, pos: Coord, power: int) -> bool:
    """True iff a bomb at `pos` would cover some opponent's current position.
    Snapshot check; does not predict whether the opponent escapes in time.
    """
    opponents = list(_opponent_positions(game_state))
    if not opponents:
        return False
    field_arr = game_state["field"]
    blast = set(blast_coords(field_arr, pos, power))
    return any(opp in blast for opp in opponents)


def compute_danger_offsets(
    field_arr: np.ndarray, bombs: List[Tuple[Coord, int]], explosion_map: np.ndarray, power: int
) -> Dict[Coord, Set[int]]:
    """Merges pending bombs (future danger) and `explosion_map` (current
    danger) into tile -> set of step offsets at which standing there is
    lethal (offset 0 = the step about to be decided).

    A bomb with timer t explodes t steps from now and stays lethal one step
    longer, hence offsets {t, t+1} (derived from environment.py's update
    order). Nonzero explosion_map tiles are lethal at offset 0.
    """
    danger: Dict[Coord, Set[int]] = {}
    for pos, timer in bombs:
        for tile in blast_coords(field_arr, _to_coord(pos), power):
            danger.setdefault(tile, set()).update({int(timer), int(timer) + 1})

    xs, ys = np.nonzero(explosion_map > 0)
    for x, y in zip(xs.tolist(), ys.tolist()):
        danger.setdefault((x, y), set()).add(0)

    return danger


def find_safe_path(
    start: Coord,
    start_offset: int,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    max_offset: int,
) -> Optional[List[Tuple[Coord, int]]]:
    """Time-expanded BFS over (tile, offset) states for an escape from
    `start` (occupied at `start_offset`): never stands on a tile at a lethal
    offset and, within `max_offset`, reaches a tile with no later danger.
    WAIT counts as staying in place.

    Returns:
        The shortest (tile, offset) sequence from start to the first safe
        tile, both ends inclusive, or None if none exists. The route may
        turn off-axis rather than follow one direction.
    """
    if start_offset in danger_offsets.get(start, ()):
        return None
    if not {o for o in danger_offsets.get(start, ()) if o > start_offset}:
        return [(start, start_offset)]

    visited = {(start, start_offset): None}
    queue = deque([(start, start_offset)])
    while queue:
        pos, offset = queue.popleft()
        if offset >= max_offset:
            continue
        next_offset = offset + 1
        candidates = [pos] + [
            neighbor_tile(pos, d)
            for d in DIRECTIONS
            if is_free(field_arr, *neighbor_tile(pos, d), blocked=blocked)
        ]
        for nxt in candidates:
            if next_offset in danger_offsets.get(nxt, ()):
                continue
            state = (nxt, next_offset)
            if state in visited:
                continue
            visited[state] = (pos, offset)
            if not {o for o in danger_offsets.get(nxt, ()) if o > next_offset}:
                path = [state]
                cur = (pos, offset)
                while cur is not None:
                    path.append(cur)
                    cur = visited[cur]
                path.reverse()
                return path
            queue.append(state)
    return None


def exists_safe_path(
    start: Coord,
    start_offset: int,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    max_offset: int,
) -> bool:
    """Bool-only view of find_safe_path()."""
    return find_safe_path(start, start_offset, field_arr, blocked, danger_offsets, max_offset) is not None


# Search horizon for escape checks: covers the longest a known bomb stays
# lethal, plus one step of margin.
SAFETY_HORIZON = cfg.BOMB_TIMER + 2

def _opponent_distance_maps(
    field_arr: np.ndarray, blocked: FrozenSet[Coord], opponents: List[Coord],
) -> Dict[Coord, Dict[Coord, int]]:
    """Per-opponent BFS distance map from its current position. Each
    opponent's own tile is removed from `blocked` so it can path out of it.
    """
    return {
        opp: bfs_distances(field_arr, opp, blocked=blocked - {opp})
        for opp in opponents
    }


def _path_conflicts(
    path: List[Tuple[Coord, int]], opponent_dist_maps: Dict[Coord, Dict[Coord, int]],
) -> Dict[Coord, Set[Coord]]:
    """Returns {tile: opponents} for every tile on `path` that some
    opponent could reach (current-position BFS, no prediction) no later than
    the path occupies it.
    """
    conflicts: Dict[Coord, Set[Coord]] = {}
    for tile, offset in path:
        hit = {opp for opp, dmap in opponent_dist_maps.items() if dmap.get(tile, float("inf")) <= offset}
        if hit:
            conflicts[tile] = hit
    return conflicts


# A tile has only four movement directions to require.
REQUIRED_ESCAPE_DIRECTIONS_CAP = 4


def _candidate_escape_paths(
    pos: Coord,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    start_offset: int,
) -> Dict[str, List[Tuple[Coord, int]]]:
    """Per-direction safe escape path from `pos` (occupied at
    `start_offset`); directions without one are omitted.
    """
    paths: Dict[str, List[Tuple[Coord, int]]] = {}
    for d in DIRECTIONS:
        neighbor = neighbor_tile(pos, d)
        if not is_free(field_arr, *neighbor, blocked=blocked):
            continue
        path = find_safe_path(neighbor, start_offset + 1, field_arr, blocked, danger_offsets, SAFETY_HORIZON)
        if path is not None:
            paths[d] = path
    return paths


def _has_sufficient_escape_directions(
    pos: Coord,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    opponents: List[Coord],
    start_offset: int,
) -> bool:
    """Whether `pos` (occupied at `start_offset`) has at least N+1 escape
    directions, where N is the number of distinct opponents conflicting with
    any candidate path (_path_conflicts()).

    The chosen paths must be pairwise conflict-disjoint: no path's conflict
    tiles may lie on another chosen path, so one opponent cannot cut off two
    routes at once. Non-conflict tiles may be shared. With N=0 this reduces
    to "at least one safe path exists".
    """
    candidate_paths = _candidate_escape_paths(pos, field_arr, blocked, danger_offsets, start_offset)
    if not candidate_paths:
        return False

    opponent_dist_maps = _opponent_distance_maps(field_arr, blocked, opponents)
    path_conflicts = {d: _path_conflicts(path, opponent_dist_maps) for d, path in candidate_paths.items()}

    all_conflicting_opponents: Set[Coord] = set()
    for conflicts in path_conflicts.values():
        for opp_set in conflicts.values():
            all_conflicting_opponents.update(opp_set)

    required = min(REQUIRED_ESCAPE_DIRECTIONS_CAP, 1 + len(all_conflicting_opponents))
    if required <= 1:
        return True
    if len(candidate_paths) < required:
        return False

    dirs = list(candidate_paths)
    tiles_of = {d: {tile for tile, _ in candidate_paths[d]} for d in dirs}
    conflict_tiles_of = {d: set(path_conflicts[d].keys()) for d in dirs}

    def _compatible(d1: str, d2: str) -> bool:
        return not (conflict_tiles_of[d1] & tiles_of[d2]) and not (conflict_tiles_of[d2] & tiles_of[d1])

    for combo in itertools.combinations(dirs, required):
        if all(_compatible(a, b) for a, b in itertools.combinations(combo, 2)):
            return True
    return False


def has_safe_escape_after_bombing(
    tile: Coord,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    power: int,
    opponents: List[Coord],
) -> bool:
    """True iff bombing at `tile` now leaves a sufficiently robust escape.

    Adds the new bomb's blast to the danger map, blocks `tile` itself, rejects
    immediate danger on `tile`, then applies the same N+1-independent-paths
    standard as the mask's per-step re-check
    (_has_sufficient_escape_directions()).

    Known limitation: uses opponents' current positions only, so later
    opponent movement or a same-tick race for an adjacent tile can still
    invalidate a route judged safe here.
    """
    hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
    for blast_tile in blast_coords(field_arr, tile, power):
        hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    blocked_with_new_bomb = blocked | {tile}

    if 0 in hypothetical_danger.get(tile, ()):
        return False

    return _has_sufficient_escape_directions(
        tile, field_arr, blocked_with_new_bomb, hypothetical_danger, opponents, start_offset=0,
    )


def _opponent_escape_difficulty(
    opp: Coord,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    hypothetical_danger: Dict[Coord, Set[int]],
    self_pos: Coord,
    other_opponents: List[Coord],
) -> Fraction:
    """Escape-difficulty score in [0, 1] for opponent `opp` standing in a
    candidate bomb's blast.

    Uses the same per-direction escape search rooted at `opp`: its own tile
    is unblocked and self_pos is blocked; `other_opponents` act as chasers
    that can create path conflicts. Score is the fraction of the four
    directions without a conflict-free escape. A spatial proxy from one
    snapshot, not a death probability.
    """
    opp_blocked = (blocked - {opp}) | {self_pos}
    paths = _candidate_escape_paths(opp, field_arr, opp_blocked, hypothetical_danger, start_offset=0)
    if not paths:
        return Fraction(1)
    opponent_dist_maps = _opponent_distance_maps(field_arr, opp_blocked, other_opponents)
    conflict_free = sum(1 for path in paths.values() if not _path_conflicts(path, opponent_dist_maps))
    return Fraction(len(DIRECTIONS) - conflict_free, len(DIRECTIONS))


def kill_target_info(
    field_arr: np.ndarray,
    self_pos: Coord,
    blocked: FrozenSet[Coord],
    power: int,
    danger_offsets: Dict[Coord, Set[int]],
    opponents: List[Coord],
):
    """Finds the best reachable tile from which a safe bomb would cover at
    least one opponent's current position (a snapshot, not a kill
    probability). Independent of crate targets.

    A tile's value is the maximum escape difficulty
    (_opponent_escape_difficulty()) among the opponents it covers. Selection
    mirrors bombing_target_info(): value / (distance + 1) with exact
    fractions, then distance, then all shortest-path first steps.

    Returns:
        (has_target, distance, path_dirs, expected_kill_value).
    """
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    candidates: Dict[Coord, Fraction] = {}
    for tile, dist in dist_map.items():
        blast = set(blast_coords(field_arr, tile, power))
        covered = [opp for opp in opponents if opp in blast]
        if not covered:
            continue
        if not has_safe_escape_after_bombing(tile, field_arr, blocked, danger_offsets, power, opponents):
            continue

        hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
        for blast_tile in blast:
            hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})

        difficulties = [
            _opponent_escape_difficulty(
                opp, field_arr, blocked, hypothetical_danger, self_pos,
                [o for o in opponents if o != opp],
            )
            for opp in covered
        ]
        candidates[tile] = max(difficulties)

    if not candidates:
        return False, None, {d: False for d in DIRECTIONS}, 0.0

    scores = {t: v / (dist_map[t] + 1) for t, v in candidates.items()}
    best_score = max(scores.values())
    best_by_score = [t for t in candidates if scores[t] == best_score]
    best_distance = min(dist_map[t] for t in best_by_score)
    tied = [t for t in best_by_score if dist_map[t] == best_distance]
    expected_value_at_target = float(max(candidates[t] for t in tied))

    can_move = legal_moves(field_arr, self_pos, blocked=blocked)
    path_dirs = {d: False for d in DIRECTIONS}
    for target in tied:
        dist_from_target = bfs_distances(field_arr, target, blocked=blocked)
        for d in DIRECTIONS:
            if path_dirs[d] or not can_move[d]:
                continue
            nxt = neighbor_tile(self_pos, d)
            if dist_from_target.get(nxt) == best_distance - 1:
                path_dirs[d] = True

    return True, best_distance, path_dirs, expected_value_at_target


def alive_opponent_distances(
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    dist_map: Dict[Coord, int],
    opponents: List[Coord],
) -> Dict[Coord, int]:
    """Per-opponent BFS distance from self_pos, keyed by opponent coord.

    Opponent tiles are blocked, so an opponent's distance is one more than
    its nearest free neighbor's distance in `dist_map`. Opponents with no
    reachable free neighbor are omitted.
    """
    distances: Dict[Coord, int] = {}
    for opp in opponents:
        candidate_dists = []
        for d in DIRECTIONS:
            nxt = neighbor_tile(opp, d)
            if is_free(field_arr, *nxt, blocked=blocked) and nxt in dist_map:
                candidate_dists.append(dist_map[nxt])
        if candidate_dists:
            distances[opp] = min(candidate_dists) + 1
    return distances


def reachable_space_count(field_arr: np.ndarray, self_pos: Coord, opponents: List[Coord], depth_cap: int) -> int:
    """Number of tiles (self_pos included) within `depth_cap` BFS steps,
    treating walls, crates and opponents as obstacles; bombs and danger are
    ignored.
    """
    blocked = frozenset(opponents)
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    return sum(1 for d in dist_map.values() if d <= depth_cap)


def extract_semantic_state(game_state: dict) -> SemanticState:
    field_arr = game_state["field"]
    self_pos = _to_coord(game_state["self"][3])
    bomb_available = bool(game_state["self"][2])
    coins = _extract_coins(game_state)
    bombs = game_state.get("bombs", [])
    opponents = [_to_coord(o[3]) for o in game_state.get("others", [])]
    explosion_map = game_state.get("explosion_map")
    if explosion_map is None:
        explosion_map = np.zeros_like(field_arr)

    # Opponents join bombs as temporary obstacles for every BFS below.
    blocked = _bomb_positions(game_state) | frozenset(opponents)
    can_move = legal_moves(field_arr, self_pos, blocked=blocked)

    dist_from_self = bfs_distances(field_arr, self_pos, blocked=blocked)
    reachable_coins = [c for c in coins if c in dist_from_self]

    coin_path_dirs = {d: False for d in DIRECTIONS}
    nearest_distance: Optional[int] = None
    coin_contested = False

    if reachable_coins:
        nearest_distance = min(dist_from_self[c] for c in reachable_coins)
        # Ties at the minimum distance mark every first step on a shortest
        # path to any tied coin rather than picking one arbitrarily.
        tied_coins = [c for c in reachable_coins if dist_from_self[c] == nearest_distance]
        for coin in tied_coins:
            dist_from_coin = bfs_distances(field_arr, coin, blocked=blocked)
            for d in DIRECTIONS:
                if coin_path_dirs[d] or not can_move[d]:
                    continue
                nxt = neighbor_tile(self_pos, d)
                if dist_from_coin.get(nxt) == nearest_distance - 1:
                    coin_path_dirs[d] = True

        # Pure same-step distance comparison; no intent modeling.
        for opp in opponents:
            dist_from_opp = bfs_distances(field_arr, opp, blocked=blocked)
            opp_coin_dists = [dist_from_opp[c] for c in tied_coins if c in dist_from_opp]
            if opp_coin_dists and min(opp_coin_dists) <= nearest_distance:
                coin_contested = True
                break

    danger_offsets = compute_danger_offsets(field_arr, bombs, explosion_map, cfg.BOMB_POWER)

    has_bombing_target, nearest_bombing_distance, bombing_path_dirs, crates_at_target = bombing_target_info(
        field_arr, self_pos, blocked, cfg.BOMB_POWER, danger_offsets, opponents
    )
    self_danger = danger_offsets.get(self_pos, set())
    current_tile_in_danger = bool(self_danger)
    nearest_threat_timer = min(self_danger) if self_danger else 0

    has_kill_target, nearest_kill_distance, kill_direction_dirs, expected_kill_value_at_target = kill_target_info(
        field_arr, self_pos, blocked, cfg.BOMB_POWER, danger_offsets, opponents
    )

    opponent_distances = alive_opponent_distances(field_arr, blocked, dist_from_self, opponents)
    nearest_alive_opponent_distance = min(opponent_distances.values()) if opponent_distances else None
    opponents_within_3 = sum(
        1 for d in opponent_distances.values() if d <= cfg.OPPONENTS_WITHIN_RANGE_THRESHOLD
    )
    reachable_space = reachable_space_count(field_arr, self_pos, opponents, cfg.REACHABLE_SPACE_DEPTH_CAP)

    return SemanticState(
        self_pos=self_pos,
        can_move=can_move,
        has_reachable_coin=bool(reachable_coins),
        nearest_coin_distance=nearest_distance,
        coin_path_dirs=coin_path_dirs,
        bomb_available=bomb_available,
        has_bombing_target=has_bombing_target,
        nearest_bombing_distance=nearest_bombing_distance,
        bombing_path_dirs=bombing_path_dirs,
        crates_destructible_at_target=crates_at_target,
        current_tile_in_danger=current_tile_in_danger,
        nearest_threat_timer=nearest_threat_timer,
        coin_contested=coin_contested,
        has_kill_target=has_kill_target,
        nearest_kill_distance=nearest_kill_distance,
        kill_direction_dirs=kill_direction_dirs,
        expected_kill_value_at_target=expected_kill_value_at_target,
        nearest_alive_opponent_distance=nearest_alive_opponent_distance,
        opponents_within_3=opponents_within_3,
        reachable_space=reachable_space,
        field_arr=field_arr,
        blocked=blocked,
        danger_offsets=danger_offsets,
        opponents=opponents,
        coins_remaining=len(coins),
    )
