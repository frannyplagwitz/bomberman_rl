"""Shared Semantic Extractor.

Parses the raw `game_state` dict into model-independent semantic
information: self position, traversability, legal movement, coin BFS
pathfinding, crate/bomb-aware traversability, bombing-position BFS,
bomb-danger timing, opponent-aware traversability, and coin
resource-competition. Reused as-is by features.py (1D adapter) and
action_mask.py.
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
    # Kill-target: a candidate bombing tile whose blast would catch an
    # opponent's current position (see kill_target_info()), scored
    # independently of has_bombing_target/crates_destructible_at_target --
    # not fused into one combined score.
    has_kill_target: bool = False
    nearest_kill_distance: Optional[int] = None
    kill_direction_dirs: Dict[str, bool] = field(
        default_factory=lambda: {d: False for d in DIRECTIONS}
    )
    expected_kill_value_at_target: float = 0.0
    # Nearest-alive-opponent proximity/crowding, independent of has_kill_target
    # (a reachable opponent need not be a valid kill target). See
    # alive_opponent_distances() -- None/0 when no opponent is reachable this
    # way, including no opponents at all.
    nearest_alive_opponent_distance: Optional[int] = None
    opponents_within_3: int = 0
    # Local mobility around self_pos, ignoring bombs/danger -- see
    # reachable_space_count().
    reachable_space: int = 0
    # Raw information kept around (not exposed as scalar features) so
    # action_mask.py can run its own escape-route search on the same
    # already-parsed representation.
    field_arr: np.ndarray = None
    blocked: FrozenSet[Coord] = frozenset()
    danger_offsets: Dict[Coord, Set[int]] = field(default_factory=dict)
    # Opponents' raw current positions, exposed the same way as
    # blocked/danger_offsets so action_mask.py's
    # has_safe_escape_after_bombing() call can determine whether an
    # opponent is close enough to warrant the stricter >=2-escape-direction
    # requirement -- see that function's docstring.
    opponents: List[Coord] = field(default_factory=list)
    # Total remaining collectable coins on the board (not reachability-filtered,
    # unlike has_reachable_coin/nearest_coin_distance above) -- kept around so
    # action_mask.py's ENABLE_NO_BOMB_WHEN_BOARD_CLEARED check can tell "no
    # coins left anywhere" apart from "no coins currently reachable".
    coins_remaining: int = 0


# Position-history utility shared by train.py's stall-v2 reward penalty
# (_update_stall_v2) and the optional stall-history feature
# (config.ENABLE_STALL_HISTORY_FEATURE).
def is_confined_to_small_range(
    position_history, window_size: int = 4, max_distinct: int = 2
) -> bool:
    """True iff `position_history` (any sequence/deque of recent positions,
    most-recent-last) has accumulated at least `window_size` entries AND,
    restricted to the last `window_size` of them, visited at most
    `max_distinct` distinct tiles. A short history always returns False --
    trivially satisfying "few distinct tiles" just because there isn't much
    history yet is not the "confined to a small range" situation this flags.
    """
    if len(position_history) < window_size:
        return False
    recent = list(position_history)[-window_size:]
    return len(set(recent)) <= max_distinct


def _to_coord(raw) -> Coord:
    return int(raw[0]), int(raw[1])


def is_free(field_arr: np.ndarray, x: int, y: int, blocked: FrozenSet[Coord] = frozenset()) -> bool:
    """Traversability: a free tile is field==0 (walls=-1 and crates=1 both
    block movement), and not currently occupied by a bomb or opponent
    (`blocked`, matches environment.py's tile_is_free).
    """
    return bool(field_arr[x, y] == 0) and (x, y) not in blocked


def neighbor_tile(pos: Coord, direction: str) -> Coord:
    dx, dy = _DIRECTION_OFFSETS[direction]
    return pos[0] + dx, pos[1] + dy


def legal_moves(field_arr: np.ndarray, pos: Coord, blocked: FrozenSet[Coord] = frozenset()) -> Dict[str, bool]:
    """can_move_up/down/left/right, shared by mask/feature/pathfinding."""
    return {d: is_free(field_arr, *neighbor_tile(pos, d), blocked=blocked) for d in DIRECTIONS}


def bfs_distances(field_arr: np.ndarray, start: Coord, blocked: FrozenSet[Coord] = frozenset()) -> Dict[Coord, int]:
    """Plain BFS over free tiles (real BFS distance, not Euclidean/Manhattan)."""
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
    """Opponents' current positions, treated as temporary obstacles exactly
    like bombs -- folded into the same `blocked` set consumed by
    is_free()/legal_moves()/bfs_distances()/exists_safe_path(), so movement
    mask and escape-route BFS pick this up with no separate handling.
    """
    return frozenset(_to_coord(o[3]) for o in game_state.get("others", []))


def classify_invalid_action(prev_game_state: dict, action: str, new_game_state: dict) -> Optional[str]:
    """Did `action`, chosen from `prev_game_state`, actually take effect by
    `new_game_state`? Returns None if it did (or if `action` is WAIT, which
    can never be invalid -- environment.py's perform_agent_action() always
    accepts it).

    Otherwise classifies why it didn't, using only information already known
    from `prev_game_state` (the state as of the decision, before any agent's
    move resolves that tick) -- no environment.py instrumentation needed:
    - "own_cause": the target tile was already blocked (wall/crate/bomb/a
      surviving opponent's then-current position) or, for BOMB, bombs_left
      was already False -- a genuine masking/judgment failure, should never
      happen when the mask is respected.
    - "contested_tile": the target tile was free in prev_game_state, so the
      only way this could still fail is another agent moving onto it the
      same tick (environment.py resolves agents sequentially in a per-step
      random order) -- a framework-level residual risk unrelated to
      mask/judgment correctness.

    BOMB has no "contested_tile" case: placing a bomb only requires
    bombs_left, never an empty target tile.
    """
    if action == "BOMB":
        prev_bomb_available = bool(prev_game_state["self"][2])
        return None if prev_bomb_available else "own_cause"
    if action not in DIRECTIONS:
        return None  # WAIT (or any other non-movement action) is never invalid.

    prev_pos = _to_coord(prev_game_state["self"][3])
    new_pos = _to_coord(new_game_state["self"][3])
    if new_pos != prev_pos:
        return None  # Moved as expected.

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
    """Lightweight variant used by rewards.py (Phi(s) = -d(s)); skips the
    per-tied-coin direction computation that only act()'s feature vector needs.
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
    """Matches environment.py Bomb.get_blast_coords exactly: propagates up to
    `power` tiles in each of the 4 directions, stopping only at a stone wall
    (-1). Crates do not stop propagation (they are destroyed by it).
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
    """BFS over reachable tiles for the tile from which placing a bomb would
    destroy >=1 crate (self_pos itself is a valid candidate, distance 0).

    Among the safety-filtered candidates, selects the one maximizing
    score = crates_destructible / (distance + 1), so a farther position
    destroying more crates can outscore a very close position destroying
    only one. Score is compared via exact fractions to avoid
    float-equality edge cases. Ties in score are broken by smaller distance,
    then by marking every first-step direction lying on some shortest path
    to any of the remaining tied positions. `crates_destructible_at_target`
    reports the maximum crate count among the final tied set.

    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA toggles between this scoring
    rule and the original nearest-distance-only rule.

    A candidate must also have a safe escape route
    (has_safe_escape_after_bombing) to count as a valid bombing target. If
    every crate-hitting candidate fails the safety check, falls through to
    the "no target" return (has_bombing_target=False), the same fallback
    used when no candidate hits any crate at all.

    `opponents` is passed straight through to has_safe_escape_after_bombing()
    for each candidate -- see that function's docstring for the dynamic
    escape-direction requirement based on opponent proximity to the
    candidate tile.
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
    """Lightweight variant for rewards.py, mirroring
    nearest_reachable_coin_distance. Delegates to bombing_target_info()
    itself (same safety filter, same score-based selection) instead of an
    independent implementation, so the reward-shaping potential and the
    feature vector's bombing-target fields can't target different tiles.

    `opponents` is passed from `_opponent_positions(game_state)` for
    consistency with every other call site; this function's own `blocked`
    set (unlike extract_semantic_state()'s) does not itself include
    opponent positions.
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
    """True iff a bomb placed at `pos` would catch >=1 opponent's current
    position within its blast radius, for rewards.py's wasteful-bomb
    penalty. Pure snapshot check: compares each opponent's coordinate at
    this instant against blast_coords(pos, power), with no adjacency-
    reachability pre-filter and no prediction of subsequent opponent
    movement -- an opponent that would still walk out of the blast before
    it detonates is not modeled here.
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
    """Combines the `bombs` list (not yet exploded -- future danger) and
    `explosion_map` (already exploded, still lethal -- current danger) into
    one map: tile -> set of step-offsets (0 = this upcoming decision) at
    which standing on that tile is lethal.

    Bomb timing (derived from environment.py's do_step/update_bombs/
    update_explosions ordering): a bomb observed with timer=t explodes t
    steps from now (t=0 means it explodes as part of resolving the action
    about to be chosen), and remains lethal for one further lingering step
    (EXPLOSION_TIMER=2 in settings.py: 1 explosion step + 1 lingering step)
    -- hence offsets {t, t+1}. explosion_map[x, y] (only populated for
    currently-lethal explosions) is nonzero exactly when that tile is
    lethal for the step this game_state snapshot is used to decide, i.e. it
    always corresponds to offset 0.
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
    """Does there exist a movement sequence (WAIT counts as "stay") starting
    at `start` (occupied at `start_offset`) that never occupies a tile at a
    step-offset when it's in danger_offsets, and reaches, within max_offset
    steps, a tile with no more recorded danger at any later offset? Time-
    expanded BFS over (tile, offset) states; board is 17x17 and max_offset
    is small (BOMB_TIMER-bounded), so this is cheap enough to run inside
    act()'s per-step time budget.

    Returns the (tile, offset) sequence from start to the first tile
    verified safe (inclusive of both ends), or None if no such sequence
    exists. Path-reconstructing so callers can identify exactly which tiles
    a given escape route passes through and when (see
    `_opponent_conflicts_on_path()`), not just whether one exists.

    Finds the shortest such sequence, not one confined to a single named
    direction -- in an open (non-1-wide-corridor) area this can legitimately
    cut off-axis (an L-shaped detour) rather than continuing straight past a
    blast line, since only tiles on the blast's own straight lines are ever
    dangerous. Expected behavior, not a bug.
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
    """Bool-only view of find_safe_path(), for callers that don't need the
    actual path (WAIT legality, plain per-direction movement legality)."""
    return find_safe_path(start, start_offset, field_arr, blocked, danger_offsets, max_offset) is not None


# Search horizon for exists_safe_path: a bomb is lethal at offsets
# {timer, timer+1}, and only one of this agent's own bombs can be in flight
# at a time, so BOMB_TIMER+1 is the longest any known danger can persist;
# +1 more for an off-by-one margin. Shared by action_mask.py's
# BOMB/movement/WAIT checks and bombing_target_info()'s safety check.
SAFETY_HORIZON = cfg.BOMB_TIMER + 2

def _opponent_distance_maps(
    field_arr: np.ndarray, blocked: FrozenSet[Coord], opponents: List[Coord],
) -> Dict[Coord, Dict[Coord, int]]:
    """Per-opponent real BFS distance map from its current position, keyed by
    opponent coord. Each opponent's own tile is excluded from `blocked` for
    its own BFS root (an opponent standing on a tile that's otherwise a
    member of `blocked`, e.g. a just-placed bomb tile if opponents overlap
    it, must still be able to path out from where it actually stands).
    """
    return {
        opp: bfs_distances(field_arr, opp, blocked=blocked - {opp})
        for opp in opponents
    }


def _path_conflicts(
    path: List[Tuple[Coord, int]], opponent_dist_maps: Dict[Coord, Dict[Coord, int]],
) -> Dict[Coord, Set[Coord]]:
    """For a candidate escape path (tile, offset) sequence, returns {tile:
    {conflicting opponents}} for every tile on the path where some
    opponent's real BFS distance (from its current position, no future-
    position prediction) to that tile is <= the path's own offset there --
    i.e. the opponent could physically be standing on that tile by the time
    this path would use it, using ordinary current-position information
    only (see has_safe_escape_after_bombing()'s docstring for the design
    this replaces).
    """
    conflicts: Dict[Coord, Set[Coord]] = {}
    for tile, offset in path:
        hit = {opp for opp, dmap in opponent_dist_maps.items() if dmap.get(tile, float("inf")) <= offset}
        if hit:
            conflicts[tile] = hit
    return conflicts


REQUIRED_ESCAPE_DIRECTIONS_CAP = 4  # a tile has at most 4 movement
# directions, so N+1 can never usefully exceed 4 regardless of how many
# opponents are involved -- there simply aren't more routes to require.


def _candidate_escape_paths(
    pos: Coord,
    field_arr: np.ndarray,
    blocked: FrozenSet[Coord],
    danger_offsets: Dict[Coord, Set[int]],
    start_offset: int,
) -> Dict[str, List[Tuple[Coord, int]]]:
    """Per-direction safe-escape path from `pos` (occupied at
    `start_offset`): one find_safe_path() BFS per movement direction whose
    neighbor tile is free. A direction is absent from the result iff no
    safe path exists that way.
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
    """Does `pos` (occupied at `start_offset`) have >=N+1 independent
    first-step escape directions, each verified via find_safe_path from
    start_offset+1 onward, where N is the number of distinct opponents that
    conflict with at least one candidate path (see `_path_conflicts()`) --
    not the number of conflicting tiles?

    Per-tile conflict, not a single distance-to-`pos` heuristic: each
    candidate direction's full found path is checked tile-by-tile against
    every opponent's real, current-position BFS distance (see
    `_opponent_distance_maps()`/`_path_conflicts()`) -- a tile at path-offset
    s is a conflict iff some opponent's distance to it is <=s, i.e. the
    opponent could physically reach it in time using only its ordinary,
    already-known position (no future-movement prediction).

    "Independent" now means: the required N+1 paths must be pairwise
    conflict-disjoint -- for any two selected paths, neither path's own
    conflict tiles may appear anywhere on the other path's route (a
    conflicting opponent seizing a shared tile must not be able to
    invalidate two of the N+1 chosen paths at once). Only conflict tiles are
    required to differ; non-conflict tiles may overlap freely between
    paths. This replaces the previous fixed "distinct first step" rule,
    which only guarded against a same-tick race for the very first escape
    tile and missed the same opponent walking into a later leg of a
    multi-step route.

    With N=0 (no opponent conflicts with any candidate path), this reduces
    to the pre-existing ">=1 safe path exists" standard.
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
    """True iff placing a bomb at `tile` right now would leave a robust safe
    escape, generalizing action_mask.py's own BOMB-legality check to an
    arbitrary candidate tile, so bombing_target_info() below can filter out
    candidates with no safe escape. Mirrors the mask's construction exactly:
    the bomb's own blast tiles become lethal at {BOMB_TIMER, BOMB_TIMER+1},
    and `tile` itself is blocked for the search horizon (can't walk back
    onto an unexploded bomb). `tile` itself is checked for immediate
    (offset 0) danger first, since placing the bomb keeps the agent
    standing on `tile` for this step.

    Delegates the escape-direction counting/conflict analysis to
    `_has_sufficient_escape_directions()` -- see its docstring for the
    N+1-independent-paths standard, shared with action_mask.py's per-step
    redundancy re-check (applied while walking away from an already-placed
    bomb, not just at the placement instant) so both apply the identical
    standard without duplicating it.

    Known limitation: this validates that some safe continuation exists
    from a position at the instant of the check, using each opponent's
    current position (no future-position prediction) -- an opponent that
    subsequently moves in a way this check didn't anticipate can still
    invalidate a previously-verified route. Also unaddressed: two agents
    racing for the same immediately-adjacent tile in the same tick, which
    the environment resolves via its own per-step random agent order and
    can turn an already-chosen, already-verified-safe move into
    INVALID_ACTION with no time left to recover.
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
    """Escape-difficulty score in [0, 1] for `opp`, standing inside a
    candidate kill-target bomb's blast, under that bomb's hypothetical
    danger. Mirrors has_safe_escape_after_bombing()'s per-direction
    machinery but rooted at `opp`'s own position instead of self_pos:
    `opp`'s own tile is excluded from `blocked` (it must be able to path
    out from where it actually stands) and self_pos joins `blocked` (the
    agent's own tile is a physical obstacle to the opponent).
    `other_opponents` (every opponent except `opp`) plays the role
    has_safe_escape_after_bombing()'s `opponents` argument plays for
    self_pos -- chasers whose current-position BFS distance can flag a
    conflict on one of `opp`'s candidate escape tiles (see
    `_path_conflicts()`).

    Score = 1 - (conflict-free escape direction count) / 4: the fraction
    of the 4 movement directions with no such chaser threat anywhere on
    their safe-escape path. A pure spatial-difficulty proxy from a single
    blast/danger snapshot, not a predicted death probability -- it does
    not model whether `opp` would actually choose that direction.
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
    """BFS over reachable tiles for the tile from which placing a bomb
    would catch >=1 opponent's CURRENT position within blast range -- a
    snapshot judgment (see bomb_threatens_reachable_opponent()), not a
    predicted real kill probability: it does not model whether a covered
    opponent would subsequently walk out of the blast before it
    detonates.

    Parallel to, but scored independently of, bombing_target_info(): a
    candidate tile qualifies here purely by covering an opponent's current
    position, regardless of whether it also hits any crate.

    Safety filtering reuses has_safe_escape_after_bombing() unchanged
    (the agent's own escape route). Each covered opponent additionally
    gets an escape-difficulty score in [0, 1] (see
    `_opponent_escape_difficulty()`); a tile's expected kill value is the
    MAX (not sum/average) of its covered opponents' difficulty scores --
    the easiest-of-the-covered-opponents-to-kill represents the tile,
    already bounded to [0, 1] with no further normalization needed.
    Selection mirrors bombing_target_info(): score = expected_kill_value
    / (distance + 1), compared via exact fractions; max score wins, ties
    broken by distance then first-step direction.
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
    """Per-opponent BFS distance, keyed by opponent coord. Since every
    opponent's own tile is itself a member of `blocked`, it never appears in
    `dist_map` (a BFS distance map already computed from self_pos over that
    same `blocked` set) -- mirroring bombing_target_info()'s treatment of a
    crate (never stood on, only required to be within range of a reachable
    tile). An opponent's distance is the minimum, among its own free
    neighboring tiles, of that tile's distance in `dist_map`, plus 1. An
    opponent with every neighboring tile walled, crated, or occupied is
    unreachable and excluded.
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
    """Count of tiles (including self_pos) reachable from self_pos within
    `depth_cap` BFS steps. Blocked = walls/crates (via is_free's field==0
    check) and living opponents' current tiles only -- bomb tiles do not
    block and no danger/timing is considered.
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

    # Opponents' current positions join bombs as currently known temporary
    # obstacles in the one shared `blocked` set consumed by movement mask,
    # escape-route BFS, coin BFS, and bombing-target BFS alike -- no
    # separate handling per call site.
    blocked = _bomb_positions(game_state) | frozenset(opponents)
    can_move = legal_moves(field_arr, self_pos, blocked=blocked)

    dist_from_self = bfs_distances(field_arr, self_pos, blocked=blocked)
    reachable_coins = [c for c in coins if c in dist_from_self]

    coin_path_dirs = {d: False for d in DIRECTIONS}
    nearest_distance: Optional[int] = None
    coin_contested = False

    if reachable_coins:
        nearest_distance = min(dist_from_self[c] for c in reachable_coins)
        # Multiple coins tied at the minimum distance -> no arbitrary
        # tie-break, mark all first-step directions that lie on some
        # shortest path to any nearest coin. A neighbor N is a valid first
        # step towards coin C iff dist(N, C) == nearest_distance - 1.
        tied_coins = [c for c in reachable_coins if dist_from_self[c] == nearest_distance]
        for coin in tied_coins:
            dist_from_coin = bfs_distances(field_arr, coin, blocked=blocked)
            for d in DIRECTIONS:
                if coin_path_dirs[d] or not can_move[d]:
                    continue
                nxt = neighbor_tile(self_pos, d)
                if dist_from_coin.get(nxt) == nearest_distance - 1:
                    coin_path_dirs[d] = True

        # coin_contested is purely a same-step distance comparison, no
        # intent/trajectory modeling. An opponent's distance to the locked
        # nearest-coin set uses its own BFS rooted at its own position, over
        # the same shared `blocked` graph used everywhere else in this
        # function.
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
