"""Task 4 offline investigation: escape-mechanism analysis (tie-break and
redundancy re-check) with step-level replay of two self-kill cases.
Read-only on task4_seed{0,1}_breaker_only_A.pkl.gz.

Since the recorded masks only hold final results, the intermediate values of
mask_from_semantic()'s two opponent-aware branches are re-derived through the
same state_processing helpers, printing which branch changed which
direction. Each escape path is annotated per tile with the opponent's
earliest BFS arrival vs the path's offset (_path_conflicts()'s per-tile test).
"""
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import (
    DIRECTIONS, REQUIRED_ESCAPE_DIRECTIONS_CAP, _candidate_escape_paths, _has_sufficient_escape_directions,
    _opponent_distance_maps, _path_conflicts, extract_semantic_state, neighbor_tile,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
FIFTEEN_ONE_CASES = {  # seed0: round -> bomb_idx
    21: 255, 26: 389, 27: 290, 32: 296, 36: 139, 60: 299, 74: 300, 75: 348, 87: 343,
}


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed):
    with gzip.open(DATA_DIR / f"task4_seed{seed}_breaker_only_A.pkl.gz", "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError("partial recording")
    return payload


def item1a():
    out("=== Item 1a: escape-mechanism code reading ===")
    out("""
Two opponent-aware branches live in action_mask.mask_from_semantic() (movement legality only -- BOMB
legality is a separate, always-on check, see note below):

1. Redundancy re-check (per-direction N+1-path requirement)
   Function: state_processing._has_sufficient_escape_directions() (called from mask_from_semantic()
   inline, not a separately named "rule" function).
   Trigger: `redundancy_trigger = semantic.current_tile_in_danger and bool(semantic.opponents)`
   i.e. only engages when the agent's OWN current tile is already inside some bomb's future blast
   (an existing bomb, not a hypothetical one) AND at least one opponent is alive. It is applied to
   every direction that is already legal under the plain exists_safe_path() check.
   Judgment: for each candidate direction's full path (from _candidate_escape_paths(), one
   find_safe_path() BFS per direction), every (tile, offset) pair on the ENTIRE path is checked via
   _path_conflicts() against every opponent's real current-position BFS distance map
   (_opponent_distance_maps()) -- a tile at path-offset s is a conflict iff some opponent's distance
   to it is <=s. Required independent path count = min(REQUIRED_ESCAPE_DIRECTIONS_CAP=4,
   1 + number of distinct opponents that conflict with >=1 candidate path) -- this is the codebase's
   actual constant (4, since a tile has at most 4 movement directions), not "3". A direction is
   masked out only if it fails to be part of some conflict-disjoint combination of that many
   directions, AND only if at least one other direction remains robust (never removes every option).
   Scope: whole path, tile-by-tile -- not just the landing tile.

2. Same-tick collision tie-break
   No separate function name; inline in mask_from_semantic() immediately after branch 1.
   Trigger: same gating condition (`semantic.current_tile_in_danger and semantic.opponents`), and
   additionally requires >=2 directions still legal after branch 1.
   Judgment: for each still-legal direction, look ONLY at its single landing tile (the immediate
   neighbor, offset 0) -- reuses the same _opponent_distance_maps() -- and calls it "uncontested" if
   no opponent's distance to that one tile is <=1. If >=1 uncontested direction exists, every
   contested direction is masked out; otherwise no change.
   Scope: landing tile only, distance threshold <=1 -- a coarser, single-tile heuristic, unlike
   branch 1's whole-path check.

BOMB-legality note: has_safe_escape_after_bombing() (called unconditionally whenever bombs_left>0,
not gated by redundancy_trigger) always routes through the SAME _has_sufficient_escape_directions()
N+1-path standard for the candidate bomb -- i.e. the bomb-placement decision always gets the full
whole-path opponent-conflict check, regardless of whether the agent's current tile is "already in
danger" (it can't be, since the bomb about to be placed is the only source of danger being
evaluated).
""")


def _report_case(label, payload_records, round_no, bomb_idx):
    r = next(rr for rr in payload_records if rr["round"] == round_no)
    recs = r["steps"]
    out(f"\n{'=' * 10} {label} round={round_no} bomb_idx={bomb_idx} {'=' * 10}")

    for tag, idx in (("bomb-placement step", bomb_idx), ("first step after placement", bomb_idx + 1)):
        if idx >= len(recs):
            out(f"  [{tag}] index {idx} out of range (round ended)")
            continue
        x = recs[idx]
        gs = restore_state(x["state"])
        sem = extract_semantic_state(gs)
        out(f"\n  --- {tag}: step={x['state']['step']} pos={sem.self_pos} action={x['action']} ---")
        opp_dist_maps = _opponent_distance_maps(sem.field_arr, sem.blocked, sem.opponents)
        for opp in sem.opponents:
            out(f"    opponent at {opp}: BFS distance to self = {opp_dist_maps[opp].get(sem.self_pos, 'unreachable')}")
        out(f"    current_tile_in_danger={sem.current_tile_in_danger}")

        redundancy_trigger = sem.current_tile_in_danger and bool(sem.opponents)
        out(f"    redundancy_trigger={redundancy_trigger}")

        candidate_paths = {}
        for d in DIRECTIONS:
            if not sem.can_move[d]:
                continue
            neighbor = neighbor_tile(sem.self_pos, d)
            from agent_code.rhine.state_processing import SAFETY_HORIZON, exists_safe_path, find_safe_path, is_free
            if not is_free(sem.field_arr, *neighbor, blocked=sem.blocked):
                continue
            if not exists_safe_path(neighbor, 0, sem.field_arr, sem.blocked, sem.danger_offsets, SAFETY_HORIZON):
                continue
            path = find_safe_path(neighbor, 1, sem.field_arr, sem.blocked, sem.danger_offsets, SAFETY_HORIZON)
            if path is not None:
                candidate_paths[d] = path
        out(f"    plain-legal directions (exists_safe_path only): {list(candidate_paths.keys())}")

        if redundancy_trigger and candidate_paths:
            path_conflicts = {d: _path_conflicts(p, opp_dist_maps) for d, p in candidate_paths.items()}
            all_conflicting = set()
            for c in path_conflicts.values():
                for s in c.values():
                    all_conflicting.update(s)
            required = min(REQUIRED_ESCAPE_DIRECTIONS_CAP, 1 + len(all_conflicting))
            out(f"    required independent paths = min(4, 1+{len(all_conflicting)}) = {required}")
            robust_by_direction = {
                d: _has_sufficient_escape_directions(
                    neighbor_tile(sem.self_pos, d), sem.field_arr, sem.blocked, sem.danger_offsets,
                    sem.opponents, start_offset=0,
                )
                for d in candidate_paths
            }
            out(f"    per-direction robust (branch 1 result): {robust_by_direction}")
            for d, path in candidate_paths.items():
                if path_conflicts[d]:
                    out(f"      dir={d} path={path} CONFLICT tiles={path_conflicts[d]}")
                else:
                    out(f"      dir={d} path={path} no conflicts")
            surviving = [d for d, robust in robust_by_direction.items() if robust or not any(robust_by_direction.values())]
        else:
            out("    branch 1 (redundancy re-check) does not engage here.")
            surviving = list(candidate_paths.keys())

        if redundancy_trigger and len(surviving) > 1:
            uncontested = [
                d for d in surviving
                if not any(dmap.get(neighbor_tile(sem.self_pos, d), float("inf")) <= 1 for dmap in opp_dist_maps.values())
            ]
            out(f"    tie-break (branch 2): landing tiles = {{{', '.join(f'{d}:{neighbor_tile(sem.self_pos, d)}' for d in surviving)}}}, "
                f"uncontested={uncontested}")
            if uncontested:
                out(f"    tie-break REMOVES: {[d for d in surviving if d not in uncontested]}")
        else:
            out("    branch 2 (tie-break) does not engage here (need >=2 surviving directions and the same trigger).")

        out(f"    recorded final_mask legal: {[cfg.ACTIONS[i] for i, m in enumerate(x['final_mask']) if m]}")
        out(f"    recorded probs_final: {{{', '.join(f'{cfg.ACTIONS[i]}:{float(p):.3f}' for i, p in enumerate(x['probs_final']))}}}")

        out("\n    per-path tile-level conflict detail (item 1c):")
        for d, path in candidate_paths.items():
            out(f"      dir={d}:")
            for tile, own_offset in path:
                earliest_opp = min((dmap.get(tile, float("inf")) for dmap in opp_dist_maps.values()), default=float("inf"))
                flag = " <-- CONFLICT (opponent arrival <= own offset)" if earliest_opp <= own_offset else ""
                out(f"        tile={tile} own_offset={own_offset} earliest_opponent_arrival={earliest_opp}{flag}")


def item1b_1c():
    out("\n=== Items 1b/1c: two example cases, step-level replay ===")
    payload1 = load(1)
    r1 = next(r for r in payload1["records"] if r["round"] == 1)
    bomb_idx_r1 = next(i for i, x in enumerate(r1["steps"]) if x["action"] == "BOMB" and x["state"]["step"] == 65)
    _report_case("seed1 c-class siege", payload1["records"], 1, bomb_idx_r1)

    payload0 = load(0)
    for rnd, bomb_idx in FIFTEEN_ONE_CASES.items():
        _report_case("seed0 (15,1)", payload0["records"], rnd, bomb_idx)


def main():
    global _out
    out_path = DATA_DIR / "investigation4_escape_mechanism.txt"
    _out = open(out_path, "w")
    item1a()
    item1b_1c()
    _out.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
