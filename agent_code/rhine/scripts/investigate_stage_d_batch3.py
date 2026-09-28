"""Stage D batch-3 read-only investigation, offline on recorded per-step data
(see investigate_stage_d_batch2.py's SOURCES).

Usage:
  python -m agent_code.rhine.scripts.investigate_stage_d_batch3 <1a|1b|2|3|all> [--source coin_collector|rulebased]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_stage_d_batch2 import SOURCES, load
from agent_code.rhine.state_processing import (
    _bomb_positions, _to_coord, bfs_distances, blast_coords, is_free, neighbor_tile, DIRECTIONS,
)

SEEDS = (0, 1, 2)
BOMB_IDX = cfg.ACTIONS.index("BOMB")


def out(msg=""):
    print(msg)


def _other_legal_exists(mask):
    return any(mask[i] for i in range(6) if i != BOMB_IDX)


# ================================================================ 1a/1b
def section_1a(source):
    out(f"=== (3.1a) actually-chosen BOMB steps split by current_tile_in_danger [{source}] ===")
    counts = {True: 0, False: 0}
    death_within = {True: {n: 0 for n in (1, 3, 5, 10)}, False: {n: 0 for n in (1, 3, 5, 10)}}
    n_rounds = 0
    for s in SEEDS:
        payload = load(source, s)
        n_rounds += len(payload["records"])
        for r in payload["records"]:
            recs = r["steps"]
            death_step = r["death"]["step"] if r["death"] else None
            died = r["metrics"]["self_kill"] or r["metrics"]["got_killed_by_opponent"]
            for x in recs:
                if x["action"] != "BOMB":
                    continue
                danger = bool(x["features"][18])
                counts[danger] += 1
                if died and death_step is not None and death_step >= x["state"]["step"]:
                    gap = death_step - x["state"]["step"]
                    for n in (1, 3, 5, 10):
                        if gap <= n:
                            death_within[danger][n] += 1
    for danger in (True, False):
        n = counts[danger]
        out(f"\ncurrent_tile_in_danger={danger}: {n} steps ({n/n_rounds:.3f}/round)")
        for k in (1, 3, 5, 10):
            out(f"  died within {k} step(s) [self-kill or got_killed]: {death_within[danger][k]}/{n} "
                f"({100*death_within[danger][k]/n if n else float('nan'):.2f}%)")


def section_1b(source):
    out(f"=== (3.1b) danger-source and nearest_threat_timer breakdown, current_tile_in_danger=True [{source}] ===")
    by_source = Counter()
    by_source_death10 = Counter()
    by_timer = Counter()
    by_timer_death10 = Counter()
    n_total = 0
    for s in SEEDS:
        payload = load(source, s)
        for r in payload["records"]:
            recs = r["steps"]
            death_step = r["death"]["step"] if r["death"] else None
            died = r["metrics"]["self_kill"] or r["metrics"]["got_killed_by_opponent"]
            for x in recs:
                if x["action"] != "BOMB" or not bool(x["features"][18]):
                    continue
                n_total += 1
                state = restore_state(x["state"])
                pos = tuple(int(v) for v in x["state"]["self"][3])
                died_within_10 = died and death_step is not None and 0 <= death_step - x["state"]["step"] <= 10

                if state["explosion_map"][pos] > 0:
                    src = "existing_explosion"
                else:
                    opp_bomb_covers = any(
                        owner != "rhine" and pos in set(blast_coords(state["field"], bpos, cfg.BOMB_POWER))
                        for bpos, timer, owner in x["world"]["bombs"]
                    )
                    src = "opponent_bomb" if opp_bomb_covers else "other_own_bomb"
                by_source[src] += 1
                by_source_death10[src] += died_within_10

                timer = int(round(float(x["features"][19]) * cfg.BOMB_TIMER))
                by_timer[timer] += 1
                by_timer_death10[timer] += died_within_10

    out(f"total steps in this population: {n_total}\n")
    out("-- by danger source --")
    for src, n in by_source.items():
        out(f"  {src}: {n} ({n/300:.3f}/round); died within 10 steps: {by_source_death10[src]}/{n} "
            f"({100*by_source_death10[src]/n:.1f}%)")
    out("\n-- by nearest_threat_timer --")
    for t in sorted(by_timer):
        n = by_timer[t]
        out(f"  timer={t}: {n} ({n/300:.3f}/round); died within 10 steps: {by_timer_death10[t]}/{n} "
            f"({100*by_timer_death10[t]/n:.1f}%)")


# ================================================================ 2: siege narrowing analysis
def _reachable_space(field_arr, self_pos, opponents, depth_cap=6):
    blocked = frozenset(opponents)
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    return sum(1 for d in dist_map.values() if d <= depth_cap)


def _opp_bfs_distance(field_arr, self_pos, opp_pos, opponents):
    blocked = frozenset(o for o in opponents if o != opp_pos)
    dmap = bfs_distances(field_arr, self_pos, blocked=blocked)
    best = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor = (opp_pos[0] + dx, opp_pos[1] + dy)
        if field_arr[neighbor] == -1:
            continue
        if neighbor in dmap:
            d = dmap[neighbor] + 1
            if best is None or d < best:
                best = d
    return best


def section_2(source):
    out(f"=== (3.2) siege narrowing analysis, last 10 steps [{source}] ===")
    out("'reachable space' = BFS-reachable tile count from self_pos, blocked=walls/crates/bombs/living "
        "opponents' current tile, danger ignored, depth capped at 6.\n")

    trace_path = SOURCES[source][0] / "narrowing_trace.txt"
    trace = open(trace_path, "w")
    out(f"full per-step (t-10..t-1) mask-legal-count/reachable-space/opponent-BFS-distances trace: {trace_path}\n")

    for kind, pred in (("self-kill", lambda r: r["metrics"]["self_kill"]),
                       ("got_killed", lambda r: r["metrics"]["got_killed_by_opponent"])):
        out(f"\n--- {kind} ---")
        cases = []
        for s in SEEDS:
            payload = load(source, s)
            for r in payload["records"]:
                if not pred(r):
                    continue
                cases.append((s, r))

        curve = [[] for _ in range(10)]  # Index 0 = t-10 ... index 9 = t-1.
        narrow_first_legal1 = []
        narrow_first_drop = []
        onset_bucket = Counter()
        global_nondeath_space = []
        for s, r in cases:
            recs = r["steps"]
            round_space = []
            for x in recs:
                state = restore_state(x["state"])
                self_pos = tuple(int(v) for v in x["state"]["self"][3])
                opponents = [tuple(int(v) for v in o[3]) for o in x["state"]["others"]]
                round_space.append(_reachable_space(state["field"], self_pos, opponents))
            baseline = float(np.mean(round_space[:-10])) if len(round_space) > 10 else float(np.mean(round_space))
            global_nondeath_space.append(baseline)

            tail = recs[-10:]
            legal_counts = [int(np.sum(x["final_mask"])) for x in tail]
            spaces = round_space[-10:]
            for i in range(10):
                if i < len(spaces):
                    curve[i].append(spaces[i])

            trace.write(f"\n{'='*15} seed{s} round={r['round']} {kind} {'='*15}\n")
            for i, x in enumerate(tail):
                state = restore_state(x["state"])
                self_pos = tuple(int(v) for v in x["state"]["self"][3])
                opponents = [tuple(int(v) for v in o[3]) for o in x["state"]["others"]]
                opp_dists = [_opp_bfs_distance(state["field"], self_pos, o, opponents) for o in opponents]
                trace.write(
                    f"  t-{10-i} step={x['state']['step']:>4} pos={self_pos} action={x['action']:<5} "
                    f"n_legal={legal_counts[i]} reachable_space={spaces[i] if i < len(spaces) else None} "
                    f"n_opp_alive={len(opponents)} opp_dists={opp_dists}\n"
                )

            first_le1 = next((i for i, n in enumerate(legal_counts) if n <= 1), None)
            narrow_first_legal1.append(first_le1)
            first_drop = next((i for i, sp in enumerate(spaces) if sp < 0.5 * baseline), None)
            narrow_first_drop.append(first_drop)
            # Tail index i is 10 - i steps before death.
            if first_drop is None:
                onset_bucket["never/<50%_not_reached"] += 1
            elif first_drop <= 6:
                onset_bucket[">=4_steps_before"] += 1
            elif first_drop <= 8:
                onset_bucket["2-3_steps_before"] += 1
            else:
                onset_bucket["last_step_only"] += 1

            lead_actions = [x["action"] for x in tail[max(0, (first_drop or 9) - 3):(first_drop or 9)]]
            out(f"  seed{s} round={r['round']}: first_legal<=1 at "
                f"t-{10-first_le1 if first_le1 is not None else '?'}, first reachable-space<50%-baseline at "
                f"t-{10-first_drop if first_drop is not None else '?'}, "
                f"baseline={baseline:.1f}, spaces={spaces}, actions_before_narrowing={lead_actions}")

        out(f"\n  onset classification ({kind}, n={len(cases)}): {dict(onset_bucket)}")
        curve_means = [float(np.mean(c)) if c else float("nan") for c in curve]
        out(f"  reachable-space curve t-10..t-1 mean: {[round(v, 2) for v in curve_means]}")
        out(f"  global non-death-step baseline mean (per-round pre-tail average): {np.mean(global_nondeath_space):.2f}")

    trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["1a", "1b", "2", "all"])
    parser.add_argument("--source", choices=["coin_collector", "rulebased"], default="rulebased")
    args = parser.parse_args()
    if args.section in ("1a", "all"):
        section_1a(args.source)
    if args.section in ("1b", "all"):
        section_1b(args.source)
    if args.section in ("2", "all"):
        section_2(args.source)


if __name__ == "__main__":
    main()
