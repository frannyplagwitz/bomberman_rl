"""Stage D batch-4 read-only investigation, offline on recorded per-step data.

Part 1: siege reachable space against a cross-round, step-matched baseline
(avoids the crate-thinning confound of a same-round early-step baseline).
Reachable space treats walls, crates and living opponents as obstacles; bombs
are not.

Part 2: spawn-point BOMB-mask verification over all of Stage D's A+B groups,
plus a detailed re-derivation of the two step-0 oscillation cases.

Usage:
  python -m agent_code.rhine.scripts.investigate_stage_d_batch4 <1|2|all> [--source coin_collector|rulebased]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import state_processing as sp
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_stage_d_batch2 import SOURCES, load
from agent_code.rhine.state_processing import bfs_distances, blast_coords, extract_semantic_state

SEEDS = (0, 1, 2)
BOMB_IDX = cfg.ACTIONS.index("BOMB")


def out(msg=""):
    print(msg)


def _reachable_space(field_arr, self_pos, opponents, depth_cap=6):
    """Tiles reachable within `depth_cap` steps; walls, crates and living
    opponents block, bombs do not.
    """
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


def _round_space_series(payload):
    """Per-round list of (step, reachable_space) for every recorded step,
    plus whether the round is a 'death round' (self_kill or got_killed)."""
    series = []
    for r in payload["records"]:
        died = r["metrics"]["self_kill"] or r["metrics"]["got_killed_by_opponent"]
        pts = []
        for x in r["steps"]:
            state = restore_state(x["state"])
            self_pos = tuple(int(v) for v in x["state"]["self"][3])
            opponents = [tuple(int(v) for v in o[3]) for o in x["state"]["others"]]
            pts.append((x["state"]["step"], _reachable_space(state["field"], self_pos, opponents)))
        series.append({"round": r["round"], "died": died, "points": pts, "record": r})
    return series


def _nearest_opponent_bomb_offset(recs, death_step, window=15):
    """Most recent opponent-owned bomb that newly appeared on the board
    within `window` steps before death; returns steps-before-death at
    placement, or None if no opponent bomb appeared in that window."""
    seen = set()
    last_new_step = None
    for x in recs:
        step = x["state"]["step"]
        if step < death_step - window:
            for bpos, timer, owner in x["world"]["bombs"]:
                if owner != "rhine":
                    seen.add((tuple(bpos), owner))
            continue
        for bpos, timer, owner in x["world"]["bombs"]:
            if owner == "rhine":
                continue
            key = (tuple(bpos), owner)
            if key not in seen:
                seen.add(key)
                last_new_step = step
    if last_new_step is None:
        return None
    return death_step - last_new_step


def section_1(source):
    out(f"=== (4.1) siege reachable-space, cross-round step-matched baseline [{source}] ===")
    out("Correction: 'reachable space' blocking was ALWAYS walls/crates(via field==0)/living-opponents-tile "
        "only -- batch3's printed description additionally claiming bombs were blocking was a documentation "
        "error, not a code difference; the actual computation is unchanged. What IS new here is the baseline: "
        "instead of this round's own early steps, it's the mean reachable space, across all OTHER non-death "
        "rounds (any seed), at steps within +/-20 of this case's death step.\n")

    all_series = {s: _round_space_series(load(source, s)) for s in SEEDS}
    pool = []  # (step, space) from every non-death round, any seed.
    for s in SEEDS:
        for rd in all_series[s]:
            if not rd["died"]:
                pool.extend(rd["points"])
    pool_steps = np.array([p[0] for p in pool])
    pool_spaces = np.array([p[1] for p in pool])

    def baseline_at(step, window=20):
        mask = np.abs(pool_steps - step) <= window
        return float(pool_spaces[mask].mean()) if mask.any() else float("nan")

    for kind, pred in (("self-kill", lambda r: r["metrics"]["self_kill"]),
                       ("got_killed", lambda r: r["metrics"]["got_killed_by_opponent"])):
        out(f"\n--- {kind} ---")
        onset_bucket = Counter()
        curve_self, curve_base = [[] for _ in range(10)], [[] for _ in range(10)]
        early_cases = []
        n_cases = 0
        for s in SEEDS:
            for rd in all_series[s]:
                r = rd["record"]
                if not pred(r):
                    continue
                n_cases += 1
                recs = r["steps"]
                death_step = r["death"]["step"] if r["death"] else recs[-1]["state"]["step"]
                pts = rd["points"][-10:]
                tail = recs[-10:]
                self_vals = [p[1] for p in pts]
                base_vals = [baseline_at(p[0]) for p in pts]
                for i in range(len(self_vals)):
                    curve_self[i].append(self_vals[i])
                    curve_base[i].append(base_vals[i])

                drop_idx = next(
                    (i for i in range(len(self_vals)) if self_vals[i] < 0.7 * base_vals[i]), None,
                )
                bomb_offset = _nearest_opponent_bomb_offset(recs, death_step)
                if drop_idx is None:
                    onset_bucket["none"] += 1
                elif drop_idx <= 4:
                    onset_bucket[">=6_steps_before"] += 1
                elif drop_idx <= 6:
                    onset_bucket["4-5_steps_before"] += 1
                elif drop_idx <= 8:
                    onset_bucket["2-3_steps_before"] += 1
                else:
                    onset_bucket["last_step_only"] += 1

                steps_before = 10 - drop_idx if drop_idx is not None else None
                out(f"  seed{s} round={r['round']}: onset_steps_before_death={steps_before} "
                    f"nearest_opp_bomb_placed={bomb_offset}_steps_before "
                    f"self={[round(v, 1) for v in self_vals]} base={[round(v, 1) for v in base_vals]}")

                if drop_idx is not None and drop_idx <= 5:
                    lead = tail[max(0, drop_idx - 3):drop_idx]
                    onset_state = restore_state(tail[drop_idx]["state"])
                    onset_pos = tuple(int(v) for v in tail[drop_idx]["state"]["self"][3])
                    onset_opps = [tuple(int(v) for v in o[3]) for o in tail[drop_idx]["state"]["others"]]
                    dists = [_opp_bfs_distance(onset_state["field"], onset_pos, o, onset_opps) for o in onset_opps]
                    early_cases.append({
                        "seed": s, "round": r["round"], "lead_actions": [x["action"] for x in lead],
                        "n_opp": len(onset_opps), "opp_dists": dists,
                    })

        out(f"\n  onset classification ({kind}, n={n_cases}): {dict(onset_bucket)}")
        cs = [float(np.nanmean(c)) if c else float("nan") for c in curve_self]
        cb = [float(np.nanmean(c)) if c else float("nan") for c in curve_base]
        out(f"  self curve  t-10..t-1: {[round(v, 2) for v in cs]}")
        out(f"  base curve  t-10..t-1: {[round(v, 2) for v in cb]}")
        if early_cases:
            out(f"\n  cases with onset in t-10..t-5 (n={len(early_cases)}):")
            for c in early_cases:
                out(f"    seed{c['seed']} round={c['round']}: actions_before_onset={c['lead_actions']} "
                    f"n_opp_alive_at_onset={c['n_opp']} opp_dists_at_onset={c['opp_dists']}")


# ================================================================ 2: spawn verification
def _bomb_reject_reason(semantic):
    hyp = {t: set(o) for t, o in semantic.danger_offsets.items()}
    for bt in blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER):
        hyp.setdefault(bt, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    if 0 in hyp.get(semantic.self_pos, ()):
        return "self_pos_lethal_at_offset_0"
    candidates = sp._candidate_escape_paths(
        semantic.self_pos, semantic.field_arr, semantic.blocked | {semantic.self_pos}, hyp, 0,
    )
    if not candidates:
        return "no_candidate_escape_direction"
    return "insufficient_independent_paths"  # Candidates exist but the N+1 requirement fails.


def section_2a():
    out("=== (4.2a) step-0 BOMB-mask legality, Stage D coin_collector A+B groups (600 rounds) ===")
    n_legal = 0
    n_total = 0
    reasons = Counter()
    import gzip, pickle
    for group in ("A", "B"):
        for s in SEEDS:
            path = SOURCES["coin_collector"][0] / f"task3_stage_d_seed{s}_{group}.pkl.gz"
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
            for r in payload["records"]:
                n_total += 1
                x = r["steps"][0]
                if bool(x["base_mask"][BOMB_IDX]):
                    n_legal += 1
                else:
                    state = restore_state(x["state"])
                    semantic = extract_semantic_state(state)
                    reasons[_bomb_reject_reason(semantic)] += 1
    out(f"total rounds checked: {n_total}")
    out(f"BOMB legal at step 0: {n_legal}/{n_total} ({100*n_legal/n_total:.2f}%)")
    out(f"BOMB illegal at step 0: {n_total - n_legal}/{n_total} ({100*(n_total-n_legal)/n_total:.2f}%)")
    out(f"rejection reason breakdown: {dict(reasons)}")


def _describe_tile(field, pos, power):
    blast = set(blast_coords(field, pos, power))
    out(f"    5x5 window around {pos} (rows=x, cols=y; -1=wall 0=free 1=crate; '*'=in blast if bombed here):")
    x0, y0 = pos
    xmax, ymax = field.shape
    for x in range(max(0, x0 - 2), min(xmax, x0 + 3)):
        row = []
        for y in range(max(0, y0 - 2), min(ymax, y0 + 3)):
            v = field[x, y]
            mark = "*" if (x, y) in blast else " "
            row.append(f"{v:>2}{mark}")
        out(f"      x={x}: " + " ".join(row))


def section_2b_2c():
    out("\n=== (4.2b/c) round6/round46 detailed re-derivation ===")
    import gzip, pickle
    path = SOURCES["coin_collector"][0] / "task3_stage_d_seed1_B.pkl.gz"
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    for r in payload["records"]:
        if r["round"] not in (6, 46):
            continue
        recs = r["steps"]
        actions = r["metrics"]["actions"]
        from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
        run_len, start = longest_oscillation_run(actions)
        out(f"\n--- round={r['round']} total_run_len={run_len} starts_at_step0={start == 0} "
            f"(OSCILLATION_THRESHOLD={OSCILLATION_THRESHOLD}) ---")

        x0 = recs[0]
        state0 = restore_state(x0["state"])
        pos0 = tuple(int(v) for v in x0["state"]["self"][3])
        _describe_tile(state0["field"], pos0, cfg.BOMB_POWER)
        sem0 = extract_semantic_state(state0)
        out(f"    step0 pos={pos0} BOMB_legal={bool(x0['base_mask'][BOMB_IDX])} "
            f"reject_reason={_bomb_reject_reason(sem0) if not bool(x0['base_mask'][BOMB_IDX]) else 'n/a'}")

        window = recs[start:start + run_len + 1]
        positions = [tuple(int(v) for v in w["state"]["self"][3]) for w in window]
        distinct = sorted(set(positions))
        out(f"    the two tiles in the loop: {distinct}")
        for tile in distinct:
            rec = next(w for w in window if tuple(int(v) for v in w["state"]["self"][3]) == tile)
            state = restore_state(rec["state"])
            sem = extract_semantic_state(state)
            legal = [cfg.ACTIONS[i] for i, m in enumerate(rec["base_mask"]) if m]
            reason = _bomb_reject_reason(sem) if not bool(rec["base_mask"][BOMB_IDX]) else "n/a (legal)"
            out(f"    tile={tile}: legal_actions={legal} BOMB_legal={bool(rec['base_mask'][BOMB_IDX])} reject_reason={reason}")
            if reason == "no_candidate_escape_direction":
                _describe_tile(state["field"], tile, cfg.BOMB_POWER)

        end = start + run_len
        resolved_by = "runs_to_round_end" if end >= len(actions) - 1 else "resolved_mid_round"
        post = actions[end:end + 5] if end < len(actions) - 1 else []
        out(f"    loop length={run_len} steps, resolved_by={resolved_by}, post_break_actions={post}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["1", "2", "all"])
    parser.add_argument("--source", choices=["coin_collector", "rulebased"], default="rulebased")
    args = parser.parse_args()
    if args.section in ("1", "all"):
        section_1(args.source)
    if args.section in ("2", "all"):
        section_2a()
        section_2b_2c()


if __name__ == "__main__":
    main()
