"""Task 4 offline investigation: c2 precursors and bomb-placement precision.
Read-only on task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Lethal own bombs come from analyze_stage_d_final._classify_selfkill();
escape-route metrics reuse state_processing's escape machinery unchanged;
raw #29/#31 values come from alive_opponent_distances()/reachable_space_count().

Definitions specific to this script:
- lethal opponent bomb (got_killed cases): the opponent bomb whose blast
  covers the death position, matched to its placement step
  (_all_opponent_bomb_placements()).
- #31-narrowing event: a drop of at least `threshold` in normalized
  feature[30] between consecutive steps (_find_events()).
- AUC via Mann-Whitney U / (n1*n2), with rank-based tie handling.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import rankdata

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import (
    DIRECTIONS, SAFETY_HORIZON, _candidate_escape_paths, _opponent_distance_maps,
    alive_opponent_distances, bfs_distances, blast_coords, compute_danger_offsets,
    extract_semantic_state, is_free, neighbor_tile, reachable_space_count,
)

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
RNG = np.random.default_rng(20260921)


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


def _auc(pos_vals, neg_vals):
    pos_vals = np.asarray([v for v in pos_vals if v is not None], dtype=np.float64)
    neg_vals = np.asarray([v for v in neg_vals if v is not None], dtype=np.float64)
    n1, n2 = len(pos_vals), len(neg_vals)
    if n1 == 0 or n2 == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos_vals, neg_vals]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n2))


def _dist_bucket_counts(dist_map, thresholds=(2, 3, 4, 5)):
    vals = list(dist_map.values())
    return {t: sum(1 for d in vals if d <= t) for t in thresholds}


def _self_escape_metrics_hypothetical(self_pos, field_arr, blocked, danger_offsets, power, opponents):
    """Same hypothetical-danger construction as has_safe_escape_after_bombing(),
    returning the candidate paths instead of a bool."""
    hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
    for blast_tile in blast_coords(field_arr, self_pos, power):
        hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    blocked_with_new_bomb = blocked | {self_pos}
    paths = _candidate_escape_paths(self_pos, field_arr, blocked_with_new_bomb, hypothetical_danger, start_offset=0)
    return paths


def _min_opp_dist_to_tiles(field_arr, blocked, opponents, tiles):
    if not tiles or not opponents:
        return None
    dist_maps = _opponent_distance_maps(field_arr, blocked, opponents)
    best = None
    for dmap in dist_maps.values():
        for t in tiles:
            d = dmap.get(t)
            if d is not None and (best is None or d < best):
                best = d
    return best


def _gather_metrics_for_bomb(gs, restore=True):
    """Per-step metrics for a step where rhine placed (or is about to place)
    a bomb."""
    state = restore_state(gs) if restore else gs
    sem = extract_semantic_state(state)
    field_arr, self_pos, blocked = sem.field_arr, sem.self_pos, sem.blocked
    opp_dists_raw = alive_opponent_distances(field_arr, blocked, bfs_distances(field_arr, self_pos, blocked=blocked), sem.opponents)
    f29_raw = min(opp_dists_raw.values()) if opp_dists_raw else None
    buckets = _dist_bucket_counts(opp_dists_raw)
    f31_raw = reachable_space_count(field_arr, self_pos, sem.opponents, cfg.REACHABLE_SPACE_DEPTH_CAP)
    paths = _self_escape_metrics_hypothetical(self_pos, field_arr, blocked, sem.danger_offsets, cfg.BOMB_POWER, sem.opponents)
    n_escape_routes = len(paths)
    escape_tiles = {t for path in paths.values() for t, _off in path}
    nearest_opp_to_escape = _min_opp_dist_to_tiles(field_arr, blocked, sem.opponents, escape_tiles)
    return {
        "f29_raw": f29_raw, "n_opp_le2": buckets[2], "n_opp_le3": buckets[3],
        "n_opp_le4": buckets[4], "n_opp_le5": buckets[5], "f31_raw": f31_raw,
        "n_escape_routes": n_escape_routes, "nearest_opp_to_escape": nearest_opp_to_escape,
    }


def item4a():
    out("=== Item 4a: self-kill -- lethal vs non-lethal own bomb placements ===")
    out("Lethal-bomb identification reuses analyze_stage_d_final._classify_selfkill()'s bomb_idx rule "
        "unchanged (most recent own BOMB action within 10 steps of death). Non-lethal = every other own "
        "BOMB action across all 300 breaker_only rounds (self-kill rounds' earlier bombs + all bombs in "
        "non-self-kill rounds).")
    lethal_metrics = []
    nonlethal_metrics = []
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            lethal_idx = None
            if r["metrics"]["self_kill"]:
                _, lethal_idx, _ = _classify_selfkill(r)
            for i, x in enumerate(recs):
                if x["action"] != "BOMB":
                    continue
                m = _gather_metrics_for_bomb(x["state"])
                if i == lethal_idx:
                    lethal_metrics.append(m)
                else:
                    nonlethal_metrics.append(m)
    out(f"\nsample sizes: lethal={len(lethal_metrics)} non-lethal={len(nonlethal_metrics)}")
    if len(lethal_metrics) < 30:
        out("NOTE: lethal n<30, small sample, for reference only.")
    for key, label in (
        ("f29_raw", "#29 raw nearest-opponent BFS distance"),
        ("n_opp_le2", "opponents within raw distance <=2"),
        ("n_opp_le3", "opponents within raw distance <=3"),
        ("n_opp_le4", "opponents within raw distance <=4"),
        ("n_opp_le5", "opponents within raw distance <=5"),
        ("f31_raw", "#31 raw reachable-space count"),
        ("n_escape_routes", "escape routes available after the bomb"),
        ("nearest_opp_to_escape", "nearest opponent's BFS distance to any escape-route tile"),
    ):
        lv = [m[key] for m in lethal_metrics if m[key] is not None]
        nv = [m[key] for m in nonlethal_metrics if m[key] is not None]
        auc = _auc(lv, nv)
        out(f"  {label}: lethal n={len(lv)} mean={np.mean(lv) if lv else float('nan'):.2f} "
            f"median={np.median(lv) if lv else float('nan'):.2f} | non-lethal n={len(nv)} "
            f"mean={np.mean(nv) if nv else float('nan'):.2f} median={np.median(nv) if nv else float('nan'):.2f} "
            f"| AUC(lethal>non-lethal)={auc:.3f}")


def _all_opponent_bomb_placements(rec):
    """Per round: [(step_index, owner, pos)] for each opponent bomb at the
    first step it appears in world.bombs."""
    recs = rec["steps"]
    seen = set()
    placements = []
    for i, x in enumerate(recs):
        for bpos, timer, owner in x["world"]["bombs"]:
            if owner == "rhine":
                continue
            key = (tuple(bpos), owner)
            if key not in seen:
                seen.add(key)
                placements.append((i, owner, tuple(bpos)))
    return placements


def item4b():
    out("\n=== Item 4b: got_killed -- lethal vs non-lethal opponent bomb placements ===")
    out("New: lethal opponent bomb identified by owner+position match against death['expl'] (killer "
        "explosion), located at its placement step via a full per-round world.bombs scan.")
    lethal_metrics = []
    nonlethal_metrics = []
    n_unmatched = 0
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            placements = _all_opponent_bomb_placements(r)
            lethal_key = None
            if r["metrics"]["got_killed_by_opponent"] and r["death"]:
                death = r["death"]
                pos = tuple(death["pos"])
                for owner, coords, timer in death["expl"]:
                    if owner == "rhine":
                        continue
                    if pos in {tuple(c) for c in coords}:
                        # Match this killer's most recent placement before death.
                        candidates = [(i, o, p) for i, o, p in placements if o == owner]
                        if candidates:
                            lethal_key = candidates[-1]
                        break
                if lethal_key is None:
                    n_unmatched += 1
            for i, owner, pos in placements:
                gs = restore_state(recs[i]["state"])
                sem = extract_semantic_state(gs)
                dist_to_me = bfs_distances(sem.field_arr, pos, blocked=sem.blocked).get(sem.self_pos)
                my_f31 = reachable_space_count(sem.field_arr, sem.self_pos, sem.opponents, cfg.REACHABLE_SPACE_DEPTH_CAP)
                my_paths = _candidate_escape_paths(sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, start_offset=0)
                m = {"dist_to_me": dist_to_me, "my_f31": my_f31, "my_n_escape_routes": len(my_paths)}
                if lethal_key is not None and (i, owner, pos) == lethal_key:
                    lethal_metrics.append(m)
                else:
                    nonlethal_metrics.append(m)
    out(f"\nsample sizes: lethal={len(lethal_metrics)} non-lethal={len(nonlethal_metrics)} "
        f"unmatched_got_killed_cases={n_unmatched} (killer bomb owner+position/step not uniquely "
        "resolvable from the recording -- excluded from the lethal set, counted here for transparency)")
    if len(lethal_metrics) < 30:
        out("NOTE: lethal n<30, small sample, for reference only.")
    for key, label in (
        ("dist_to_me", "opponent's bomb-placement position -> my BFS distance"),
        ("my_f31", "my #31 raw reachable-space at that step"),
        ("my_n_escape_routes", "my escape routes available at that step"),
    ):
        lv = [m[key] for m in lethal_metrics if m[key] is not None]
        nv = [m[key] for m in nonlethal_metrics if m[key] is not None]
        auc = _auc(lv, nv)
        out(f"  {label}: lethal n={len(lv)} mean={np.mean(lv) if lv else float('nan'):.2f} "
            f"median={np.median(lv) if lv else float('nan'):.2f} | non-lethal n={len(nv)} "
            f"mean={np.mean(nv) if nv else float('nan'):.2f} median={np.median(nv) if nv else float('nan'):.2f} "
            f"| AUC(lethal<non-lethal shown as 1-AUC below if inverted)={auc:.3f}")


# ---------------------------------------------------------------- 4c/4d/4e
def _round_f31_series(payload):
    out_list = []
    for r in payload["records"]:
        pts = [float(x["features"][30]) for x in r["steps"]]
        m = r["metrics"]
        death_step = r["death"]["step"] if r["death"] else None
        out_list.append({
            "round": r["round"], "self_kill": m["self_kill"], "got_killed": m["got_killed_by_opponent"],
            "pts": pts, "steps": [x["state"]["step"] for x in r["steps"]], "death_step": death_step,
        })
    return out_list


def _find_events(pts, threshold):
    """Start indices i where pts[i] - pts[i+1] >= threshold. A start directly
    following the previously kept event is merged into it."""
    raw = [i for i in range(len(pts) - 1) if pts[i] - pts[i + 1] >= threshold]
    events = []
    for i in raw:
        if events and i == events[-1] + 1:
            continue
        events.append(i)
    return events


def _precision_recall_for(all_series, threshold, H, kind):
    pred = (lambda rd: rd["self_kill"]) if kind == "self-kill" else (lambda rd: rd["got_killed"])
    n_events = 0
    n_events_leading_to_death = 0
    death_rounds = 0
    death_rounds_with_prior_event = 0
    for rd in all_series:
        events = _find_events(rd["pts"], threshold)
        n_events += len(events)
        is_death_round = pred(rd)
        if is_death_round and rd["death_step"] is not None:
            death_rounds += 1
            death_idx = None
            for j, st in enumerate(rd["steps"]):
                if st == rd["death_step"]:
                    death_idx = j
                    break
            if death_idx is None:
                death_idx = len(rd["steps"]) - 1
            has_prior_event = any(0 <= death_idx - e <= H for e in events)
            if has_prior_event:
                death_rounds_with_prior_event += 1
        for e in events:
            if is_death_round and rd["death_step"] is not None:
                death_idx = next((j for j, st in enumerate(rd["steps"]) if st == rd["death_step"]), len(rd["steps"]) - 1)
                if 0 <= death_idx - e <= H:
                    n_events_leading_to_death += 1
    precision = n_events_leading_to_death / n_events if n_events else float("nan")
    recall = death_rounds_with_prior_event / death_rounds if death_rounds else float("nan")
    return {
        "n_events": n_events, "n_events_leading_to_death": n_events_leading_to_death, "precision": precision,
        "death_rounds": death_rounds, "death_rounds_with_prior_event": death_rounds_with_prior_event, "recall": recall,
    }


def _random_anchor_baseline(all_series, H, n_samples=500):
    eligible = [(rd, j) for rd in all_series for j in range(len(rd["pts"]) - 1)]
    hits_sk = hits_gk = 0
    for _ in range(n_samples):
        rd, j = eligible[RNG.integers(0, len(eligible))]
        window_end = min(j + H, len(rd["steps"]) - 1)
        if rd["self_kill"] and rd["death_step"] is not None:
            death_idx = next((k for k, st in enumerate(rd["steps"]) if st == rd["death_step"]), len(rd["steps"]) - 1)
            if j <= death_idx <= window_end:
                hits_sk += 1
        if rd["got_killed"] and rd["death_step"] is not None:
            death_idx = next((k for k, st in enumerate(rd["steps"]) if st == rd["death_step"]), len(rd["steps"]) - 1)
            if j <= death_idx <= window_end:
                hits_gk += 1
    return hits_sk / n_samples, hits_gk / n_samples


def item4c_d():
    out("\n=== Item 4c/4d: #31-narrowing event precision/recall (+ sensitivity grid) ===")
    out("New event definition: a normalized features[:,30] drop of >=threshold within 2 consecutive "
        "steps; adjacent starts merged into one event. Primary: threshold=0.2, H=8.")
    all_series = []
    for s in SEEDS:
        all_series.extend(_round_f31_series(load(s)))
    n_rounds = len(all_series)
    out(f"sample: {n_rounds} rounds (3 seeds x 100)")

    out("\n--- primary (threshold=0.2, H=8) ---")
    for kind in ("self-kill", "got_killed"):
        pr = _precision_recall_for(all_series, 0.2, 8, kind)
        base_sk, base_gk = _random_anchor_baseline(all_series, 8)
        base = base_sk if kind == "self-kill" else base_gk
        out(f"  {kind}: n_events={pr['n_events']} precision={pr['precision']:.3f} "
            f"(events leading to this-kind-of-death within H: {pr['n_events_leading_to_death']}); "
            f"n_death_rounds={pr['death_rounds']} recall={pr['recall']:.3f} "
            f"(death rounds with a prior event: {pr['death_rounds_with_prior_event']}); "
            f"random-anchor baseline precision={base:.4f}")
        if pr["death_rounds"] < 30:
            out(f"    NOTE: n_death_rounds={pr['death_rounds']} <30, small sample, for reference only.")

    out("\n--- 4d sensitivity grid (H x threshold), not used to pick a best setting ---")
    for H in (4, 6, 8, 12):
        for threshold in (0.1, 0.2, 0.3):
            row = []
            for kind in ("self-kill", "got_killed"):
                pr = _precision_recall_for(all_series, threshold, H, kind)
                row.append(f"{kind}: n_events={pr['n_events']} P={pr['precision']:.3f} R={pr['recall']:.3f}")
            out(f"  H={H:>2} thr={threshold:.1f}: " + " | ".join(row))

    out("\n=== Item 4e: round-level precision (has >=1 event in round, threshold=0.2) ===")
    for kind in ("self-kill", "got_killed"):
        pred = (lambda rd: rd["self_kill"]) if kind == "self-kill" else (lambda rd: rd["got_killed"])
        has_event = [len(_find_events(rd["pts"], 0.2)) > 0 for rd in all_series]
        died = [pred(rd) for rd in all_series]
        n_event_rounds = sum(has_event)
        n_event_and_died = sum(1 for h, d in zip(has_event, died) if h and d)
        n_died = sum(died)
        n_died_no_event = sum(1 for h, d in zip(has_event, died) if d and not h)
        round_precision = n_event_and_died / n_event_rounds if n_event_rounds else float("nan")
        base_rate = n_died / n_rounds
        out(f"  {kind}: rounds_with_event={n_event_rounds}/{n_rounds} round_precision={round_precision:.3f} "
            f"(P(died|event)) vs base_rate={base_rate:.3f} (P(died) unconditional); "
            f"died_rounds_without_any_event={n_died_no_event}/{n_died}")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation2_c2_precursor.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item4a()
    item4b()
    item4c_d()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
