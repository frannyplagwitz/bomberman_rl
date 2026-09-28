"""Task 4 offline investigation: c2 precursor recompute excluding the
repeated seed0 self-kill cases at one fixed tile, plus lethality rate by
proximity bucket (3b) and a threat-only denominator for got_killed (3c).
Read-only on task4_seed{0,1,2}_breaker_only_A.pkl.gz; reuses
investigate_task4_c2_precursor.py's helpers.

A "threatening opponent bomb" (3c) is one whose blast (blast_coords()) covers
rhine's position at placement time.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_task4_c2_precursor import (
    _all_opponent_bomb_placements, _auc, _gather_metrics_for_bomb,
)
from agent_code.rhine.state_processing import _candidate_escape_paths, blast_coords, extract_semantic_state

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
EXCLUDED_15_1 = {(0, r) for r in (21, 26, 27, 32, 36, 60, 74, 75, 87)}


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


def item3a():
    out("=== Item 3a: 4a recomputed excluding the 9 (15,1) seed0 self-kill cases ===")
    lethal_all, nonlethal_all = [], []
    lethal_excl, nonlethal_excl = [], []
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            lethal_idx = None
            if r["metrics"]["self_kill"]:
                _, lethal_idx, _ = _classify_selfkill(r)
            is_15_1_case = (s, r["round"]) in EXCLUDED_15_1
            for i, x in enumerate(recs):
                if x["action"] != "BOMB":
                    continue
                m = _gather_metrics_for_bomb(x["state"])
                if i == lethal_idx:
                    lethal_all.append(m)
                    if not is_15_1_case:
                        lethal_excl.append(m)
                else:
                    nonlethal_all.append(m)
                    if not is_15_1_case:
                        nonlethal_excl.append(m)

    out(f"\nsample sizes: lethal all={len(lethal_all)} lethal excl-(15,1)={len(lethal_excl)} "
        f"(removed {len(lethal_all) - len(lethal_excl)}); non-lethal all={len(nonlethal_all)} "
        f"excl-(15,1)={len(nonlethal_excl)}")
    if len(lethal_excl) < 30:
        out("NOTE: lethal-excluded n<30, small sample, for reference only.")
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
        lv_all = [m[key] for m in lethal_all if m[key] is not None]
        nv_all = [m[key] for m in nonlethal_all if m[key] is not None]
        lv_ex = [m[key] for m in lethal_excl if m[key] is not None]
        nv_ex = [m[key] for m in nonlethal_excl if m[key] is not None]
        out(f"  {label}:")
        out(f"    ALL:      lethal n={len(lv_all)} mean={np.mean(lv_all) if lv_all else float('nan'):.2f} "
            f"| non-lethal mean={np.mean(nv_all) if nv_all else float('nan'):.2f} | AUC={_auc(lv_all, nv_all):.3f}")
        out(f"    EXCLUDED: lethal n={len(lv_ex)} mean={np.mean(lv_ex) if lv_ex else float('nan'):.2f} "
            f"| non-lethal mean={np.mean(nv_ex) if nv_ex else float('nan'):.2f} | AUC={_auc(lv_ex, nv_ex):.3f}")


def item3b():
    out("\n=== Item 3b: lethality rate by opponent-proximity bucket (own bombs, all 8072 placements) ===")
    all_bombs = []
    lethal_flags_all = []
    lethal_flags_excl = []
    all_bombs_excl = []
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            lethal_idx = None
            if r["metrics"]["self_kill"]:
                _, lethal_idx, _ = _classify_selfkill(r)
            is_15_1_case = (s, r["round"]) in EXCLUDED_15_1
            for i, x in enumerate(recs):
                if x["action"] != "BOMB":
                    continue
                m = _gather_metrics_for_bomb(x["state"])
                is_lethal = i == lethal_idx
                all_bombs.append(m)
                lethal_flags_all.append(is_lethal)
                if not is_15_1_case:
                    all_bombs_excl.append(m)
                    lethal_flags_excl.append(is_lethal)

    n_total = len(all_bombs)
    n_lethal_total = sum(lethal_flags_all)
    out(f"\ntotal own bomb placements: {n_total}, lethal={n_lethal_total}, "
        f"overall lethality rate={100 * n_lethal_total / n_total:.3f}%")
    for label, bombs, flags in (("INCLUDING (15,1)", all_bombs, lethal_flags_all),
                                  ("EXCLUDING (15,1)", all_bombs_excl, lethal_flags_excl)):
        out(f"\n  --- {label} (n={len(bombs)}, lethal={sum(flags)}) ---")
        for thr in (2, 3, 4, 5):
            key = f"n_opp_le{thr}"
            has_opp = [(m[key] > 0) for m in bombs]
            n_has = sum(has_opp)
            n_lethal_has = sum(1 for h, f in zip(has_opp, flags) if h and f)
            rate_has = 100 * n_lethal_has / n_has if n_has else float("nan")
            n_not = len(bombs) - n_has
            n_lethal_not = sum(flags) - n_lethal_has
            rate_not = 100 * n_lethal_not / n_not if n_not else float("nan")
            out(f"    distance<=<{thr}: has_opponent_within n={n_has} lethal={n_lethal_has} rate={rate_has:.3f}% "
                f"| no_opponent_within n={n_not} lethal={n_lethal_not} rate={rate_not:.3f}% "
                f"| overall={100 * sum(flags) / len(bombs):.3f}%")
            if n_has < 30:
                out(f"      NOTE: has_opponent_within n={n_has} <30, small sample, for reference only.")


def item3c():
    out("\n=== Item 3c: got_killed, threatening-opponent-bomb-only denominator ===")
    out("New definition: a 'threatening' opponent bomb is one whose blast_coords(bomb_pos, BOMB_POWER) "
        "(wall/crate occlusion already handled by that function) covers rhine's position at the exact "
        "step the bomb is first observed placed.")
    threat_lethal = []
    threat_nonlethal = []
    n_total_placements = 0
    n_threatening = 0
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            placements = _all_opponent_bomb_placements(r)
            n_total_placements += len(placements)
            lethal_key = None
            if r["metrics"]["got_killed_by_opponent"] and r["death"]:
                death = r["death"]
                pos = tuple(death["pos"])
                for owner, coords, timer in death["expl"]:
                    if owner == "rhine":
                        continue
                    if pos in {tuple(c) for c in coords}:
                        candidates = [(i, o, p) for i, o, p in placements if o == owner]
                        if candidates:
                            lethal_key = candidates[-1]
                        break
            for i, owner, pos in placements:
                gs = restore_state(recs[i]["state"])
                sem = extract_semantic_state(gs)
                blast = set(blast_coords(sem.field_arr, pos, cfg.BOMB_POWER))
                if sem.self_pos not in blast:
                    continue
                n_threatening += 1
                my_paths = _candidate_escape_paths(sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, start_offset=0)
                has_safe_escape_now = len(my_paths) > 0
                m = {
                    "my_f31": None, "my_n_escape_routes": len(my_paths),
                    "dist_to_bomb": None, "has_safe_escape_now": has_safe_escape_now,
                }
                from agent_code.rhine.state_processing import bfs_distances, reachable_space_count
                m["my_f31"] = reachable_space_count(sem.field_arr, sem.self_pos, sem.opponents, cfg.REACHABLE_SPACE_DEPTH_CAP)
                m["dist_to_bomb"] = bfs_distances(sem.field_arr, pos, blocked=sem.blocked).get(sem.self_pos)
                if lethal_key is not None and (i, owner, pos) == lethal_key:
                    threat_lethal.append(m)
                else:
                    threat_nonlethal.append(m)

    out(f"\ntotal opponent bomb placements (all rounds): {n_total_placements}")
    out(f"threatening placements (blast covers my position at placement): {n_threatening}")
    out(f"  of which lethal (killed me): {len(threat_lethal)}; non-lethal: {len(threat_nonlethal)}")
    if len(threat_lethal) < 30:
        out("  NOTE: lethal n<30, small sample, for reference only.")
    for key, label in (("my_f31", "my #31 raw reachable-space"),
                        ("my_n_escape_routes", "my escape routes available"),
                        ("dist_to_bomb", "my BFS distance to the bomb position")):
        lv = [m[key] for m in threat_lethal if m[key] is not None]
        nv = [m[key] for m in threat_nonlethal if m[key] is not None]
        out(f"  {label}: lethal n={len(lv)} mean={np.mean(lv) if lv else float('nan'):.2f} median={np.median(lv) if lv else float('nan'):.2f} "
            f"| non-lethal n={len(nv)} mean={np.mean(nv) if nv else float('nan'):.2f} median={np.median(nv) if nv else float('nan'):.2f} "
            f"| AUC={_auc(lv, nv):.3f}")
    n_lethal_no_escape = sum(1 for m in threat_lethal if not m["has_safe_escape_now"])
    n_nonlethal_no_escape = sum(1 for m in threat_nonlethal if not m["has_safe_escape_now"])
    out(f"\n  already-no-safe-escape-route at the moment of the threatening placement: "
        f"lethal={n_lethal_no_escape}/{len(threat_lethal)} non-lethal={n_nonlethal_no_escape}/{len(threat_nonlethal)}")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation3_c2_precursor_v2.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item3a()
    item3b()
    item3c()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
