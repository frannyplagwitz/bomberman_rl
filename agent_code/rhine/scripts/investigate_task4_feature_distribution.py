"""Task 4 offline investigation: full 31-dim feature distribution over the
trained breaker_only checkpoint's evaluation steps. Read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Compared with the earlier untrained-model numbers from
run_task4_smoke_test.py. Raw #29/#31 values are recomputed from the recorded
state with the production functions (alive_opponent_distances(),
reachable_space_count()).
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import alive_opponent_distances, bfs_distances, extract_semantic_state, reachable_space_count

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None

FEATURE_NAMES = [
    "can_move_up", "can_move_down", "can_move_left", "can_move_right", "has_reachable_coin",
    "nearest_reachable_coin_distance", "coin_path_up", "coin_path_down", "coin_path_left", "coin_path_right",
    "bomb_available", "has_bombing_target", "nearest_bombing_position_distance", "bombing_path_up",
    "bombing_path_down", "bombing_path_left", "bombing_path_right", "crates_destructible_at_target",
    "current_tile_in_danger", "nearest_threat_timer", "coin_contested", "has_kill_target",
    "nearest_kill_distance", "kill_direction_up", "kill_direction_down", "kill_direction_left",
    "kill_direction_right", "expected_kill_value_at_target", "nearest_alive_opponent_distance",
    "opponents_within_3", "reachable_space",
]


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


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation2_feature_distribution.txt"))
    parser.add_argument("--raw-sample-stride", type=int, default=5,
                         help="Recomputing raw #29/#31 needs a full BFS per step; subsample every Nth "
                              "step across the 300 rounds to keep this tractable while still giving a "
                              "large, representative sample.")
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== Item 5: full 31-dim feature distribution (group breaker_only, 3 seeds x 100 rounds) ===")

    all_feats = []
    all_states_subsampled = []
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            for i, x in enumerate(r["steps"]):
                all_feats.append(x["features"])
                if i % args.raw_sample_stride == 0:
                    all_states_subsampled.append(x["state"])
    feats = np.stack(all_feats)
    n = feats.shape[0]
    out(f"\ntotal steps: {n} (3 seeds x 100 rounds)")

    qs = [0, 10, 25, 50, 75, 90, 99, 100]
    out(f"\n{'idx':>4} {'name':>32} {'min':>7} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'p99':>7} {'max':>7} {'sat@1.0%':>9} {'zero%':>7}")
    for j in range(31):
        col = feats[:, j]
        pct = np.percentile(col, qs)
        sat = 100 * np.mean(col >= 1.0)
        zero = 100 * np.mean(col == 0.0)
        out(f"{j + 1:>4} {FEATURE_NAMES[j]:>32} " + " ".join(f"{v:>7.3f}" for v in pct) + f" {sat:>9.2f} {zero:>7.2f}")

    out(f"\n--- raw (un-normalized) #29/#31 recompute, subsampled every {args.raw_sample_stride}th step "
        f"(n={len(all_states_subsampled)}) ---")
    raw29, raw31 = [], []
    for st in all_states_subsampled:
        gs = restore_state(st)
        sem = extract_semantic_state(gs)
        dist_map = bfs_distances(sem.field_arr, sem.self_pos, blocked=sem.blocked)
        opp_dists = alive_opponent_distances(sem.field_arr, sem.blocked, dist_map, sem.opponents)
        raw29.append(min(opp_dists.values()) if opp_dists else None)
        raw31.append(reachable_space_count(sem.field_arr, sem.self_pos, sem.opponents, cfg.REACHABLE_SPACE_DEPTH_CAP))

    raw29_reachable = [v for v in raw29 if v is not None]
    n_gt16 = sum(1 for v in raw29_reachable if v > cfg.DISTANCE_FEATURE_NORM)
    out(f"\n#29 raw (reachable only, n={len(raw29_reachable)}/{len(raw29)}): "
        f"quantiles(min/p25/p50/p75/p90/max)={np.percentile(raw29_reachable, [0, 25, 50, 75, 90, 100]) if raw29_reachable else 'n/a'} "
        f"raw>16 (would clip in the normalized feature): {n_gt16}/{len(raw29_reachable)} "
        f"({100 * n_gt16 / len(raw29_reachable) if raw29_reachable else float('nan'):.2f}%)")
    n_unreachable = len(raw29) - len(raw29_reachable)
    out(f"#29 unreachable (no opponent reachable this way, feature=1.0 by definition): "
        f"{n_unreachable}/{len(raw29)} ({100 * n_unreachable / len(raw29):.2f}%)")

    n_ge40 = sum(1 for v in raw31 if v >= cfg.REACHABLE_SPACE_NORM)
    out(f"\n#31 raw (n={len(raw31)}): quantiles(min/p25/p50/p75/p90/max)="
        f"{np.percentile(raw31, [0, 25, 50, 75, 90, 100])} "
        f"raw>={cfg.REACHABLE_SPACE_NORM} (saturates the normalized feature): {n_ge40}/{len(raw31)} "
        f"({100 * n_ge40 / len(raw31):.2f}%)")

    out("\n--- comparison to the earlier smoke-test distribution "
        "(run_task4_smoke_test.py, 30 rounds, RANDOM-INIT untrained model) ---")
    out("  prior (untrained, random policy): #31 min/p25/median/p75/p90/max=1/23/34/46/54/65, "
        "saturated@1.0=37.45%; #29 reachable-only min/p25/median/p75/max=1/5/7/11/30, reachable=82.9%; "
        "#30 counts 0/1/2/3 = 9412/1353/37/4, mean=0.133.")
    out("  current (trained breaker_only checkpoint) numbers are in the table above -- not reproduced "
        "here as a second copy; compare directly. Any difference reflects the trained policy visiting "
        "different parts of the state space than a random policy (e.g. more often near opponents while "
        "hunting, or in more open post-clearing states), not a change in the feature computation itself.")

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
