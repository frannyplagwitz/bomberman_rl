"""Task 4 offline investigation: reward audit. Read-only on config.py's
RewardConfig and the breaker_only recordings' per-round metrics (event
counts; per-step rewards are not recorded).

Also prints the earlier kill-vs-crate sensitivity conclusion unverified.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None


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
    parser.add_argument("--out", default=str(DATA_DIR / "investigation2_reward_audit.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== Item 6: reward audit ===")
    out("\n--- reward constants (config.py RewardConfig, current values) ---")
    rc = cfg.REWARD_CONFIG
    rows = [
        ("COIN_REWARD", rc.COIN_REWARD), ("STEP_COST", rc.STEP_COST),
        ("CRATE_DESTROYED_REWARD", rc.CRATE_DESTROYED_REWARD),
        ("CRATE_DESTROYED_REWARD_NO_COIN", rc.CRATE_DESTROYED_REWARD_NO_COIN),
        ("SELF_KILL_PENALTY", rc.SELF_KILL_PENALTY),
        ("TRAINING_KILLED_OPPONENT_REWARD", rc.TRAINING_KILLED_OPPONENT_REWARD),
        ("TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY", rc.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY),
        ("BOMBING_PROGRESS_SHAPING_WEIGHT", rc.BOMBING_PROGRESS_SHAPING_WEIGHT),
        ("STALL_PENALTY", rc.STALL_PENALTY), ("STALL_PENALTY_V2", rc.STALL_PENALTY_V2),
        ("WASTEFUL_BOMB_PENALTY", rc.WASTEFUL_BOMB_PENALTY),
    ]
    rows_sorted = sorted(rows, key=lambda kv: -abs(kv[1]))
    out(f"{'name':>42} {'value':>10} {'|value| rank':>13}")
    for i, (name, val) in enumerate(rows_sorted):
        out(f"{name:>42} {val:>10.3f} {i + 1:>13}")
    out(f"\ntoggles: ENABLE_BOMBING_PROGRESS_SHAPING={rc.ENABLE_BOMBING_PROGRESS_SHAPING} "
        f"ENABLE_STALL_PENALTY={rc.ENABLE_STALL_PENALTY} ENABLE_STALL_PENALTY_V2={rc.ENABLE_STALL_PENALTY_V2} "
        f"ENABLE_CRATE_NO_COIN_BONUS={rc.ENABLE_CRATE_NO_COIN_BONUS} "
        f"ENABLE_WASTEFUL_BOMB_PENALTY={rc.ENABLE_WASTEFUL_BOMB_PENALTY}")
    out(f"relative scale: SELF_KILL_PENALTY is {abs(rc.SELF_KILL_PENALTY / rc.COIN_REWARD):.1f}x COIN_REWARD; "
        f"TRAINING_KILLED_OPPONENT_REWARD is {rc.TRAINING_KILLED_OPPONENT_REWARD / rc.COIN_REWARD:.1f}x COIN_REWARD "
        f"and {rc.TRAINING_KILLED_OPPONENT_REWARD / rc.CRATE_DESTROYED_REWARD:.1f}x CRATE_DESTROYED_REWARD; "
        f"note the real PDF score (settings.py) uses coin=1/kill=5, a 1:5 ratio, decoupled from these "
        f"training-shaping values (see config.py's own docstring on this decoupling).")

    out("\n--- per-round reward-relevant event-count means (from recorded agg, 3 seeds x 100 rounds) ---")
    tot = {}
    for s in SEEDS:
        payload = load(s)
        agg = payload["agg"]
        for key in ("coins_collected_mean", "crates_destroyed_mean", "bombs_dropped_mean",
                    "opponent_kills_mean", "self_kill_rate", "got_killed_by_opponent_rate",
                    "wait_fraction_mean"):
            tot.setdefault(key, []).append(agg[key])
        out(f"  seed{s}: coins/round={agg['coins_collected_mean']:.2f} crates/round={agg['crates_destroyed_mean']:.2f} "
            f"bombs/round={agg['bombs_dropped_mean']:.1f} kills/round={agg['opponent_kills_mean']:.2f} "
            f"self_kill_rate={agg['self_kill_rate']:.3f} got_killed_rate={agg['got_killed_by_opponent_rate']:.3f} "
            f"wait_fraction={agg['wait_fraction_mean']:.3f}")
    out("\n  3-seed mean±std (ddof=1):")
    for key, vals in tot.items():
        arr = np.array(vals)
        out(f"    {key}: {arr.mean():.3f}±{arr.std(ddof=1):.3f}")
    out("\nNote: per-step/per-event RAW reward values (as actually added to the training signal) are not "
        "stored in eval_stage_d_final.py's recording -- only these aggregate per-round event counts/rates "
        "are available offline; a literal 'reward received per event, from the recording' cannot be "
        "computed from this data (would need re-running compute_reward() live during training/eval, out "
        "of scope for this read-only investigation).")

    out("\n--- earlier kill-vs-crate sensitivity conclusion (restated verbatim, not re-verified) ---")
    out(
        "Goal: check whether expected_kill_value_at_target(#34, now #28 in the 31-dim vector) is numerically much smaller than "
        "crates_destructible_at_target(#18), and whether the network really treats it as a weak signal.\n"
        "Numeric scale: #18 realized std=0.179-0.180, #34 realized std=0.311-0.322 -- #34 is larger than #18, so the 'scale too small' hypothesis fails.\n"
        "First-layer weight L2 norm x realized std: #18 ranks first (amplified ~4.3-5.0x), #34 ranks 24-30 (only ~1.1-1.2x) -- training barely "
        "amplified this dimension's weights.\n"
        "Crate-baseline correlations: crate r=0.150/0.274/0.218; kill r=0.121/0.138/0.288 -- similar magnitude in this population.\n"
        "Conclusion: the numeric-scale hypothesis is rejected; first-layer amplification is the most robust evidence, consistent across all three seeds, that the network perceives the kill signal weakly. "
        "Overall root cause: an exploration problem from sparse kill rewards, not missing information in the feature itself."
    )
    out("\nThis Task 4 investigation's own item d.3 (investigate_task4_core.py) found the same qualitative "
        "pattern with the current 31-dim breaker_only checkpoints: #18 first-layer amplification x3.9-4.1 "
        "(rank 3-5/31) vs #28 (kill value) x1.1-1.2 (rank 27/31) -- consistent with, not a re-verification "
        "of, the restated Stage C conclusion above.")

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
