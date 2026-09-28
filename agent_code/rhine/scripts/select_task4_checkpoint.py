"""Offline Task 4 checkpoint selection: rebuilds run_stage_a2_task3.py's
_select_best_checkpoint() input (dicts of round, snapshot_path, score_mean,
self_kill_rate, got_killed_by_opponent_rate, oscillation_fraction,
completion_rate) from run_task4_async_eval.py's CSV and calls that function
unchanged. Run manually after a seed's training and evaluator have finished.

Usage:
    python -m agent_code.rhine.scripts.select_task4_checkpoint --seed 0 \
        --eval-csv agent_code/rhine/logs/task4_async_eval_seed0.csv \
        --out agent_code/rhine/models/task4_seed0.pt
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task3 import _select_best_checkpoint

HISTORY_FIELDS = (
    "round", "snapshot_path", "score_mean", "self_kill_rate",
    "got_killed_by_opponent_rate", "oscillation_fraction", "completion_rate",
)


def _load_history(csv_path: Path) -> list:
    history = []
    for row in common._read_csv_rows(csv_path):
        history.append({
            "round": int(row["round"]),
            "snapshot_path": Path(row["snapshot_path"]),
            "score_mean": float(row["score_mean"]),
            "self_kill_rate": float(row["self_kill_rate"]),
            "got_killed_by_opponent_rate": float(row["got_killed_by_opponent_rate"]),
            "oscillation_fraction": float(row["oscillation_fraction"]),
            "completion_rate": float(row["completion_rate"]),
        })
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    history = _load_history(Path(args.eval_csv))
    if not history:
        print(f"FAIL: no rows in {args.eval_csv}")
        sys.exit(1)

    selection = _select_best_checkpoint(history)
    best = selection["chosen"]
    print(
        f"seed={args.seed}: shortlisted top {len(selection['shortlist'])} eval points by score_mean; "
        f"{len(selection['qualified'])}/{len(selection['shortlist'])} passed the quality filter."
    )
    for h in selection["shortlist"]:
        reasons = selection["shortlist_reasons"][h["round"]]
        status = "PASS" if not reasons else "FILTERED (" + "; ".join(reasons) + ")"
        print(
            f"  candidate round={h['round']} score_mean={h['score_mean']:.2f} "
            f"self_kill_rate={h['self_kill_rate']:.3f} "
            f"got_killed_by_opponent_rate={h['got_killed_by_opponent_rate']:.3f} "
            f"oscillation_fraction={h['oscillation_fraction']:.3f} -> {status}"
        )
    print(
        f"Selected round={best['round']} score_mean={best['score_mean']:.2f} "
        f"self_kill_rate={best['self_kill_rate']:.3f} "
        f"got_killed_by_opponent_rate={best['got_killed_by_opponent_rate']:.3f} "
        f"oscillation_fraction={best['oscillation_fraction']:.3f} "
        f"completion_rate={best['completion_rate']:.3f}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["snapshot_path"], out_path)

    if not filecmp.cmp(best["snapshot_path"], out_path, shallow=False):
        print(f"FAIL: {out_path} is not byte-identical to {best['snapshot_path']}")
        sys.exit(1)
    print(f"OK: {out_path} written and verified byte-identical to {best['snapshot_path']}")


if __name__ == "__main__":
    main()
