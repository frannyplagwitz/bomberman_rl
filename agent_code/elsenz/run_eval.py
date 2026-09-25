"""Evaluator script
Reads all currently available checkpoints, tests the game for N_ROUNDS rounds for each,
finds the averages of several metrics, and outputs those values into a comparison file.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import argparse

import pandas as pd

CKPT_PATTERN = "actor-critic-lut-peaceful-classic-*-rounds-1-opponents-ep*.pt"
EVAL_FILENAME = "elsenz-lut-peaceful-classic.pt"
AGENT_NAME = "elsenz"

STATS_DIR = "eval_stats"
OUT_DIR = "eval_results"
N_ROUNDS = 100


def build_eval_cmd(stats_path: str):
    """Command to run one checkpoint's evaluation, saving results to stats_path."""
    return [
        "python", "../../main.py", "play",
        "--agents", AGENT_NAME, "random_agent", "random_agent", "random_agent", "random_agent", "random_agent",
        "--scenario", "classic",
        "--n-rounds", str(N_ROUNDS), "--no-gui",
        "--save-stats", stats_path,
    ]


def find_checkpoints(pattern=CKPT_PATTERN):
    """Find every checkpoint matching ep<NUM> naming pattern and sort numerically"""
    
    files = glob.glob(pattern)

    def episode_number(path):
        match = re.search(r'ep(\d+)\.pt$', path)
        return int(match.group(1)) if match else -1

    return sorted(files, key=episode_number)


def get_checkpoint_label(filename: str) -> str:
    """Extract 'ep###' from '...ep###.pt"""
    match = re.search(r'(ep\d+)\.pt$', filename)
    if not match:
        raise ValueError(f"No 'ep<N>.pt' suffix found in {filename}")
    return match.group(1)


def summarize_checkpoint(stats_path: str, label: str, agent_name: str = AGENT_NAME) -> dict:
    """Collapse one checkpoint's --save-stats JSON into a single summary row.
    
    Note: kills/suicides absent from agent's entry, so lookup uses .get(key, 0)
    """
    with open(stats_path) as f:
        data = json.load(f)

    by_agent = data.get("by_agent", {})
    agent_stats = by_agent.get(agent_name, {})
    n_rounds = agent_stats.get("rounds", 0)

    row = {"checkpoint": label, "n_rounds": n_rounds}
    if n_rounds == 0:
        return row

    # Per-round means for totals reported across the whole eval batch
    for key in ["score", "coins", "crates", "bombs", "invalid", "moves", "steps"]:
        row[f"{key}_mean"] = round(agent_stats.get(key, 0) / n_rounds, 3)

    own_suicides = agent_stats.get("suicides", 0)
    own_kills = agent_stats.get("kills", 0)  # times THIS agent killed an opponent
    row["suicide_rate_%"] = round(100.0 * own_suicides / n_rounds, 1)
    row["killed_opponent_rate_%"] = round(100.0 * own_kills / n_rounds, 1)

    # Times killed BY an opponent: inferred from the opponent's own "kills" tally
    opponent_kills_on_us = sum(
        stats.get("kills", 0) for name, stats in by_agent.items() if name != agent_name
    )
    row["got_killed_rate_%"] = round(100.0 * opponent_kills_on_us / n_rounds, 1)

    # Estimated survival: rounds not ended by our own suicide or being killed.
    # Inferred from two ways agent can die 
    survived = max(0, n_rounds - own_suicides - opponent_kills_on_us)
    row["survived_round_rate_%"] = round(100.0 * survived / n_rounds, 1)

    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sort-by", default="survived_round_rate_%",
                         help="Column to sort the comparison table by, descending")
    parser.add_argument("--out", default="checkpoint_comparison.csv",
                         help="Where to write the summary CSV")
    args = parser.parse_args()

    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoints = find_checkpoints()

    if not checkpoints:
        print(f"No checkpoints found matching the pattern: {CKPT_PATTERN}")
        return

    print(f"Found {len(checkpoints)} checkpoints: "
          f"{[get_checkpoint_label(c) for c in checkpoints]}")

    summary_rows = []

    for ckpt_path in checkpoints:
        label = get_checkpoint_label(ckpt_path)
        print(f"\n=== Testing {label} ({ckpt_path}) ===")

        shutil.copy(ckpt_path, EVAL_FILENAME)

        stats_path = os.path.join(STATS_DIR, f"{label}.json")
        result = subprocess.run(build_eval_cmd(stats_path))
        
        if result.returncode != 0:
            print(f"  eval run failed for {label} (exit code {result.returncode}), skipping")
            continue

        if not os.path.isfile(stats_path):
            print(f"  WARNING: expected stats file not found at {stats_path}, skipping")
            continue

        summary_rows.append(summarize_checkpoint(stats_path, label))
        print(f"  saved -> {stats_path}")

    if not summary_rows:
        print("\nNo checkpoint produced usable results.")
        return

    summary = pd.DataFrame(summary_rows)

    if args.sort_by in summary.columns:
        summary = summary.sort_values(args.sort_by, ascending=False)
    else:
        print(f"\nWarning: sort column '{args.sort_by}' not found, leaving unsorted. "
              f"Available columns: {list(summary.columns)}")

    summary.to_csv(args.out, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== Checkpoint comparison ===")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison to {args.out}")


if __name__ == "__main__":
    main()