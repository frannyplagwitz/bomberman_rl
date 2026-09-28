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
 
from pathlib import Path
 
import pandas as pd
 
# Resolve main.py by an absolute path to work no matter the directory invoked from 
MAIN_PY = Path(__file__).resolve().parent / "../../main.py"
 
# Match whatever model/behavior this agent is currently configured for
MODEL_TYPE = os.environ.get("NECKAR_MODEL_TYPE", "cnn")
BEHAVIOR = os.environ.get("NECKAR_BEHAVIOR", "peaceful")
SCENARIO = os.environ.get("NECKAR_SCENARIO", "coin_heaven")

CKPT_PATTERN = f"actor-critic-{MODEL_TYPE}-{BEHAVIOR}-{SCENARIO}-*-rounds-*-opponents-ep*.pt"
EVAL_FILENAME = f"neckar-{MODEL_TYPE}-{BEHAVIOR}-{SCENARIO}.pt"
AGENT_NAME = "neckar"
EVAL_POLICIES = ["greedy", "sample"]
 
N_ROUNDS = 200 # Originally 100, but switched to 200 because added greedy/stochastic sampling comparison
 
 
def build_eval_cmd(stats_path: str, task: str):
    """Command to run one checkpoint's evaluation, saving results to stats_path.
    Params: 
    stats_path: path that stats are saved in 
    task: task that is being run, will define scenario to test
    """
    if task == "task1":
        return [
        "python", str(MAIN_PY.resolve()), "play",
        "--agents", AGENT_NAME, "random_agent", "random_agent", "random_agent",
        "--scenario", "coin-heaven",
        "--n-rounds", str(N_ROUNDS), "--no-gui",
        "--save-stats", stats_path,
        ]
        
    else: 
        return [
        "python", str(MAIN_PY.resolve()), "play",
        "--agents", AGENT_NAME, "random_agent", "random_agent", "random_agent",
        "--scenario", "classic",
        "--n-rounds", str(N_ROUNDS), "--no-gui",
        "--save-stats", stats_path,
        ]

 
def find_checkpoints(run_dir: str, pattern=CKPT_PATTERN):
    """Find every checkpoint matching ep<NUM> naming pattern inside run_dir and sort numerically"""
 
    files = glob.glob(os.path.join(run_dir, pattern))
 
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
    kills/suicides aren't available, so grab them from .get(key, 0)
    
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
 
    # Estimated survival =  rounds not ended by our own suicide or being killed.
    # Inferred from two ways agent can die
    survived = max(0, n_rounds - own_suicides - opponent_kills_on_us)
    row["survived_round_rate_%"] = round(100.0 * survived / n_rounds, 1)
 
    return row
 
 
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=".",
                         help="Folder holding this run's checkpoints (e.g. results/task1/run_1). "
                              "Also where eval_stats/, eval_results/ and the comparison CSV are "
                              "written, and where the agent stages the checkpoint it's evaluating. "
                              "Defaults to the current directory (single-run usage, unchanged).")
    parser.add_argument("--task", default="task1")
    parser.add_argument("--sort-by", default="survived_round_rate_%",
                         help="Column to sort the comparison table by, descending")
    parser.add_argument("--out", default=None,
                         help="Where to write the summary CSV. Defaults to "
                              "<run-dir>/checkpoint_comparison.csv")
    args = parser.parse_args()
 
    run_dir = args.run_dir
    task = args.task
    stats_dir = os.path.join(run_dir, "eval_stats")
    out_dir = os.path.join(run_dir, "eval_results")
    out_path = args.out or os.path.join(run_dir, "checkpoint_comparison.csv")
    eval_filename = os.path.join(run_dir, EVAL_FILENAME)
 
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = find_checkpoints(run_dir)
 
    if not checkpoints:
        print(f"No checkpoints found matching {CKPT_PATTERN!r} inside {run_dir!r}")
        return
 
    print(f"[{run_dir}] Found {len(checkpoints)} checkpoints: "
          f"{[get_checkpoint_label(c) for c in checkpoints]}")
 
    # Tell the agent which run's checkpoint to load/stage — see module docstring.
    subprocess_env = os.environ.copy()
    subprocess_env["NECKAR_RUN_DIR"] = run_dir
 
    summary_rows = []
 
    for ckpt_path in checkpoints:
        label = get_checkpoint_label(ckpt_path)
 
        shutil.copy(ckpt_path, eval_filename)
        
        # For each policy (greedy, sample), run test evaluation 
        for policy in EVAL_POLICIES:
            print(f"\n=== [{run_dir}] Testing {label} with {policy} policy ===")

            env = subprocess_env.copy()
            env["NECKAR_EVAL_POLICY"] = policy
        
            # Obtain the path where we will save the output        
            stats_path = os.path.join(stats_dir, f"{label}-{policy}.json")
            if os.path.exists(stats_path):
                os.remove(stats_path)   # never summarize a stale file
        
            # Run the eval command     
            result = subprocess.run(build_eval_cmd(stats_path, task), env=env)
            if result.returncode != 0 or not os.path.isfile(stats_path):
                print(f"  eval failed for {label} ({policy}), skipping")
                continue
        
            # Create a row for the checkpoint 
            row = summarize_checkpoint(stats_path, label)
            row["policy"] = policy
            summary_rows.append(row)


    if not summary_rows:
        print(f"\n[{run_dir}] No checkpoint produced usable results.")
        return
 
    summary = pd.DataFrame(summary_rows)
 
    if args.sort_by in summary.columns:
        summary = summary.sort_values(args.sort_by, ascending=False)
    else:
        print(f"\nWarning: sort column '{args.sort_by}' not found, leaving unsorted. "
              f"Available columns: {list(summary.columns)}")
 
    summary.to_csv(out_path, index=False)
 
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(f"\n=== [{run_dir}] Checkpoint comparison ===")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison to {out_path}")
 
 
if __name__ == "__main__":
    main()