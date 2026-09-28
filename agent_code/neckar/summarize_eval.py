"""Aggregate checkpoint evaluations (checkpoint_comparison.csv from run_eval.py) over runs.

For every group (e.g. "CNN greedy", "CNN sampled") it reads one comparison CSV per run,
matches checkpoints by name (ep500, ep1000, ...), and reports mean +/- std across runs.

    python summarize_eval.py \
        --group "greedy"  "results/task2_v2/run_*/checkpoint_comparison_greedy.csv" \
        --group "sampled" "results/task2_v2/run_*/checkpoint_comparison_sampled.csv" \
        --metrics score_mean coins_mean crates_mean suicide_rate_% survived_round_rate_% \
        --out plots/task2_eval

Outputs (in --out):
    eval_summary.csv          mean/std per group, checkpoint and metric
    eval_<metric>.png         metric vs. training episode, one line per group (+/- 1 std band)
and prints a table to the terminal.

If greedy and sampled results are in the SAME file, add --split-by:
    --split-by auto      policy read from the checkpoint label (ep2000_greedy, ep2000_sample)
    --split-by policy    policy read from a column called 'policy'
Each group is then split into '<label> greedy' and '<label> sampled'. With --latex-checkpoint ep2000 it also prints a LaTeX
table row per group for that checkpoint.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = ["score_mean", "coins_mean", "crates_mean", "bombs_mean", "steps_mean",
                   "suicide_rate_%", "killed_opponent_rate_%", "survived_round_rate_%"]


def episode_of(label: str) -> int:
    match = re.search(r"ep(\d+)", str(label))
    return int(match.group(1)) if match else -1


def load_group(pattern: str) -> pd.DataFrame:
    """All runs of one group stacked into one table with a 'run' column."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["run"] = os.path.basename(os.path.dirname(os.path.abspath(path)))
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    table["episode"] = table["checkpoint"].map(episode_of)
    return table


def policy_from_label(label: str) -> str:
    """Read the policy from a checkpoint label such as 'ep2000_greedy' or 'ep2000-sample'."""
    match = re.search(r"(greedy|sampl\w*)", str(label), flags=re.IGNORECASE)
    if not match:
        return "unknown"
    return "greedy" if match.group(1).lower() == "greedy" else "sampled"


def split_groups(groups: dict, split_by: str) -> dict:
    """Split every group by policy, e.g. 'CNN' -> 'CNN greedy' and 'CNN sampled'.

    split_by = "auto":   policy read from the checkpoint label (ep2000_greedy, ep2000_sample, ...)
    split_by = <column>: policy read from that column of the CSV (e.g. 'policy')
    """
    result = {}
    for label, table in groups.items():
        if split_by == "auto":
            key = table["checkpoint"].map(policy_from_label)
        elif split_by in table.columns:
            key = table[split_by].astype(str)
        else:
            raise KeyError(f"column {split_by!r} not found; available: {list(table.columns)}")
        for value, part in table.groupby(key, sort=False):
            result[f"{label} {value}".strip()] = part.reset_index(drop=True)
    return result


def summarize(groups: dict, metrics: list) -> pd.DataFrame:
    rows = []
    for label, table in groups.items():
        n_runs_total = table["run"].nunique()
        for episode, part in table.groupby("episode"):
            row = {"group": label, "checkpoint": f"ep{episode}", "episode": episode,
                   "runs": part["run"].nunique()}
            if row["runs"] < n_runs_total:
                print(f"[eval] {label} ep{episode}: only {row['runs']}/{n_runs_total} runs have it")
            for metric in metrics:
                if metric not in part.columns:
                    continue
                values = part[metric].astype(float)
                row[f"{metric}"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "episode"]).reset_index(drop=True)


def plot_metric(summary: pd.DataFrame, metric: str, out_dir: str):
    if metric not in summary.columns:
        return
    plt.figure(figsize=(8, 4.5))
    for label, part in summary.groupby("group", sort=False):
        x = part["episode"].to_numpy()
        mean = part[metric].to_numpy()
        std = part[f"{metric}_std"].fillna(0).to_numpy()
        line, = plt.plot(x, mean, marker="o", label=label)
        plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)
    plt.xlabel("Training episode of checkpoint")
    plt.ylabel(metric)
    plt.title(f"Evaluation: {metric} (mean over runs, shaded: ±1 std)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_]+", "", metric)
    plt.savefig(os.path.join(out_dir, f"eval_{safe}.png"), dpi=150)
    plt.close()


def latex_rows(summary: pd.DataFrame, checkpoint: str, metrics: list):
    part = summary[summary["checkpoint"] == checkpoint]
    if part.empty:
        print(f"[eval] no results for {checkpoint}")
        return
    print(f"\n% LaTeX rows for {checkpoint}: " + " & ".join(metrics))
    for _, row in part.iterrows():
        cells = []
        for metric in metrics:
            if metric in row:
                std = row.get(f"{metric}_std", float("nan"))
                std_txt = "" if pd.isna(std) else f" \\pm {std:.2f}"
                cells.append(f"${row[metric]:.2f}{std_txt}$")
        print(f"{row['group']} & " + " & ".join(cells) + " \\\\")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs=2, action="append", metavar=("LABEL", "PATTERN"),
                        required=True, help="label and quoted glob pattern for its comparison CSVs")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help="columns of checkpoint_comparison.csv to summarize")
    parser.add_argument("--latex-checkpoint", default=None,
                        help="also print LaTeX table rows for this checkpoint, e.g. ep2000")
    parser.add_argument("--split-by", default=None,
                        help="split each group by policy when greedy and sampled results are in the "
                             "same file: 'auto' reads it from the checkpoint label (e.g. ep2000_greedy), "
                             "or give the name of a column that holds the policy (e.g. policy)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out", default="plots/eval")
    args = parser.parse_args()

    groups = {label: load_group(pattern) for label, pattern in args.group}
    if args.split_by:
        groups = split_groups(groups, args.split_by)
    for label, table in groups.items():
        print(f"[eval] group {label!r}: {table['run'].nunique()} run(s), "
              f"checkpoints {sorted(int(e) for e in table['episode'].unique())}")

    summary = summarize(groups, args.metrics)

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "eval_summary.csv"), index=False)

    shown = ["group", "checkpoint", "runs"] + [c for m in args.metrics
                                               for c in (m, f"{m}_std") if c in summary.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Evaluation: mean and std across runs, per checkpoint ===")
    print(summary[shown].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if not args.no_plots:
        for metric in args.metrics:
            plot_metric(summary, metric, args.out)

    if args.latex_checkpoint:
        latex_rows(summary, args.latex_checkpoint, args.metrics)

    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()