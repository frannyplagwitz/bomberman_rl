"""Compare per-episode metrics across multiple training runs.

Loads one or more CSVs written by episode_logger.EpisodeCSVLogger and
plots rolling-average comparisons across runs

Usage:
    python plot_comparison.py logs/episodes-run1.csv logs/episodes-run2.csv \
        --labels "wasteful=-0.4" "wasteful=-0.3" --window 15 --out plots/comparison
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(paths, labels=None):
    """Load one or more episode CSVs into a single labeled DataFrame."""
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"Got {len(paths)} files but {len(labels)} labels - must match 1:1")

    frames = []
    for path, label in zip(paths, labels):
        df = pd.read_csv(path)
        df["run_label"] = label
        df = df.sort_values("episodes").reset_index(drop=True)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _plot_rolling_metric(df, column, window, ylabel, title, out_path):
    plt.figure(figsize=(9, 5))
    for label, sub in df.groupby("run_label", sort=False):
        sub = sub.sort_values("episodes")
        rolling = sub[column].rolling(window, min_periods=1).mean()
        plt.plot(sub["episodes"], rolling, label=label)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({window}-episode rolling average)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_rolling_rate(df, event_column, window, ylabel, title, out_path):
    """Same as _plot_rolling_metric, but for a 0/1 event column, plotted as a rate (%)."""
    plt.figure(figsize=(9, 5))
    for label, sub in df.groupby("run_label", sort=False):
        sub = sub.sort_values("episodes")
        rolling_rate = 100.0 * sub[event_column].rolling(window, min_periods=1).mean()
        plt.plot(sub["episodes"], rolling_rate, label=label)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({window}-episode rolling average)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_comparison(df: pd.DataFrame, window: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Task-performance metrics
    _plot_rolling_metric(df, "crates_destroyed", window, "Crates destroyed / episode",
                          "Crate-hunting output", os.path.join(out_dir, "compare_crates.png"))
    _plot_rolling_metric(df, "coins_collected", window, "Coins collected / episode",
                          "Coin collection", os.path.join(out_dir, "compare_coins.png"))
    _plot_rolling_metric(df, "bombs_dropped", window, "Bombs dropped / episode",
                          "Bombing volume", os.path.join(out_dir, "compare_bombs_dropped.png"))
    _plot_rolling_metric(df, "useful_rate", window, "Useful bomb rate (%)",
                          "Bombing precision", os.path.join(out_dir, "compare_useful_rate.png"))

    # Survival / outcome metrics
    _plot_rolling_rate(df, "got_killed", window, "Death rate (%)",
                        "Overall death rate", os.path.join(out_dir, "compare_death_rate.png"))
    _plot_rolling_rate(df, "killed_self", window, "Self-kill rate (%)",
                        "Self-kill rate", os.path.join(out_dir, "compare_self_kill_rate.png"))
    _plot_rolling_rate(df, "killed_opponent", window, "Killed-opponent rate (%)",
                        "Opponent eliminations (agent survives)",
                        os.path.join(out_dir, "compare_killed_opponent_rate.png"))
    # "opponent_killed" is what this used to call "opponent_eliminated" (see train.py's
    # episode row) - it already covers any elimination, including mutual kills.
    _plot_rolling_rate(df, "opponent_killed", window, "Opponent-eliminated rate (%)",
                        "Any opponent elimination (incl. mutual kills)",
                        os.path.join(out_dir, "compare_opponent_eliminated_rate.png"))

    # Terminal-action breakdown - the specific diagnostic behind the frozen-WAIT bug
    df = df.copy()
    df["terminal_wait"] = (df["terminal_action"] == "WAIT").astype(int)
    _plot_rolling_rate(df, "terminal_wait", window, "Deaths ending in WAIT (%)",
                        "Terminal action = WAIT, among all episodes",
                        os.path.join(out_dir, "compare_terminal_wait.png"))

    # Convergence diagnostics
    _plot_rolling_metric(df, "value_error", window, "|Critic V(s) - Shaped Reward| (terminal)",
                          "Terminal value error", os.path.join(out_dir, "compare_value_error.png"))
    _plot_rolling_metric(df, "entropy", window, "Policy entropy",
                          "Entropy", os.path.join(out_dir, "compare_entropy.png"))

    print(f"Saved comparison plots to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="episode CSV files to compare")
    parser.add_argument("--labels", nargs="*", default=None,
                         help="one label per CSV, in the same order (defaults to filenames)")
    parser.add_argument("--window", type=int, default=15,
                         help="rolling-average window in episodes (default 15, matching the "
                              "15-round bucketing used throughout this project's analysis)")
    parser.add_argument("--out", default="plots/comparison", help="output directory for PNGs")
    args = parser.parse_args()

    df = load_runs(args.csv_paths, args.labels)
    plot_comparison(df, window=args.window, out_dir=args.out)


if __name__ == "__main__":
    main()