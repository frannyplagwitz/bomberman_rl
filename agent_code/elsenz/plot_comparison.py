"""Compare per-episode metrics across multiple LUT training runs.

Loads one or more CSVs written by episode_logger.EpisodeCSVLogger and
plots rolling-average comparisons across runs.

Usage:
    python plot_comparison.py logs/episodes-run1.csv logs/episodes-run2.csv \
        v1:--labels "eps=300" "eps=2500" --window 15 --out plots/comparison
        v2:--labels "wasteful=-0.4" "wasteful=-0.3" --window 15 --out plots/comparison
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
        # df = df.sort_values("episode").reset_index(drop=True) check if it is singular or plural
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _plot_rolling_metric(df, column, window, ylabel, title, out_path):
    plt.figure(figsize=(9, 5))
    for label, sub in df.groupby("run_label", sort=False):
        sub = sub.sort_values("episodes")
        # sub = sub.sort_values("episode") #singular/plural?
        rolling = sub[column].rolling(window, min_periods=1).mean()
        plt.plot(sub["episodes"], rolling, label=label)
        # plt.plot(sub["episode"], rolling, label=label) # singular/plural?

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
        # sub = sub.sort_values("episode")
        rolling_rate = 100.0 * sub[event_column].rolling(window, min_periods=1).mean()
        plt.plot(sub["episodes"], rolling_rate, label=label)
        # plt.plot(sub["episode"], rolling_rate, label=label)

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
    _plot_rolling_metric(df, "crates_destroyed", window,
                         "Crates destroyed / episode", "Crate destruction",
                         os.path.join(out_dir, "compare_crates.png"))

    _plot_rolling_metric(df, "coins_collected", window,
                         "Coins collected / episode", "Coin collection",
                         os.path.join(out_dir, "compare_coins.png"))

    _plot_rolling_metric(df, "bombs_dropped", window,
                         "Bombs dropped / episode", "Bombing volume",
                         os.path.join(out_dir, "compare_bombs_dropped.png"))

    _plot_rolling_metric(df, "trapped_enemy", window,
                         "Times trapped enemies / episode", "Times trapped enemies",
                         os.path.join(out_dir, "compare_times_trapped_enemies.png"))

    _plot_rolling_metric(df, "trapped_self", window,
                         "Times trapped self / episode", "Times trapped self",
                         os.path.join(out_dir, "compare_times_trapped_self.png"))

    _plot_rolling_metric(df, "bomb_masked", window,
                         "Bomb masked / episode", "Bombs masked",
                         os.path.join(out_dir, "bombs_masked.png"))

    _plot_rolling_metric(df, "bomb_legal_not_taken", window,
                         "Legal Bomb masked / episode", "Legal Bombs masked",
                         os.path.join(out_dir, "legal_bombs_masked.png"))

    _plot_rolling_metric(df, "useful_rate", window,
                         "Useful bomb rate (%)", "Bombing precision rate",
                         os.path.join(out_dir, "compare_useful_rate.png"))

    # Survival / outcome metrics
    _plot_rolling_rate(df, "got_killed", window,
                       "Overall death rate (%)", "Overall death rate",
                       os.path.join(out_dir, "compare_death_rate.png"))

    _plot_rolling_rate(df, "killed_self", window,
                       "Self-kill rate (%)", "Self-kill rate",
                       os.path.join(out_dir, "compare_self_kill_rate.png"))

    _plot_rolling_rate(df, "killed_opponent", window,
                       "Opponent kill rate (%)", "Opponent kill rate",
                       os.path.join(out_dir, "compare_killed_opponent_rate.png"))

    # NB: the episode CSV's "opponent_killed" column is what the LUT branch used to call
    # "opponent_eliminated" (see train.py) - it already covers any opponent elimination,
    # including mutual kills, so there is no separate plot for the old name.
    _plot_rolling_rate(df, "opponent_killed", window,
                       "Opponent-death rate (%)", "Opponent death rate",
                       os.path.join(out_dir, "compare_opponent_death_rate.png"))

    # Terminal-action breakdown
    df = df.copy()
    df["terminal_wait"] = (df["terminal_action"] == "WAIT").astype(int)

    _plot_rolling_rate(df, "terminal_wait", window, "Deaths ending in WAIT (%)",
                        "Terminal action = WAIT, among all episodes",
                        os.path.join(out_dir, "compare_terminal_wait.png"))

    # Convergence diagnostic (critic_value_pred = max_a Q(s,a) for LUT)
    _plot_rolling_metric(df, "value_error", window,
                        "|max_a Q(s,a) - Shaped Reward| (terminal)",
                        "Terminal value error",
                        os.path.join(out_dir, "compare_value_error.png"))

    # Optional: reward trend, per-episode (LUT has no per-update batching)
    _plot_rolling_metric(df, "shaped_reward", window,
                        "Shaped reward / episode", "Reward trend",
                        os.path.join(out_dir, "compare_shaped_reward.png"))

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