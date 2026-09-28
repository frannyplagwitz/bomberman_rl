"""Compare per-episode metrics across multiple training runs.

Loads one or more CSVs written by episode_logger.EpisodeCSVLogger and
plots rolling-average comparisons across runs

Two ways to call 
1. Groups of runs: each --group is one configuration
   (e.g. CNN vs MLP) made of several independent runs. Each group is drawn as the
   mean over its runs, with a shaded band showing the spread between runs.
 
    python plot_comparison.py \
        --group CNN "results/task2_v2/run_*/logs/episodes-*.csv" \
        --group MLP "results/task2_mlp/run_*/logs/episodes-*.csv" \
        --window 50 --out plots/task2_cnn_vs_mlp

2. Individual files: 
    python plot_comparison.py logs/episodes-run1.csv logs/episodes-run2.csv \
        --labels "wasteful=-0.4" "wasteful=-0.3" --window 15 --out plots/comparison
    
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd



def newest_csv_per_run(paths):
    """Keep one CSV per run folder (newest), warning about any extras.
       This is run in case where multiple training runs were run
 
    Episode CSVs are stored in <run_dir>/logs/, and a restarted run leaves a second
    timestamped CSV behind. 
    """
    
    # Gather all directories related to a particular run 
    by_run = {}
    for path in paths:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        by_run.setdefault(run_dir, []).append(path)
 
    chosen = []
    for run_dir, files in sorted(by_run.items()):
        newest = max(files, key=os.path.getmtime)
        if len(files) > 1:
            print(f"[plot] {run_dir} has {len(files)} episode CSVs, using the newest: "
                  f"{os.path.basename(newest)}")
        chosen.append(newest)
    return chosen
 
 
def load_groups(group_specs):
    """group_specs: list of (label, pattern). Returns {label: [DataFrame per run]}."""
    groups = {}
    for label, pattern in group_specs:
        paths = newest_csv_per_run(sorted(glob.glob(pattern)))
        if not paths:
            raise FileNotFoundError(f"Group {label!r}: no files match {pattern!r}")
        runs = [pd.read_csv(p).sort_values("episodes").reset_index(drop=True) for p in paths]
        groups.setdefault(label, []).extend(runs)
        print(f"[plot] group {label!r}: {len(runs)} run(s)")
    return groups

def load_files(paths, labels=None):
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"Got {len(paths)} files but {len(labels)} labels - must match 1:1")
    return {label: [pd.read_csv(path).sort_values("episodes").reset_index(drop=True)]
            for path, label in zip(paths, labels)}
 

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


# ---------------------------------------------------------------------------
# Per-run series
# ---------------------------------------------------------------------------
 
def episode_series(run: pd.DataFrame, column: str, window: int, as_rate: bool) -> pd.Series:
    """Rolling average of a per-episode column, indexed by episode number."""
    values = run[column].astype(float)
    if as_rate:
        values = 100.0 * values
    rolled = values.rolling(window, min_periods=1).mean()
    return pd.Series(rolled.to_numpy(), index=run["episodes"].to_numpy())
 
 
def update_series(run: pd.DataFrame, column: str, episodes_per_update: int, window: int) -> pd.Series:
    """For columns that only change once per PPO update (entropy, losses, mean reward):
    one value per update (last episode of each block), rolling-averaged over updates."""
    update_idx = (run["episodes"] - 1) // episodes_per_update
    per_update = run.groupby(update_idx)[column].last().astype(float)
    rolled = per_update.rolling(window, min_periods=1).mean()
    return pd.Series(rolled.to_numpy(), index=(per_update.index + 1).to_numpy())

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
 
def plot_groups(groups, series_fn, ylabel, title, xlabel, out_path, band="std", show_runs=False):
    """Draw one line per group: the mean over its runs, with a shaded band.
 
    band: "std"    -> mean +/- one standard deviation across runs
          "minmax" -> from the lowest to the highest run
          "none"   -> mean only
    """
    plt.figure(figsize=(9, 5))
    drew_anything = False
 
    for label, runs in groups.items():
        series = []
        for run in runs:
            try:
                series.append(series_fn(run))
            except KeyError:
                continue  # column missing in this CSV (older logger version)
        if not series:
            continue
 
        # Align runs on the x-axis; only keep x values that every run reached,
        # so the mean is never taken over a changing number of runs
        table = pd.concat(series, axis=1).dropna()
        if table.empty:
            continue
 
        x = table.index.to_numpy()
        mean = table.mean(axis=1).to_numpy()
        line, = plt.plot(x, mean, label=f"{label} (n={table.shape[1]})")
        color = line.get_color()
 
        if show_runs and table.shape[1] > 1:
            for col in table.columns:
                plt.plot(x, table[col].to_numpy(), color=color, alpha=0.25, linewidth=0.8)
 
        if table.shape[1] > 1 and band != "none":
            if band == "minmax":
                low, high = table.min(axis=1).to_numpy(), table.max(axis=1).to_numpy()
            else:
                std = table.std(axis=1, ddof=1).to_numpy()
                low, high = mean - std, mean + std
            plt.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
        drew_anything = True
 
    if not drew_anything:
        plt.close()
        print(f"[plot] skipped {os.path.basename(out_path)} (column not found)")
        return
 
    band_note = {"std": ", shaded: ±1 std across runs",
                 "minmax": ", shaded: min–max across runs"}.get(band, "")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}{band_note}", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
 
 
def plot_comparison(groups, window: int, out_dir: str, episodes_per_update: int = 4,
                    update_window: int = 5, band: str = "std", show_runs: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    ep_label = f"Episode ({window}-episode rolling average)"
    up_label = f"PPO update, 1 update = {episodes_per_update} episodes ({update_window}-update rolling average)"
 
    def metric(column, ylabel, title, filename):
        plot_groups(groups, lambda r: episode_series(r, column, window, as_rate=False),
                    ylabel, title, ep_label, os.path.join(out_dir, filename), band, show_runs)
 
    def rate(column, ylabel, title, filename):
        plot_groups(groups, lambda r: episode_series(r, column, window, as_rate=True),
                    ylabel, title, ep_label, os.path.join(out_dir, filename), band, show_runs)
 
    def per_update(column, ylabel, title, filename):
        plot_groups(groups, lambda r: update_series(r, column, episodes_per_update, update_window),
                    ylabel, title, up_label, os.path.join(out_dir, filename), band, show_runs)
        
        
    # Check if the last action was WAIT
    for runs in groups.values():
        for run in runs:
            if "terminal_action" in run.columns:
                run["terminal_wait"] = (run["terminal_action"] == "WAIT").astype(int)

 
        # Task-performance metrics
    metric("coins_collected", "Coins collected / episode", "Coin collection", "compare_coins.png")
    metric("crates_destroyed", "Crates destroyed / episode", "Crate destruction", "compare_crates.png")
    metric("bombs_dropped", "Bombs dropped / episode", "Bombing volume", "compare_bombs_dropped.png")
    metric("useful_rate", "Useful bomb rate (%)", "Bombing precision", "compare_useful_rate.png")
    metric("trapped_enemy", "Times enemies trapped / episode", "Enemies trapped", "compare_trapped_enemy.png")
    metric("trapped_self", "Times self trapped / episode", "Self trapped", "compare_trapped_self.png")
    metric("bomb_masked", "Steps with BOMB masked / episode", "BOMB masked", "compare_bomb_masked.png")
    metric("bomb_legal_not_taken", "Steps with BOMB legal but not taken / episode",
           "BOMB legal but not taken", "compare_bomb_legal_not_taken.png")
    metric("steps", "Steps / episode", "Episode length", "compare_episode_length.png")
 
    # Survival / outcome metrics (0/1 per episode, shown as %)
    rate("got_killed", "Death rate (%)", "Death rate", "compare_death_rate.png")
    rate("killed_self", "Self-kill rate (%)", "Self-kill rate", "compare_self_kill_rate.png")
    rate("killed_opponent", "Opponent kill rate (%)", "Opponents killed by the agent",
         "compare_killed_opponent_rate.png")
    rate("opponent_killed", "Opponent death rate (%)", "Opponents eliminated (by anyone)",
         "compare_opponent_death_rate.png")
    rate("terminal_wait", "Episodes ending with WAIT (%)", "Terminal action = WAIT",
         "compare_terminal_wait.png")
 
    # Convergence diagnostics
    metric("value_error", "|V(s) - shaped reward| at terminal step", "Terminal value error",
           "compare_value_error.png")
    per_update("entropy", "Policy entropy", "Entropy", "compare_entropy.png")
    per_update("total_loss", "Total loss", "Total loss", "compare_total_loss.png")
    per_update("critic_loss", "Critic loss", "Critic loss", "compare_critic_loss.png")
    per_update("mean_reward", "Mean shaped reward per step", "Mean reward", "compare_mean_reward.png")
 
    print(f"Saved comparison plots to {out_dir}/")
 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="*", help="episode CSV files to compare")
    
    parser.add_argument("--labels", nargs="*", default=None,
                         help="one label per CSV, in the same order (defaults to filenames)")
    
    parser.add_argument("--group", nargs=2, action="append", metavar=("LABEL", "PATTERN"),
                        default=[],
                        help="a configuration made of several runs: label and a quoted glob "
                             "pattern for its episode CSVs; repeat for each configuration")
    
    
    parser.add_argument("--window", type=int, default=15,
                         help="rolling-average window in episodes applies to per-episode metrics only "
                              "(default=15)")
                         
    parser.add_argument("--episodes-per-update", type=int, default=4,
                            help="episodes_per_update the runs were trained with (default 4) - used "
                              "to convert entropy/total_loss/critic_loss/mean_reward from episode "
                              "rows to one point per PPO update")
    
    parser.add_argument("--update-window", type=int, default=5,
                         help="rolling-average window in PPO updates (default 5), for "
                              "entropy/total_loss/critic_loss/mean_reward only")
    
    parser.add_argument("--band", choices=["std", "minmax", "none"], default="std",
                        help="shaded band across runs of a group (default: std)")
    
    parser.add_argument("--show-runs", action="store_true",
                        help="also draw each run as a faint line")
    
    parser.add_argument("--out", default="plots/comparison", help="output directory for PNGs")
    args = parser.parse_args()
    
    # Deal with group settings
    if args.group and args.csv_paths:
        parser.error("use either --group or individual CSV paths, not both")
        
    if not args.group and not args.csv_paths:
        parser.error("give at least one --group or one CSV path")
        
    groups = load_groups(args.group) if args.group else load_files(args.csv_paths, args.labels)
    plot_comparison(groups, window=args.window, out_dir=args.out,
                    episodes_per_update=args.episodes_per_update,
                    update_window=args.update_window, band=args.band, show_runs=args.show_runs)
 
 
if __name__ == "__main__":
    main()