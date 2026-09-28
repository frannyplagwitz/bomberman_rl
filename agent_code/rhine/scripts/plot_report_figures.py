"""Report figures built only from saved records.

  N1 task4_training_progress.png : Task 4 periodic-evaluation curves per seed
                                   (async evaluator CSVs), selected checkpoints marked.
  N3 task4_vs_stage_d_baseline.png: Stage D checkpoints (untrained on rule_based_agent)
                                   vs Task 4 checkpoints, breaker on/off, eval_seed=1000.
  N5 task4_vs_rule_based.pdf     : per-game final scores and RHINE's placement against
                                   the three rule_based_agent opponents, eval_seed=1000.

Usage:
  python -m agent_code.rhine.scripts.plot_report_figures [--out-dir figures]
"""
import argparse
import csv
import gzip
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
LOGS = REPO / "agent_code" / "rhine" / "logs"
RERUN_DIR = LOGS / "eval_rerun_20260924"
SEEDS = (0, 1, 2)
# Selected rounds from the offline checkpoint selection (select_task4_checkpoint.py output).
SELECTED_ROUND = {0: 900, 1: 1100, 2: 1100}

SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SEED_MARKERS = ["o", "s", "^"]
CONDITION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
RANK_RAMP = ["#184f95", "#3987e5", "#86b6ef", "#b7d3f6"]
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e0"
OPPONENT_GRAY = "#9a9a9a"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def plot_training_progress(out: Path):
    panels = [("score_mean", "Score per game"), ("self_kill_rate", "Self-kill rate"),
              ("got_killed_by_opponent_rate", "Killed-by-opponent rate")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for n in SEEDS:
        with open(LOGS / f"task4_async_eval_seed{n}.csv") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["round"]))
        rounds = np.array([int(r["round"]) for r in rows])
        for ax, (key, _) in zip(axes, panels):
            vals = np.array([float(r[key]) for r in rows])
            ax.plot(rounds, vals, color=SEED_COLORS[n], marker=SEED_MARKERS[n], markersize=4,
                    linewidth=1.6, label=f"seed {n}")
            sel = rounds == SELECTED_ROUND[n]
            ax.plot(rounds[sel], vals[sel], linestyle="none", marker=SEED_MARKERS[n], markersize=10,
                    markerfacecolor="none", markeredgecolor=SEED_COLORS[n], markeredgewidth=1.6)
    for ax, (_, title) in zip(axes, panels):
        _style(ax)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("Training round", fontsize=8, color=INK)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_baseline_comparison(out: Path):
    conditions = [("Stage D\nbreaker on", "stage_d_seed{n}_vs_rulebased_e1000_A"),
                  ("Task 4\nbreaker off", "task4_seed{n}_no_breaker_e1000_B"),
                  ("Task 4\nbreaker on", "task4_seed{n}_breaker_only_e1000_A")]
    panels = [("score_mean", "Score per game"), ("self_kill_rate", "Self-kill rate"),
              ("got_killed_by_opponent_rate", "Killed-by-opponent rate"),
              ("oscillation_fraction", "Oscillation fraction")]
    data = {label: [_load(RERUN_DIR / f"{pat.format(n=n)}.pkl.gz")["agg"] for n in SEEDS]
            for label, pat in conditions}
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    x = np.arange(len(conditions))
    for ax, (key, title) in zip(axes, panels):
        for i, (label, _) in enumerate(conditions):
            vals = np.array([agg[key] for agg in data[label]])
            ax.bar(i, vals.mean(), width=0.6, color=CONDITION_COLORS[i], edgecolor="white", linewidth=2)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), color=INK, capsize=3, linewidth=1)
            for n, v in zip(SEEDS, vals):
                ax.plot(i + (n - 1) * 0.12, v, linestyle="none", marker=SEED_MARKERS[n], markersize=4,
                        markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8)
        _style(ax)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(x, [c[0] for c in conditions], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_vs_rule_based(out: Path):
    """Sized for 0.8 textwidth on A4 with 2.5 cm margins so text renders near 9 pt; saved as vector PDF."""
    runs = [_load(RERUN_DIR / f"task4_seed{n}_breaker_only_e1000_A.pkl.gz") for n in SEEDS]
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))

        ax = axes[0]
        positions, box_data, colors = [], [], []
        for n, run in enumerate(runs):
            scores = np.array([[s for _, _, s in rec["final_scores"]] for rec in run["records"]], dtype=float)
            box_data += [scores[:, 0], scores[:, 1:].ravel()]
            positions += [n * 3, n * 3 + 1]
            colors += [SEED_COLORS[0], OPPONENT_GRAY]
        bp = ax.boxplot(box_data, positions=positions, widths=0.75, patch_artist=True, showmeans=True,
                        medianprops=dict(color=INK, linewidth=1),
                        whiskerprops=dict(color=MUTED, linewidth=0.8), capprops=dict(color=MUTED, linewidth=0.8),
                        meanprops=dict(marker="D", markersize=3.5, markerfacecolor="white", markeredgecolor=INK,
                                       markeredgewidth=0.7),
                        flierprops=dict(marker=".", markersize=2.5, markeredgecolor=MUTED))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor("white")
        ax.set_xticks([n * 3 + 0.5 for n in SEEDS], [f"seed {n}" for n in SEEDS])
        ax.set_ylabel("Final score per game")
        ax.legend([bp["boxes"][0], bp["boxes"][1]], ["RHINE", "rule-based"], fontsize=8, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=1.0)

        ax = axes[1]
        labels = ["1st (sole)", "1st (tied)", "2nd", "3rd/4th"]
        bottoms = np.zeros(len(SEEDS))
        fractions = []
        for run in runs:
            scores = np.array([[s for _, _, s in rec["final_scores"]] for rec in run["records"]], dtype=float)
            rhine, others = scores[:, 0], scores[:, 1:]
            n_higher = (others > rhine[:, None]).sum(axis=1)
            n_equal = (others == rhine[:, None]).sum(axis=1)
            sole_first = (n_higher == 0) & (n_equal == 0)
            tied_first = (n_higher == 0) & (n_equal > 0)
            second = n_higher == 1
            lower = n_higher >= 2
            fractions.append([sole_first.mean(), tied_first.mean(), second.mean(), lower.mean()])
        fractions = np.array(fractions)
        for k, label in enumerate(labels):
            ax.bar(np.arange(len(SEEDS)), fractions[:, k], bottom=bottoms, width=0.65, color=RANK_RAMP[k],
                   edgecolor="white", linewidth=1, label=label)
            bottoms += fractions[:, k]
        ax.set_xticks(np.arange(len(SEEDS)), [f"seed {n}" for n in SEEDS])
        ax.set_ylabel("Fraction of games")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  handlelength=1.0, columnspacing=1.0, labelspacing=0.2)

        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5)
            ax.tick_params(axis="x", length=0)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        fig.savefig(out)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_progress(out_dir / "task4_training_progress.png")
    plot_baseline_comparison(out_dir / "task4_vs_stage_d_baseline.png")
    plot_vs_rule_based(out_dir / "task4_vs_rule_based.pdf")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
