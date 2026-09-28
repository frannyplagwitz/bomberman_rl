"""Task 1 reward-configuration comparison figure (report version).

Re-plots the periodic training evaluations of the three Task 1 reward
configurations (completion rate and mean training reward) as one row of two
subplots with a shared legend on top,
sized for a 16 cm text width so it can be included at 100% scale. Writes a
PNG and a vector PDF.

Also verifies the result: re-renders the original figure with the code of
common.plot_comparison() and compares it pixel by pixel with the saved
original, then checks that every plotted point, the axis limits and the line
colors of the new figure match that rendering, and that no text overlaps a
subplot or leaves the figure.

Usage (from the repository root or anywhere else):
  python agent_code/rhine/scripts/plot_task1_comparison.py [--out PATH.png]
"""
import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "agent_code" / "rhine" / "logs"
RUNS = {k: LOGS / f"stage_a_{k}_run_20260827_163549.csv" for k in ("A1", "A2", "A3")}
ORIGINAL_PNG = LOGS / "stage_a_comparison_20260827_163549.png"
DEFAULT_OUT = ROOT / "figures" / "task1_reward_comparison.png"

X = "total_steps"
ORIGINAL_Y = ["completion_rate", "steps_per_coin_mean", "train_reward_mean"]
REPORT_Y = ["completion_rate", "train_reward_mean"]
ORIGINAL_TITLE = "Stage A ablation comparison (A1 sparse / A2 +step_cost / A3 +distance_shaping)"
LEGEND = {"A1": "1 (sparse)", "A2": "2 (step cost)", "A3": "3 (distance shaping)"}
YLABEL = {"completion_rate": "Completion rate", "train_reward_mean": "Mean training reward"}
XLABEL = "Training steps"
Y_TICKS = {"completion_rate": [0.4, 0.6, 0.8, 1.0], "train_reward_mean": [0.15, 0.25, 0.35]}

# Print size: \textwidth = 16 cm, included without scaling.
FIGSIZE_IN = (6.3, 2.4)
REPORT_RC = {
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable
}
X_TICK_STEP = 100_000
PNG_DPI = 300


def read_rows(path: Path) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def series(path: Path, y_col: str):
    rows = read_rows(path)
    x = [float(r[X]) for r in rows]
    y = [float(r[y_col]) if r[y_col] not in ("", "nan") else float("nan") for r in rows]
    return x, y


def plot_original(png: Path):
    """Reproduces common.plot_comparison() exactly (stacked subplots, original
    title, raw column names, one legend per subplot). Returns (fig, axes)."""
    fig, axes = plt.subplots(len(ORIGINAL_Y), 1, figsize=(8, 3 * len(ORIGINAL_Y)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for key, path in RUNS.items():
        for ax, y_col in zip(axes, ORIGINAL_Y):
            ax.plot(*series(path, y_col), marker="o", markersize=3, label=key)
    for ax, y_col in zip(axes, ORIGINAL_Y):
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(X)
    fig.suptitle(ORIGINAL_TITLE)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    return fig, axes


def plot_report(png: Path, reference_axes: dict):
    """Draws the one-row report version and saves it as `png` and as a PDF
    next to it. Axis limits are copied from `reference_axes`, the original
    rendering's axes keyed by column name. Returns (fig, axes)."""
    with plt.rc_context(REPORT_RC):
        fig, axes = plt.subplots(1, len(REPORT_Y), figsize=FIGSIZE_IN, layout="constrained")
        for key, path in RUNS.items():
            for ax, y_col in zip(axes, REPORT_Y):
                ax.plot(*series(path, y_col), linewidth=1.2, marker="o", markersize=2.25,
                        markeredgewidth=0, label=LEGEND[key])

        k_format = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
        for ax, y_col in zip(axes, REPORT_Y):
            ref = reference_axes[y_col]
            ax.set_ylabel(YLABEL[y_col])
            ax.grid(True, alpha=0.3)
            ax.set_yticks(Y_TICKS[y_col])
            ax.set_ylim(ref.get_ylim())
            ax.set_xlim(ref.get_xlim())
            ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
            ax.xaxis.set_major_formatter(k_format)
        fig.supxlabel(XLABEL, fontsize=REPORT_RC["axes.labelsize"])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
        fig.savefig(png, dpi=PNG_DPI)
        fig.savefig(png.with_suffix(".pdf"))
    return fig, axes


def layout_problems(fig, axes) -> list:
    """Lists text/area collisions in the rendered figure: legend or shared
    x-label overlapping a subplot (incl. its tick and axis labels), subplots
    overlapping each other, tick labels overlapping within an axis, and
    anything extending past the figure edge. Empty list means clean."""
    r = fig.canvas.get_renderer()
    fig_box = fig.bbox
    boxes = {f"subplot {i + 1}": ax.get_tightbbox(r) for i, ax in enumerate(axes)}
    boxes["legend"] = fig.legends[0].get_window_extent(r)
    boxes["x-label"] = fig._supxlabel.get_window_extent(r)

    problems = []
    for name, b in boxes.items():
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig_box.width or b.y1 > fig_box.height:
            problems.append(f"{name} extends past the figure edge")
    for (n1, b1), (n2, b2) in itertools.combinations(boxes.items(), 2):
        if b1.overlaps(b2):
            problems.append(f"{n1} overlaps {n2}")
    for i, ax in enumerate(axes):
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_window_extent(r) for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            for b1, b2 in itertools.combinations(labels, 2):
                if b1.overlaps(b2):
                    problems.append(f"subplot {i + 1}: overlapping {axis.axis_name}-tick labels")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="PNG path; the PDF is written next to it.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        repro_png = Path(tmp) / "original_repro.png"
        _, orig_axes = plot_original(repro_png)
        from PIL import Image
        a = np.asarray(Image.open(ORIGINAL_PNG).convert("RGB"))
        b = np.asarray(Image.open(repro_png).convert("RGB"))
        repro_ok = a.shape == b.shape and np.array_equal(a, b)
        print(f"original figure reproduced pixel-identically: {repro_ok}")

    orig_by_col = dict(zip(ORIGINAL_Y, orig_axes))
    fig, axes = plot_report(args.out, orig_by_col)

    data_ok = True
    for i, y_col in enumerate(REPORT_Y):
        for j, (key, path) in enumerate(RUNS.items()):
            ref = np.array([[float(r[X]), float(r[y_col])] for r in read_rows(path)])
            new = axes[i].lines[j].get_xydata()
            same = (np.array_equal(new, ref, equal_nan=True)
                    and np.array_equal(new, orig_by_col[y_col].lines[j].get_xydata(), equal_nan=True))
            data_ok &= same
            print(f"  {y_col:20s} {key}: n={len(ref)} identical={same}")
    limits_ok = all(o.get_xlim() == n.get_xlim() and o.get_ylim() == n.get_ylim()
                    for o, n in zip((orig_by_col[c] for c in REPORT_Y), axes))
    colors_ok = all([l.get_color() for l in o.lines] == [l.get_color() for l in n.lines]
                    for o, n in zip((orig_by_col[c] for c in REPORT_Y), axes))
    yticks_ok = all(list(ax.get_yticks()) == Y_TICKS[y_col] for ax, y_col in zip(axes, REPORT_Y))
    problems = layout_problems(fig, axes)
    print(f"data identical: {data_ok} | axis limits identical: {limits_ok} | "
          f"colors identical: {colors_ok} | y-ticks as specified: {yticks_ok}")
    print("layout: " + ("no overlaps, nothing outside the figure" if not problems else "; ".join(problems)))
    print(f"saved {args.out} and {args.out.with_suffix('.pdf')}")
    if not (repro_ok and data_ok and limits_ok and colors_ok and yticks_ok) or problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
