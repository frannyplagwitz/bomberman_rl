"""Task 4 offline investigation: oscillation deep dive. Read-only on
task4_seed{0,1,2}_{no_breaker_B,breaker_only_A}.pkl.gz.

Windows come from diagnose_oscillation.longest_oscillation_run(); the
exclusion filter reuses analyze_stage_d_final._osc_cases()'s crates-left
check.

Definitions:
- "opponent moving in the window": some opponent's recorded position
  changes within the window.
- "#29 not saturated": features[:, 28] < 1.0 (an opponent within the
  distance-feature range), as a fraction of window steps.
- "top-2 action probability gap": largest minus second-largest value of the
  recorded probs_final.
- category: "features_frozen" (period2_identical_features >= 0.9),
  "features_vary_same_preference" (otherwise, but the argmax pattern repeats
  with period two on >= 90% of the window) or "other".
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.eval_stage_d_final import restore_state

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed, label, suffix):
    with gzip.open(DATA_DIR / f"task4_seed{seed}_{label}_{suffix}.pkl.gz", "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError("partial recording")
    return payload


def _case_windows(payload):
    """(round_record, window_steps) for every sustained-oscillation round."""
    out_list = []
    for r in payload["records"]:
        actions = r["metrics"]["actions"]
        run_len, start = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue
        window = r["steps"][start:start + run_len + 1]
        out_list.append((r, window, start, run_len))
    return out_list


def _analyze_window(window):
    feats = np.stack([w["features"] for w in window])
    period2_same = (
        float(np.mean([np.array_equal(feats[i], feats[i + 2]) for i in range(len(feats) - 2)]))
        if len(feats) > 2 else float("nan")
    )
    opp_positions_by_step = [
        sorted(tuple(int(v) for v in o[3]) for o in w["state"]["others"]) for w in window
    ]
    opponent_moving = any(opp_positions_by_step[i] != opp_positions_by_step[0] for i in range(len(opp_positions_by_step)))
    f29_not_saturated_frac = float(np.mean(feats[:, 28] < 1.0))
    top2_gaps = []
    argmax_actions = []
    for w in window:
        p = w["probs_final"]
        order = np.argsort(-p)
        top2_gaps.append(float(p[order[0]] - p[order[1]]) if len(order) > 1 else float("nan"))
        argmax_actions.append(cfg.ACTIONS[order[0]])
    # Period-2 test on the argmax action: the argmax itself alternates in a
    # genuine two-tile oscillation, so comparing to the first action would fail.
    argmax_period2_frac = (
        float(np.mean([argmax_actions[i] == argmax_actions[i + 2] for i in range(len(argmax_actions) - 2)]))
        if len(argmax_actions) > 2 else float("nan")
    )
    if period2_same >= 0.9:
        category = "features_frozen"
    elif argmax_period2_frac >= 0.9:
        category = "features_vary_same_preference"
    else:
        category = "other"
    return {
        "period2_identical_features": period2_same, "opponent_moving": opponent_moving,
        "f29_not_saturated_frac": f29_not_saturated_frac, "mean_top2_gap": float(np.mean(top2_gaps)),
        "argmax_period2_frac": argmax_period2_frac, "category": category,
    }


def _crates_left_at(window):
    return int(np.sum(restore_state(window[0]["state"])["field"] == 1))


def _coins_at(window):
    return len(window[0]["state"]["coins"])


def run_group(label, suffix, title, exclude_late_game):
    out(f"\n--- {title} ---")
    cat_counts = {}
    n_total = 0
    n_excluded = 0
    rows = []
    for s in SEEDS:
        payload = load(s, label, suffix)
        for r, window, start, run_len in _case_windows(payload):
            n_total += 1
            crates_left = _crates_left_at(window)
            coins_at_start = _coins_at(window)
            late_game = crates_left == 0 and coins_at_start == 0
            if exclude_late_game and late_game:
                n_excluded += 1
                continue
            analysis = _analyze_window(window)
            cat_counts[analysis["category"]] = cat_counts.get(analysis["category"], 0) + 1
            rows.append((s, r["round"], run_len, crates_left, coins_at_start, analysis))

    out(f"n_total_sustained_cases={n_total}, excluded(late_game_no_score)={n_excluded if exclude_late_game else 'n/a'}, "
        f"analyzed={len(rows)}")
    if not rows:
        out("  n=0 after exclusion -- data insufficient to characterize this subset.")
        return
    for s, rnd, run_len, crates_left, coins_at_start, a in rows:
        out(f"  seed{s} round={rnd:>3} run_len={run_len:>3} crates_left={crates_left} coins={coins_at_start} "
            f"period2_identical={a['period2_identical_features']:.2f} opponent_moving={a['opponent_moving']} "
            f"f29_visible_frac={a['f29_not_saturated_frac']:.2f} mean_top2_prob_gap={a['mean_top2_gap']:.3f} "
            f"argmax_period2_frac={a['argmax_period2_frac']:.2f} category={a['category']}")
    out(f"\n  category totals (n={len(rows)}): {cat_counts}")
    if len(rows) < 30:
        out("  NOTE: n<30, small sample, for reference only.")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation2_oscillation_deepdive.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== Item 3: oscillation deep dive ===")
    out("Reuses: diagnose_oscillation.longest_oscillation_run()/OSCILLATION_THRESHOLD for window "
        "detection; period2_identical_features convention from analyze_stage_d_final._osc_cases(). "
        "New: opponent_moving / f29_visible_frac / top2_prob_gap / category classification (see module "
        "docstring for exact definitions).")

    run_group("no_breaker", "B", "no_breaker group, excluding late_game_no_score", exclude_late_game=True)
    run_group("breaker_only", "A", "breaker_only group (4 early long-run cases)", exclude_late_game=False)

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
