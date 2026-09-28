"""Summarizes the reproducible Task 4 / Stage D-vs-rule_based evaluation
rerun (eval_stage_d_final.py --reproducible) and compares it with the
earlier, non-reproducible Task 4 records.

Per group: 3-seed mean +/- std (ddof=1) of the headline metrics. Per run:
each agent's mean final score, RHINE's strict-first / shared-first rates and
mean rank (tied agents share the average of their rank positions).

Usage:
  python -m agent_code.rhine.scripts.analyze_eval_rerun [--rerun-dir DIR] [--old-dir DIR]
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parents[1]
RERUN_DIR = AGENT_DIR / "logs" / "eval_rerun_20260924"
OLD_DIR = AGENT_DIR / "logs" / "stage_d_final" / "task4"

GROUPS = {
    "task4_breaker_only_e1000": "task4_seed{n}_breaker_only_e1000_A",
    "task4_breaker_only_e2000": "task4_seed{n}_breaker_only_e2000_A",
    "task4_no_breaker_e1000": "task4_seed{n}_no_breaker_e1000_B",
    "task4_deadlock_e1000": "task4_seed{n}_deadlock_e1000_A",
    "stage_d_vs_rulebased_e1000": "stage_d_seed{n}_vs_rulebased_e1000_A",
}
OLD_GROUPS = {
    "task4_breaker_only_e1000": "task4_seed{n}_breaker_only_A",
    "task4_breaker_only_e2000": "task4_seed{n}_xval2000_A",
    "task4_no_breaker_e1000": "task4_seed{n}_no_breaker_B",
    "task4_deadlock_e1000": "task4_seed{n}_deadlock_A",
}
METRICS = ["score_mean", "self_kill_rate", "got_killed_by_opponent_rate", "oscillation_fraction",
           "completion_rate", "opponent_kills_mean"]
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def rank_stats(records: list) -> dict:
    """Per-agent mean score and RHINE's placement among all four agents."""
    names = [name for name, _, _ in records[0]["final_scores"]]
    scores = np.array([[score for _, _, score in rec["final_scores"]] for rec in records], dtype=float)
    rhine = scores[:, 0]
    others = scores[:, 1:]
    strict_first = rhine > others.max(axis=1)
    shared_first = rhine >= others.max(axis=1)
    n_higher = (others > rhine[:, None]).sum(axis=1)
    n_equal = (others == rhine[:, None]).sum(axis=1)
    mean_rank = n_higher + 1 + n_equal / 2.0
    return {
        "names": names,
        "mean_scores": scores.mean(axis=0),
        "opponent_mean": others.mean(),
        "best_opponent_mean": others.max(axis=1).mean(),
        "strict_first_rate": strict_first.mean(),
        "shared_first_rate": shared_first.mean(),
        "mean_rank": mean_rank.mean(),
        "margin_vs_best_opponent": (rhine - others.max(axis=1)).mean(),
    }


def mean_std(values) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.3f}+/-{arr.std(ddof=1):.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-dir", default=str(RERUN_DIR))
    parser.add_argument("--old-dir", default=str(OLD_DIR))
    args = parser.parse_args()
    rerun_dir, old_dir = Path(args.rerun_dir), Path(args.old_dir)

    for group, pattern in GROUPS.items():
        runs = [load(rerun_dir / f"{pattern.format(n=n)}.pkl.gz") for n in SEEDS]
        for n, run in zip(SEEDS, runs):
            meta = run["meta"]
            if not (meta["complete"] and meta.get("reproducible") and meta["disable_think_time_limit"]):
                sys.exit(f"{group} seed{n}: unexpected meta {meta}")
            if len(run["records"]) != 100:
                sys.exit(f"{group} seed{n}: {len(run['records'])} rounds")
        print(f"\n=== {group} (n=100 per seed) ===")
        for m in METRICS:
            vals = [run["agg"][m] for run in runs]
            print(f"  {m:30s} per-seed={[round(v, 3) for v in vals]}  mean+/-std={mean_std(vals)}")
        own = [run["agg"]["invalid_action_own_cause_count_total"] for run in runs]
        print(f"  invalid_action_own_cause_count_total per-seed={own}")
        stats = [rank_stats(run["records"]) for run in runs]
        for n, st in zip(SEEDS, stats):
            per_agent = ", ".join(f"{name}={s:.2f}" for name, s in zip(st["names"], st["mean_scores"]))
            print(f"  seed{n}: {per_agent} | strict_first={st['strict_first_rate']:.2f} "
                  f"shared_first={st['shared_first_rate']:.2f} mean_rank={st['mean_rank']:.2f} "
                  f"margin_vs_best_opp={st['margin_vs_best_opponent']:.2f}")
        for key in ("opponent_mean", "best_opponent_mean", "strict_first_rate", "shared_first_rate",
                    "mean_rank", "margin_vs_best_opponent"):
            print(f"  {key:30s} mean+/-std={mean_std([st[key] for st in stats])}")

        if group in OLD_GROUPS:
            old_paths = [old_dir / f"{OLD_GROUPS[group].format(n=n)}.pkl.gz" for n in SEEDS]
            if all(p.exists() for p in old_paths):
                old = [load(p) for p in old_paths]
                print("  vs old (non-reproducible) records:")
                for m in METRICS:
                    new_vals = [run["agg"][m] for run in runs]
                    old_vals = [o["agg"][m] for o in old]
                    print(f"    {m:28s} old={mean_std(old_vals)} new={mean_std(new_vals)} "
                          f"per-seed diff={[round(a - b, 3) for a, b in zip(new_vals, old_vals)]}")
                for n, (o, r) in enumerate(zip(old, runs)):
                    so = np.array([rec["metrics"]["coins_collected"] + 5 * rec["metrics"]["opponent_kills"]
                                   for rec in o["records"]], dtype=float)
                    sn = np.array([rec["final_scores"][0][2] for rec in r["records"]], dtype=float)
                    se = np.sqrt(so.var(ddof=1) / len(so) + sn.var(ddof=1) / len(sn))
                    print(f"    seed{n} score diff new-old={sn.mean() - so.mean():+.2f} (SE of diff {se:.2f})")


if __name__ == "__main__":
    main()
