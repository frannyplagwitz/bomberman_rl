"""Task 4 offline investigation: opportunity cost of death in real game score
(settings.py REWARD_COIN/REWARD_KILL), not training reward. Read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

The per-step score is game_state["self"][1], the framework's cumulative
score, cross-checked against EpisodeMetrics.score.

Definitions:
- future_gain(t) = final_score - score(t), over survived rounds only.
- estimated lost score of a death = mean future_gain of survived rounds in
  the same 50-step bucket as the death step.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import settings as s
from agent_code.rhine import config as cfg

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
EXCLUDED_15_1 = {(0, r) for r in (21, 26, 27, 32, 36, 60, 74, 75, 87)}
BUCKET_EDGES = list(range(0, 401, 50))


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed):
    with gzip.open(DATA_DIR / f"task4_seed{seed}_breaker_only_A.pkl.gz", "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError("partial recording")
    return payload


def _bucket(step_num):
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= step_num < BUCKET_EDGES[i + 1]:
            return f"{BUCKET_EDGES[i]}-{BUCKET_EDGES[i + 1]}"
    return f"{BUCKET_EDGES[-1]}+"


def item4a():
    out("=== Item 4a: score reconstruction check ===")
    out(f"settings.py: REWARD_COIN={s.REWARD_COIN} REWARD_KILL={s.REWARD_KILL}")
    n_checked = 0
    n_mismatch = 0
    for seed in SEEDS:
        payload = load(seed)
        for r in payload["records"]:
            final_score_from_self = int(r["steps"][-1]["state"]["self"][1])
            # The recorded score precedes the last action's resolution, so it
            # can under-count what that action earned.
            m = r["metrics"]
            final_score_from_metrics = m["coins_collected"] * s.REWARD_COIN + m["opponent_kills"] * s.REWARD_KILL
            n_checked += 1
            if abs(final_score_from_self - final_score_from_metrics) > s.REWARD_KILL:
                # Slack of one kill plus one coin for the unresolved final action.
                n_mismatch += 1
    out(f"cross-check: self[1] (live per-step score) vs EpisodeMetrics.score (coins*"
        f"{s.REWARD_COIN}+kills*{s.REWARD_KILL}) over {n_checked} rounds: "
        f"{n_mismatch} case(s) differing by more than one event's worth -- "
        f"{'consistent (as expected, only the unresolved-final-action gap)' if n_mismatch == 0 else 'unexpected mismatches, see raw data'}.")
    out("Method used below: self[1] for the PER-STEP score progression (item 4b/4c), "
        "EpisodeMetrics.score (coins_collected*REWARD_COIN+opponent_kills*REWARD_KILL) for each "
        "round's FINAL score (matches common.py's own EpisodeMetrics.score property, used everywhere "
        "else in this project).")


def _final_score(r):
    m = r["metrics"]
    return m["coins_collected"] * s.REWARD_COIN + m["opponent_kills"] * s.REWARD_KILL


def item4b():
    out("\n=== Item 4b: future_gain(t) by step bucket, survived rounds only ===")
    bucket_gains = {}
    n_survived = 0
    for seed in SEEDS:
        payload = load(seed)
        for r in payload["records"]:
            if r["metrics"]["self_kill"] or r["metrics"]["got_killed_by_opponent"]:
                continue
            n_survived += 1
            final_score = _final_score(r)
            for x in r["steps"]:
                step_num = x["state"]["step"]
                score_t = int(x["state"]["self"][1])
                gain = final_score - score_t
                bucket_gains.setdefault(_bucket(step_num), []).append(gain)
    out(f"survived rounds: {n_survived}/300")
    out(f"{'bucket':>10} {'n':>8} {'mean':>8} {'median':>8}")
    bucket_summary = {}
    for i in range(len(BUCKET_EDGES) - 1):
        b = f"{BUCKET_EDGES[i]}-{BUCKET_EDGES[i + 1]}"
        vals = bucket_gains.get(b, [])
        mean_v = float(np.mean(vals)) if vals else float("nan")
        median_v = float(np.median(vals)) if vals else float("nan")
        bucket_summary[b] = mean_v
        note = "  (n<30, small sample)" if len(vals) < 30 else ""
        out(f"{b:>10} {len(vals):>8} {mean_v:>8.3f} {median_v:>8.3f}{note}")
    return bucket_summary


def item4c(bucket_summary):
    out("\n=== Item 4c: estimated lost score at death, by cause ===")
    for kind_label, pred in (("self-kill", lambda m: m["self_kill"]), ("got_killed", lambda m: m["got_killed_by_opponent"])):
        losses = []
        per_case = []
        for seed in SEEDS:
            payload = load(seed)
            for r in payload["records"]:
                if not pred(r["metrics"]) or not r["death"]:
                    continue
                death_step = r["death"]["step"]
                b = _bucket(death_step)
                loss = bucket_summary.get(b, float("nan"))
                losses.append(loss)
                per_case.append((seed, r["round"], death_step, b, loss))
        out(f"\n{kind_label}: n={len(losses)}")
        if len(losses) < 30:
            out("  NOTE: n<30, small sample, for reference only.")
        valid = [l for l in losses if not np.isnan(l)]
        out(f"  mean estimated lost score={np.mean(valid) if valid else float('nan'):.3f} "
            f"median={np.median(valid) if valid else float('nan'):.3f}")
        from collections import Counter
        bucket_dist = Counter(b for _, _, _, b, _ in per_case)
        out(f"  death-step bucket distribution: {dict(bucket_dist)}")

        if kind_label == "self-kill":
            fifteen_one = [c for c in per_case if c[0] == 0 and c[1] in {21, 26, 27, 32, 36, 60, 74, 75, 87}]
            out(f"\n  (15,1) nine cases specifically (n={len(fifteen_one)}):")
            for seed, rnd, step, b, loss in fifteen_one:
                out(f"    seed{seed} round={rnd} death_step={step} bucket={b} estimated_lost_score={loss:.3f}")
            vals = [c[4] for c in fifteen_one]
            out(f"    mean={np.mean(vals):.3f} (n={len(vals)}, small sample, for reference only)")


def item4d():
    out("\n=== Item 4d: comparison to training penalty constants (numbers only) ===")
    rc = cfg.REWARD_CONFIG
    out(f"  SELF_KILL_PENALTY = {rc.SELF_KILL_PENALTY}")
    out(f"  TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY = {rc.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY}")
    out(f"  TRAINING_KILLED_OPPONENT_REWARD = {rc.TRAINING_KILLED_OPPONENT_REWARD} (context only, not a death penalty)")
    out("  (see item 4c above for the estimated real-score opportunity cost to compare these against)")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation3_death_opportunity_cost.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item4a()
    bucket_summary = item4b()
    item4c(bucket_summary)
    item4d()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
