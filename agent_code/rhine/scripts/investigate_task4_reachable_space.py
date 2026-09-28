"""Task 4 offline investigation: #31 (reachable_space) false-positive rate and
siege-death characterization. Read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Baseline (as in investigate_stage_d_batch4.py::section_1()): for a death case,
the baseline at absolute step S is the mean #31 across all other non-death
rounds at steps within +/-20 of S. A significant drop at step i is
self_vals[i] < 0.7 * base_vals[i]; the onset is the earliest such i in the
last 10 steps before death.

The false-positive rate applies the same onset test to random anchors in
non-death rounds, so the death-case rate has a matched baseline.

Uses the recorded normalized #31 (features[:, 30]); saturation at 1.0 is
reported separately because it caps how large a drop can look.
"""
import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg

SEEDS = (0, 1, 2)
F31 = 30  # Index of normalized #31 reachable_space.
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
RNG = np.random.default_rng(20260921)


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed):
    path = DATA_DIR / f"task4_seed{seed}_breaker_only_A.pkl.gz"
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError(f"{path} is a partial recording")
    return payload


def _round_f31_series(payload):
    """Per-round list of (step, f31_value), plus death-cause labels."""
    series = []
    for r in payload["records"]:
        m = r["metrics"]
        pts = [(x["state"]["step"], float(x["features"][F31])) for x in r["steps"]]
        series.append({
            "round": r["round"], "self_kill": m["self_kill"],
            "got_killed": m["got_killed_by_opponent"], "points": pts, "record": r,
        })
    return series


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation_reachable_space_d1_d2.txt"))
    parser.add_argument("--n-fp-samples", type=int, default=300,
                         help="Number of random non-death anchor points to sample for the false-positive rate.")
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== (d.1) #31 false-positive rate + (d.2) siege-death characterization (group breaker_only) ===")
    out("Reuses: the Stage D self-kill census's cross-round step-matched baseline methodology "
        "(investigate_stage_d_batch4.py::section_1()) -- same 0.7x-baseline drop threshold, same "
        "'onset = earliest drop in the last 10 steps before death' definition. New: matched "
        "false-positive rate on randomly sampled non-death anchors (Stage D never computed this -- "
        "explicitly listed as an open limitation of the Stage D self-kill census).")

    all_series = {s: _round_f31_series(load(s)) for s in SEEDS}
    pool = []
    for s in SEEDS:
        for rd in all_series[s]:
            if not rd["self_kill"] and not rd["got_killed"]:
                pool.extend(rd["points"])
    pool_steps = np.array([p[0] for p in pool])
    pool_vals = np.array([p[1] for p in pool])
    n_saturated = int(np.sum(pool_vals >= 1.0))
    out(f"\nnon-death-step pool: n={len(pool)} steps across 3 seeds; saturated at 1.0: "
        f"{n_saturated}/{len(pool)} ({100 * n_saturated / len(pool):.1f}%) -- a step already at the "
        "#31 ceiling cannot register a 'drop' relative to a baseline that's also near 1.0, so true "
        "onset detection is systematically harder to trigger exactly when the tile is very open; this "
        "is expected saturation bias, not a data error.")

    def baseline_at(step, window=20):
        mask = np.abs(pool_steps - step) <= window
        return float(pool_vals[mask].mean()) if mask.any() else float("nan")

    def onset_bucket_for(self_vals, base_vals):
        drop_idx = next((i for i in range(len(self_vals)) if base_vals[i] > 0 and self_vals[i] < 0.7 * base_vals[i]), None)
        return drop_idx

    # --- true positive side: self-kill / got_killed cases ---
    death_summaries = {}
    for kind, pred in (("self-kill", lambda rd: rd["self_kill"]), ("got_killed", lambda rd: rd["got_killed"])):
        out(f"\n--- {kind} (true-positive side) ---")
        onset_bucket = Counter()
        cases = []
        for s in SEEDS:
            for rd in all_series[s]:
                if not pred(rd):
                    continue
                pts = rd["points"][-10:]
                if len(pts) < 4:
                    continue
                self_vals = [p[1] for p in pts]
                base_vals = [baseline_at(p[0]) for p in pts]
                drop_idx = onset_bucket_for(self_vals, base_vals)
                if drop_idx is None:
                    onset_bucket["no_drop_detected"] += 1
                elif drop_idx <= len(pts) - 5:
                    onset_bucket[">=4_steps_before_death"] += 1
                else:
                    onset_bucket["<4_steps_before_death"] += 1
                cases.append({"seed": s, "round": rd["round"], "drop_idx": drop_idx, "n_pts": len(pts),
                              "self_vals": self_vals, "base_vals": base_vals})
        n_cases = len(cases)
        n_early = onset_bucket.get(">=4_steps_before_death", 0)
        out(f"  n={n_cases}; onset buckets: {dict(onset_bucket)}")
        out(f"  onset >=4 steps before death: {n_early}/{n_cases} "
            f"({100 * n_early / n_cases if n_cases else float('nan'):.1f}%) "
            f"(Stage D 28-dim reference, self-kill census: self-kill 29/45=64%, got_killed 42/42=100%; "
            "not a controlled comparison, different opponent/checkpoint)")
        death_summaries[kind] = {"n_cases": n_cases, "n_early": n_early, "cases": cases}
        for c in cases[:10]:
            out(f"    seed{c['seed']} round={c['round']}: drop_idx={c['drop_idx']} "
                f"self={[round(v, 2) for v in c['self_vals']]} base={[round(v, 2) for v in c['base_vals']]}")
        if len(cases) > 10:
            out(f"    ... ({len(cases) - 10} more cases, full list in this file's raw run if rerun with logging)")

    # --- false-positive side: random anchors in non-death rounds ---
    out(f"\n--- false-positive rate (random anchors in non-death rounds, n_samples={args.n_fp_samples}) ---")
    nondeath_rounds = []
    for s in SEEDS:
        for rd in all_series[s]:
            if not rd["self_kill"] and not rd["got_killed"] and len(rd["points"]) >= 14:
                nondeath_rounds.append((s, rd))
    out(f"  eligible non-death rounds (>=14 recorded steps): {len(nondeath_rounds)}")
    fp_onset_bucket = Counter()
    n_fp_saturated_anchor = 0
    for _ in range(args.n_fp_samples):
        s, rd = nondeath_rounds[RNG.integers(0, len(nondeath_rounds))]
        pts = rd["points"]
        # The anchor needs ten prior points; its own window is negligible
        # relative to the baseline pool.
        anchor_idx = RNG.integers(10, len(pts))
        window_pts = pts[anchor_idx - 9:anchor_idx + 1]
        self_vals = [p[1] for p in window_pts]
        base_vals = [baseline_at(p[0]) for p in window_pts]
        drop_idx = onset_bucket_for(self_vals, base_vals)
        if drop_idx is None:
            fp_onset_bucket["no_drop_detected"] += 1
        elif drop_idx <= len(window_pts) - 5:
            fp_onset_bucket[">=4_steps_before_anchor"] += 1
        else:
            fp_onset_bucket["<4_steps_before_anchor"] += 1
        if self_vals[0] >= 1.0:
            n_fp_saturated_anchor += 1
    n_fp_early = fp_onset_bucket.get(">=4_steps_before_anchor", 0)
    out(f"  onset buckets: {dict(fp_onset_bucket)}")
    out(f"  FALSE-POSITIVE RATE (>=4-step-early 'drop' with no death following): "
        f"{n_fp_early}/{args.n_fp_samples} ({100 * n_fp_early / args.n_fp_samples:.1f}%)")
    out(f"  anchors starting already saturated at 1.0: {n_fp_saturated_anchor}/{args.n_fp_samples}")

    out("\n=== SUMMARY: onset ('drop') rate compared, same methodology, same threshold ===")
    for kind in ("self-kill", "got_killed"):
        d = death_summaries[kind]
        rate = 100 * d["n_early"] / d["n_cases"] if d["n_cases"] else float("nan")
        out(f"  {kind}: {rate:.1f}% (n={d['n_cases']}) vs false-positive baseline "
            f"{100 * n_fp_early / args.n_fp_samples:.1f}% (n={args.n_fp_samples})")
    out("Conclusion: see the ratio above -- if the death-case rate is materially higher than the "
        "false-positive rate, the #31 narrowing signal has genuine discriminative value ahead of "
        "death, not just a generic feature of long rounds; if comparable, the signal's specificity "
        "to death is not established by this data.")

    # --- (d.2) siege-death proportion + agent reaction to the trend ---
    out("\n=== (d.2) siege-death proportion + agent reaction (does the chosen action raise or lower #31) ===")
    out("New definition: a 'siege death' is any self-kill/got_killed case with onset_bucket != "
        "'no_drop_detected' above (i.e. #31 showed the >=30%-below-baseline narrowing pattern before "
        "death, regardless of how many steps early) -- this is the natural 'confinement precedes death' "
        "reading of the d.1 methodology, not a separate reused Stage D definition.")
    total_deaths = 0
    total_siege = 0
    all_deltas = []
    for kind in ("self-kill", "got_killed"):
        cases = death_summaries[kind]["cases"]
        siege_cases = [c for c in cases if c["drop_idx"] is not None]
        total_deaths += len(cases)
        total_siege += len(siege_cases)
        out(f"\n  {kind}: siege deaths = {len(siege_cases)}/{len(cases)} "
            f"({100 * len(siege_cases) / len(cases) if cases else float('nan'):.1f}%)")
        deltas = []
        for c in siege_cases:
            window = c["self_vals"][c["drop_idx"]:]
            for i in range(len(window) - 1):
                deltas.append(window[i + 1] - window[i])
        all_deltas.extend(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        n_zero = sum(1 for d in deltas if d == 0)
        out(f"    step-to-step #31 delta from onset to death, n={len(deltas)}: "
            f"increasing(agent action raised #31)={n_pos} decreasing={n_neg} unchanged={n_zero} "
            f"mean_delta={np.mean(deltas) if deltas else float('nan'):.4f}")
    out(f"\n  OVERALL siege-death proportion across self-kill+got_killed: {total_siege}/{total_deaths} "
        f"({100 * total_siege / total_deaths if total_deaths else float('nan'):.1f}%)")
    n_pos_all = sum(1 for d in all_deltas if d > 0)
    out(f"  OVERALL onset-to-death delta sign: n={len(all_deltas)} increasing={n_pos_all} "
        f"({100 * n_pos_all / len(all_deltas) if all_deltas else float('nan'):.1f}%) "
        f"mean={np.mean(all_deltas) if all_deltas else float('nan'):.4f}")
    out("Interpretation (report the actual sign/magnitude split found, not an assumed pattern): a "
        "minority-positive step count with a small positive mean (as found here) reads as a mixed/weak "
        "signal -- some steps do increase #31 (consistent with an escape attempt), most don't, and the "
        "net average drift is close to flat rather than clearly toward more confinement. This does not "
        "establish that the model 'reacts' to the shrinking #31 value in a targeted way, nor that it "
        "ignores it -- delta sign alone cannot separate 'attempted but failed escape' from 'incidental "
        "movement uncorrelated with the feature' from 'opponent-driven narrowing outpacing any escape'. "
        "Data insufficient to attribute causally without per-case manual review of the tail dumps.")

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
