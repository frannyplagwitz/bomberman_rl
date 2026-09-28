"""Diagnostic: re-evaluates selected snapshots of the normalize_returns=True
long runs (B2S2RETNORMLONGSEED0/1) at 100 rounds each, to check whether the
oscillation_fraction trend seen in the small-sample periodic evaluation is
real or noise. Uses run_stage_a2_task2.run_full_evaluation(); read-only.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task2 import run_full_evaluation

MODELS_DIR = common.MODELS_DIR / "snapshots"

# (seed_label, ablation, round): first-half and second-half representatives
# per seed, matching the periodic-eval CSV's split.
TARGETS = [
    (seed, round_)
    for seed in (0, 1)
    for round_ in (100, 500, 900, 1300, 1700, 2050)
]


def _evaluate_one(seed: int, round_: int):
    ckpt = MODELS_DIR / f"task2_stage_a2_B2S2RETNORMLONGSEED{seed}_round{round_}.pt"
    t0 = time.time()
    result = run_full_evaluation(str(ckpt), n_rounds=100, seed=1000)
    elapsed = time.time() - t0
    return seed, round_, result, elapsed


def main():
    log_file, log_path = common.open_log_file("task2_oscillation_trend_snapshots")
    csv_path = log_path.with_suffix(".csv")
    common.log_print(log_file, f"Oscillation trend re-check (100-round eval per snapshot) -- log: {log_path}")
    common.log_print(log_file, f"Targets: {TARGETS}")

    results = {}
    # Low concurrency on purpose: CPU contention is a known confound for
    # oscillation_fraction.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_evaluate_one, seed, round_) for seed, round_ in TARGETS]
        for fut in as_completed(futures):
            seed, round_, result, elapsed = fut.result()
            results[(seed, round_)] = result
            common.log_print(
                log_file,
                f"[seed{seed} round{round_}] oscillation_fraction={result['oscillation_fraction']:.3f} "
                f"crates_destroyed_mean={result['crates_destroyed_mean']:.2f} "
                f"wait_fraction_mean={result['wait_fraction_mean']:.3f} "
                f"completion_rate={result['completion_rate']:.2f} "
                f"self_kill_rate={result['self_kill_rate']:.2f} "
                f"({elapsed:.1f}s)",
            )

    common.log_print(log_file, "\n=== Ordered summary ===")
    header = f"{'seed':>4} {'round':>5} {'oscillation_fraction':>20} {'crates':>8} {'wait':>6} {'completion':>10}"
    common.log_print(log_file, header)
    for seed, round_ in TARGETS:
        r = results[(seed, round_)]
        common.write_metrics_csv_row(csv_path, {
            "seed": seed, "round": round_,
            "oscillation_fraction": r["oscillation_fraction"],
            "crates_destroyed_mean": r["crates_destroyed_mean"],
            "wait_fraction_mean": r["wait_fraction_mean"],
            "completion_rate": r["completion_rate"],
            "self_kill_rate": r["self_kill_rate"],
            "missed_opportunity_rate": r["missed_opportunity_rate"],
        })
        common.log_print(
            log_file,
            f"{seed:>4} {round_:>5} {r['oscillation_fraction']:>20.3f} "
            f"{r['crates_destroyed_mean']:>8.2f} {r['wait_fraction_mean']:>6.3f} {r['completion_rate']:>10.2f}",
        )

    common.log_print(log_file, f"\nCSV: {csv_path}")
    common.ring_bell()


if __name__ == "__main__":
    main()
