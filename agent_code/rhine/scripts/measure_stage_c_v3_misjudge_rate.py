"""Task 3 Stage C v3 (34-dim) misjudge_rate measurement, using the same method
as the Stage A baseline: replay_self_kill_cases.replay_checkpoint() with
Stage C's opponent (coin_collector_agent) and the three v3 seeds.

Diagnostic pass, so the think-time limit is disabled; seeds run in parallel.

Usage:
  python -m agent_code.rhine.scripts.measure_stage_c_v3_misjudge_rate --seed N
"""
import argparse
import random as _random_module
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.replay_self_kill_cases import replay_checkpoint

OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100
SEEDS = [0, 1, 2]


@contextmanager
def _pin_opponent_reseed(fixed_seed: int):
    """Pins opponents' Python `random` reseeding; see
    diagnose_selfkill_bfs_trace.py's identical helper."""
    original_np_seed = np.random.seed
    original_random_state = _random_module.getstate()
    np.random.seed = lambda *args, **kwargs: original_np_seed(fixed_seed)
    _random_module.seed(fixed_seed)
    try:
        yield
    finally:
        np.random.seed = original_np_seed
        _random_module.setstate(original_random_state)


def run_one_seed(seed, log_file):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    ckpt = common.MODELS_DIR / f"task3_stage_c_v3_seed{seed}.pt"
    label = f"seed{seed}"
    common.log_print(log_file, f"\n=== {label} ({ckpt.name}) ===")

    with _pin_opponent_reseed(EVAL_SEED):
        self_kills, stats = replay_checkpoint(ckpt, OPPONENTS, N_ROUNDS, EVAL_SEED, label)

    conclusive = stats["prediction_confirmed"] + stats["prediction_false_positives"]
    misjudge_rate = stats["prediction_false_positives"] / conclusive if conclusive else float("nan")
    common.log_print(
        log_file,
        f"  self_kill_rate (deterministic-opponent replay) = {len(self_kills)}/{N_ROUNDS} = {len(self_kills)/N_ROUNDS:.3f}\n"
        f"  misjudge_rate = {misjudge_rate:.3f} "
        f"({stats['prediction_false_positives']}/{conclusive} conclusively-checkable prediction-only denials) "
        f"[{stats['prediction_denials']} total denials, {stats['prediction_inconclusive']} inconclusive]",
    )
    return {
        "seed": seed, "misjudge_rate": misjudge_rate,
        "false_positives": stats["prediction_false_positives"], "conclusive": conclusive,
        "total_denials": stats["prediction_denials"], "inconclusive": stats["prediction_inconclusive"],
        "self_kill_rate": len(self_kills) / N_ROUNDS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                         help="Run only this training seed (for parallel invocation). Default: all seeds sequentially.")
    args = parser.parse_args()

    seeds_to_run = [args.seed] if args.seed is not None else SEEDS
    tag = f"seed{args.seed}" if args.seed is not None else "all"
    log_file, log_path = common.open_log_file(f"stage_c_v3_misjudge_rate_{tag}")
    common.log_print(log_file, f"Stage C v3 misjudge_rate -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={N_ROUNDS}")

    results = [run_one_seed(seed, log_file) for seed in seeds_to_run]

    if len(results) > 1:
        rates = [r["misjudge_rate"] for r in results]
        common.log_print(log_file, f"\n3-seed misjudge_rate: mean={np.nanmean(rates):.3f} "
                                    f"values={[round(r, 3) for r in rates]}")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
