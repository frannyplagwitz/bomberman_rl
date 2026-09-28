"""One-off: Task 3 Stage C's final 100-round evaluation of
task3_stage_c_seed{0,1,2}.pt with the real think-time limit, via
run_stage_a2_task3.run_full_evaluation(), plus a minimal act() timing
monkeypatch (as in diagnose_act_timing_distribution.py).

Real-timing pass for report data: seeds run serially, and nothing else
should run concurrently.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task3 import run_full_evaluation

OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


def main():
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    log_file, log_path = common.open_log_file("task3_stage_c_real_timing_final_eval")
    common.log_print(log_file, f"Stage C real-timing final eval -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} "
                                f"disable_think_time_limit=False (real timing)")

    per_seed_summary = []

    for seed_id in (0, 1, 2):
        checkpoint = common.MODELS_DIR / f"task3_stage_c_seed{seed_id}.pt"
        common.log_print(log_file, f"\n=== seed{seed_id} ({checkpoint}) ===")

        original_act = rhine_callbacks.act
        durations_ms = []

        def timed_act(self, game_state):
            t0 = time.perf_counter()
            result = original_act(self, game_state)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)
            return result

        rhine_callbacks.act = timed_act
        try:
            agg = run_full_evaluation(
                str(checkpoint), N_ROUNDS, EVAL_SEED, OPPONENTS, disable_think_time_limit=False,
            )
        finally:
            rhine_callbacks.act = original_act

        durations = np.array(durations_ms)
        near_or_over_budget = int(np.sum(durations >= 400))
        over_budget = int(np.sum(durations >= 500))

        common.log_print(
            log_file,
            f"score_mean={agg['score_mean']:.2f} score_total={agg['score_total']:.2f}\n"
            f"  self_kill_rate={agg['self_kill_rate']:.3f} opponent_kills_mean={agg['opponent_kills_mean']:.3f} "
            f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.3f}\n"
            f"  completion_rate={agg['completion_rate']:.3f} oscillation_fraction={agg['oscillation_fraction']:.3f} "
            f"({int(agg['oscillation_fraction'] * N_ROUNDS)}/{N_ROUNDS})\n"
            f"  episode_length_mean={agg['episode_length_mean']:.2f} "
            f"invalid_action_count_total={agg.get('invalid_action_count_total')}\n"
            f"act() timing (ms): n={len(durations)} mean={durations.mean():.2f} median={np.median(durations):.2f} "
            f"p95={np.percentile(durations, 95):.2f} p99={np.percentile(durations, 99):.2f} max={durations.max():.2f}\n"
            f"  steps>=400ms: {near_or_over_budget}/{len(durations)}  steps>=500ms (over budget): "
            f"{over_budget}/{len(durations)}",
        )

        per_seed_summary.append({
            "seed": seed_id,
            "score_mean": agg["score_mean"], "score_total": agg["score_total"],
            "self_kill_rate": agg["self_kill_rate"], "opponent_kills_mean": agg["opponent_kills_mean"],
            "got_killed_by_opponent_rate": agg["got_killed_by_opponent_rate"],
            "completion_rate": agg["completion_rate"],
            "act_mean_ms": float(durations.mean()), "act_max_ms": float(durations.max()),
        })

    common.log_print(log_file, "\n=== mean +/- std across seeds (ddof=1) ===")
    keys = ["score_mean", "score_total", "self_kill_rate", "opponent_kills_mean",
            "got_killed_by_opponent_rate", "completion_rate", "act_mean_ms", "act_max_ms"]
    for key in keys:
        values = [s[key] for s in per_seed_summary]
        common.log_print(log_file, f"  {key}: {np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}  (values={values})")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
