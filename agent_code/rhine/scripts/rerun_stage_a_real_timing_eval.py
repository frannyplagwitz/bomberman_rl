"""One-off: re-runs Task 3 Stage A's final 100-round evaluation of
task3_stage_a_seed{0,1,2}.pt with the real think-time limit, via
run_stage_a2_task3.run_full_evaluation().

Real-timing pass for report data: seeds run serially, and nothing else
should run concurrently.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task3 import run_full_evaluation

OPPONENTS = ["peaceful_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


def main():
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    log_file, log_path = common.open_log_file("task3_stage_a_real_timing_final_eval")
    common.log_print(log_file, f"Stage A real-timing final eval -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} "
                                f"disable_think_time_limit=False (real timing)")

    for seed_id in (0, 1, 2):
        checkpoint = common.MODELS_DIR / f"task3_stage_a_seed{seed_id}.pt"
        common.log_print(log_file, f"\n=== seed{seed_id} ({checkpoint}) ===")
        agg = run_full_evaluation(
            str(checkpoint), N_ROUNDS, EVAL_SEED, OPPONENTS, disable_think_time_limit=False,
        )
        common.log_print(
            log_file,
            f"score_mean={agg['score_mean']:.2f} score_total={agg['score_total']:.2f}\n"
            f"  self_kill_rate={agg['self_kill_rate']:.3f} oscillation_fraction={agg['oscillation_fraction']:.3f} "
            f"({int(agg['oscillation_fraction'] * N_ROUNDS)}/{N_ROUNDS})\n"
            f"  completion_rate={agg['completion_rate']:.3f} opponent_kills_mean={agg['opponent_kills_mean']:.3f} "
            f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.3f}\n"
            f"  invalid_action_count_total={agg.get('invalid_action_count_total')}\n"
            f"  bombing_target_optimal_rate={agg['bombing_target_optimal_rate']:.3f} "
            f"missed_opportunity_rate={agg['missed_opportunity_rate']:.3f} "
            f"crates_destroyed_mean={agg['crates_destroyed_mean']:.2f}\n"
            f"  wait_fraction_mean={agg['wait_fraction_mean']:.3f}",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
