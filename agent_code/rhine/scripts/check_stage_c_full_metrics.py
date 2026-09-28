"""One-off: full-metric 100-round re-evaluation of a Task 3 Stage C checkpoint
(task3_stage_c_seed{0,1,2}.pt) against one coin_collector_agent, with the
oscillation breaker on or off, via run_stage_a2_task3.run_full_evaluation().

Usage: check_stage_c_full_metrics.py <seed_id> <on|off>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task3 import run_full_evaluation

OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


def main():
    seed_id = int(sys.argv[1])
    breaker_on = sys.argv[2] == "on"

    cfg.ENABLE_OSCILLATION_BREAKER = breaker_on
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    tag = "breaker_on" if breaker_on else "breaker_off"
    checkpoint = common.MODELS_DIR / f"task3_stage_c_seed{seed_id}.pt"
    log_file, log_path = common.open_log_file(f"stage_c_seed{seed_id}_{tag}_full_metrics")
    common.log_print(log_file, f"Stage C seed{seed_id} {tag} full-metrics eval -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={checkpoint} eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} "
                                f"ENABLE_OSCILLATION_BREAKER={breaker_on}")

    agg = run_full_evaluation(
        str(checkpoint), N_ROUNDS, EVAL_SEED, OPPONENTS, disable_think_time_limit=True,
    )
    common.log_print(
        log_file,
        f"score_mean={agg['score_mean']:.2f} score_total={agg['score_total']:.2f}\n"
        f"  self_kill_rate={agg['self_kill_rate']:.3f} oscillation_fraction={agg['oscillation_fraction']:.3f} "
        f"({int(agg['oscillation_fraction'] * N_ROUNDS)}/{N_ROUNDS})\n"
        f"  completion_rate={agg['completion_rate']:.3f} opponent_kills_mean={agg['opponent_kills_mean']:.3f} "
        f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.3f}\n"
        f"  episode_length_mean={agg['episode_length_mean']:.2f} "
        f"invalid_action_count_total={agg.get('invalid_action_count_total')}\n"
        f"  bombing_target_optimal_rate={agg['bombing_target_optimal_rate']:.3f} "
        f"missed_opportunity_rate={agg['missed_opportunity_rate']:.3f} "
        f"crates_destroyed_mean={agg['crates_destroyed_mean']:.2f}\n"
        f"  wait_fraction_mean={agg['wait_fraction_mean']:.3f}",
    )
    log_file.close()


if __name__ == "__main__":
    main()
