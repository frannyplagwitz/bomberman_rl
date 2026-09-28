"""One-off: reports invalid_action_count_total for Task 3 Stage B's
checkpoints by re-running run_full_evaluation() on them (no retraining).
Supplementary diagnostic, so the think-time limit is disabled.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.run_stage_a2_task3 import run_full_evaluation

OPPONENTS = ["peaceful_agent", "peaceful_agent", "peaceful_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


def main():
    seed_id = int(sys.argv[1])
    cfg.ENABLE_OSCILLATION_BREAKER = False
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    # Configuration these checkpoints were trained and evaluated under.
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    checkpoint = common.MODELS_DIR / f"task3_stage_b_seed{seed_id}.pt"
    log_file, log_path = common.open_log_file(f"stage_b_seed{seed_id}_invalid_action_check")
    common.log_print(log_file, f"Stage B seed{seed_id} invalid_action_count check -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={checkpoint} eval_seed={EVAL_SEED} n_rounds={N_ROUNDS}")

    agg = run_full_evaluation(
        str(checkpoint), N_ROUNDS, EVAL_SEED, OPPONENTS, disable_think_time_limit=True,
    )
    common.log_print(
        log_file,
        f"score_mean={agg['score_mean']:.2f} invalid_action_count_total={agg.get('invalid_action_count_total')} "
        f"self_kill_rate={agg['self_kill_rate']:.3f}",
    )
    log_file.close()


if __name__ == "__main__":
    main()
