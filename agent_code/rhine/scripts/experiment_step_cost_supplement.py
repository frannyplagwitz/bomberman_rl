"""Supplement to experiment_step_cost.py: re-evaluates its four scratch
checkpoints and additionally reports crates_destroyed_mean,
bombs_dropped_mean and bomb_efficiency, so lower wait/oscillation can be
checked against actual productivity. No retraining.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.experiment_step_cost import (
    SCRATCH_DIR, STEP_COST_VARIANTS, EVAL_ROUNDS, EVAL_SEED, SCENARIO,
)


def evaluate_full(checkpoint_path, n_rounds, seed):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False)
    episodes = [common.run_one_round(world) for _ in range(n_rounds)]
    world.end()

    crates_mean = float(np.mean([ep.crates_destroyed for ep in episodes]))
    bombs_mean = float(np.mean([ep.bombs_dropped for ep in episodes]))
    bomb_efficiency = crates_mean / bombs_mean if bombs_mean > 0 else float("nan")
    return crates_mean, bombs_mean, bomb_efficiency


def main():
    log_file, log_path = common.open_log_file("task2_step_cost_supplement")
    common.log_print(log_file, f"STEP_COST experiment supplement (crates/bombs) -- log file: {log_path}")

    for ablation in ["B1", "B2"]:
        common.log_print(log_file, f"\n=== {ablation} ===")
        for variant_name, step_cost in STEP_COST_VARIANTS.items():
            ckpt = SCRATCH_DIR / f"{ablation}_stepcost{step_cost}.pt"
            crates_mean, bombs_mean, bomb_eff = evaluate_full(ckpt, EVAL_ROUNDS, EVAL_SEED)
            common.log_print(
                log_file,
                f"  {variant_name}: crates_destroyed_mean={crates_mean:.2f} "
                f"bombs_dropped_mean={bombs_mean:.2f} bomb_efficiency={bomb_eff:.2f}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
