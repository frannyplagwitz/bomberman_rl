"""Small-scale experiment: does the higher CRATE_DESTROYED_REWARD_NO_COIN
(used when no coin is reachable) reduce WAIT collapse (B1) / oscillation (B2)
and increase crate clearing, before committing to a full retrain?

Trains only the new-reward variant under the default reward config. The
baseline is the matching-budget, same-protocol baseline_-0.01 checkpoints
from experiment_step_cost.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import longest_oscillation_run, OSCILLATION_THRESHOLD

STEP_BUDGET = 40_000
EVAL_ROUNDS = 15
EVAL_SEED = 1000
TRAIN_SEED = 0
SCENARIO = "loot-crate"
SCRATCH_DIR = common.LOGS_DIR / "crate_no_coin_reward_experiment"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = {
    "B1": {"ENABLE_BOMBING_PROGRESS_SHAPING": False, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False},
    "B2": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False},
}


def train_fresh(ablation, seed, step_budget):
    checkpoint_path = SCRATCH_DIR / f"{ablation}_new_crate_reward.pt"
    import json
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", json.dumps(ABLATIONS[ablation]))

    world = common.build_world(scenario=SCENARIO, seed=seed, train=True)
    total = 0
    while total < step_budget:
        ep = common.run_one_round(world)
        total += ep.steps
    world.end()
    return checkpoint_path, total


def evaluate_full(checkpoint_path, n_rounds, seed):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False)
    agent = world.agents[0]

    wait_fractions = []
    crates, bombs = [], []
    n_sustained = 0
    for _ in range(n_rounds):
        ep = common.run_one_round(world)
        wait_fractions.append(ep.wait_fraction)
        crates.append(ep.crates_destroyed)
        bombs.append(ep.bombs_dropped)
        actions = list(world.replay["actions"][agent.name])
        run_len, _ = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            n_sustained += 1
    world.end()

    crates_mean = float(np.mean(crates))
    bombs_mean = float(np.mean(bombs))
    return {
        "wait_fraction_mean": float(np.mean(wait_fractions)),
        "oscillation_fraction": n_sustained / n_rounds,
        "crates_destroyed_mean": crates_mean,
        "bombs_dropped_mean": bombs_mean,
        "bomb_efficiency": crates_mean / bombs_mean if bombs_mean > 0 else float("nan"),
    }


def main():
    log_file, log_path = common.open_log_file("task2_crate_no_coin_reward_experiment")
    common.log_print(log_file, f"CRATE_DESTROYED_REWARD_NO_COIN diagnostic experiment -- log file: {log_path}")
    common.log_print(log_file, f"step_budget={STEP_BUDGET} eval_rounds={EVAL_ROUNDS}")

    for ablation in ["B1", "B2"]:
        ckpt, actual_steps = train_fresh(ablation, TRAIN_SEED, STEP_BUDGET)
        metrics = evaluate_full(ckpt, EVAL_ROUNDS, EVAL_SEED)
        common.log_print(
            log_file,
            f"\n=== {ablation} (new reward, trained {actual_steps} steps) ===\n"
            f"  wait_fraction_mean={metrics['wait_fraction_mean']:.3f} "
            f"oscillation_fraction={metrics['oscillation_fraction']:.2f} "
            f"({int(metrics['oscillation_fraction']*EVAL_ROUNDS)}/{EVAL_ROUNDS} rounds)\n"
            f"  crates_destroyed_mean={metrics['crates_destroyed_mean']:.2f} "
            f"bombs_dropped_mean={metrics['bombs_dropped_mean']:.2f} "
            f"bomb_efficiency={metrics['bomb_efficiency']:.2f}",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
