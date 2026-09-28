"""Small-scale experiment: does a doubled STEP_COST reduce WAIT collapse (B1) /
oscillation (B2), before committing to a full retrain?

Trains B1 and B2 under both the baseline and the candidate STEP_COST from a
fresh random init with the same short budget; other reward fields are
unchanged. Each checkpoint is evaluated deterministically for
wait_fraction_mean and the oscillation-round fraction
(diagnose_oscillation.py's detector). Everything uses a scratch directory.
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.diagnose_oscillation import longest_oscillation_run, OSCILLATION_THRESHOLD

STEP_BUDGET = 40_000
EVAL_ROUNDS = 15
EVAL_SEED = 1000
TRAIN_SEED = 0
SCENARIO = "loot-crate"
SCRATCH_DIR = common.LOGS_DIR / "step_cost_experiment"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = {
    "B1": {"ENABLE_BOMBING_PROGRESS_SHAPING": False, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False},
    "B2": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False},
}
STEP_COST_VARIANTS = {
    "baseline_-0.01": -0.01,
    "candidate_-0.02": -0.02,
}


def train_fresh(ablation, step_cost, seed, step_budget):
    reward_override = dict(ABLATIONS[ablation])
    reward_override["STEP_COST"] = step_cost
    checkpoint_path = SCRATCH_DIR / f"{ablation}_stepcost{step_cost}.pt"

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    import json
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", json.dumps(reward_override))

    world = common.build_world(scenario=SCENARIO, seed=seed, train=True)
    total = 0
    while total < step_budget:
        ep = common.run_one_round(world)
        total += ep.steps
    world.end()
    return checkpoint_path, total


def evaluate_checkpoint(checkpoint_path, n_rounds, seed):
    """Computes wait_fraction_mean and the sustained-oscillation round
    fraction in a single evaluation pass.
    """
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False)
    agent = world.agents[0]

    wait_fractions = []
    n_sustained = 0
    for _ in range(n_rounds):
        ep = common.run_one_round(world)
        wait_fractions.append(ep.wait_fraction)
        actions = list(world.replay["actions"][agent.name])
        run_len, _ = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            n_sustained += 1
    world.end()

    return float(np.mean(wait_fractions)), n_sustained / n_rounds


def main():
    log_file, log_path = common.open_log_file("task2_step_cost_experiment")
    common.log_print(log_file, f"STEP_COST diagnostic experiment -- log file: {log_path}")
    common.log_print(log_file, f"step_budget={STEP_BUDGET} eval_rounds={EVAL_ROUNDS}")

    for ablation in ["B1", "B2"]:
        common.log_print(log_file, f"\n=== {ablation} ===")
        for variant_name, step_cost in STEP_COST_VARIANTS.items():
            ckpt, actual_steps = train_fresh(ablation, step_cost, TRAIN_SEED, STEP_BUDGET)
            wait_mean, osc_frac = evaluate_checkpoint(ckpt, EVAL_ROUNDS, EVAL_SEED)
            common.log_print(
                log_file,
                f"  {variant_name}: trained {actual_steps} steps -- "
                f"wait_fraction_mean={wait_mean:.3f} oscillation_fraction={osc_frac:.2f} "
                f"({int(osc_frac*EVAL_ROUNDS)}/{EVAL_ROUNDS} rounds)",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
