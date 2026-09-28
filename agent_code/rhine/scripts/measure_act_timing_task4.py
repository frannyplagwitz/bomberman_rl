"""Per-step act() wall-clock timing for the Task 4 submission checkpoint,
loaded through cfg.DEPLOYMENT_CHECKPOINT with no env overrides (the real
submission path), with the real think-time limit enforced, against
3x rule_based_agent.

ENABLE_NO_BOMB_WHEN_BOARD_CLEARED is off (Task 4 config); breaker/deadlock
stay at config.py defaults. Times only the callbacks.act() call; whole
do_step() time (including opponents) is recorded for context.

Run alone on an idle machine; nothing is written under models/.

Usage:
  python -m agent_code.rhine.scripts.measure_act_timing_task4 [--n-rounds 20]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_act_timing_distribution import bucket_label
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _pin_opponent_reseed

SCENARIO = "classic"
OPPONENTS = ["rule_based_agent"] * 3
EVAL_SEED = 1000


def summarize(label, act_ms, step_ms, overruns, log_file):
    common.log_print(
        log_file,
        f"{label}: steps={len(act_ms)} act() ms mean={np.mean(act_ms):.3f} median={np.median(act_ms):.3f} "
        f"p95={np.percentile(act_ms, 95):.3f} p99={np.percentile(act_ms, 99):.3f} max={np.max(act_ms):.3f} | "
        f"do_step() ms mean={np.mean(step_ms):.3f} max={np.max(step_ms):.3f} | real overruns={overruns}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int, default=20)
    args = parser.parse_args()

    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False

    log_file, log_path = common.open_log_file("task4_act_timing_seed0")
    common.log_print(log_file, f"Task 4 act() timing (seed0), real think-time enforced -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={args.n_rounds}")
    common.log_print(
        log_file,
        f"switches: ENABLE_OSCILLATION_BREAKER={cfg.ENABLE_OSCILLATION_BREAKER} "
        f"ENABLE_DEADLOCK_BOMB={cfg.ENABLE_DEADLOCK_BOMB} "
        f"ENABLE_NO_BOMB_WHEN_BOARD_CLEARED={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED} "
        f"n_features_active={cfg.n_features_active()} "
        f"expected_checkpoint={cfg.DEPLOYMENT_CHECKPOINT}",
    )

    # No override: load through the default submission path.
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)

    act_ms, step_ms, budgets = [], [], []
    original_act = rhine_callbacks.act

    def timed_act(self, game_state):
        t0 = time.perf_counter()
        result = original_act(self, game_state)
        act_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    rhine_callbacks.act = timed_act
    try:
        with _pin_opponent_reseed(EVAL_SEED):
            world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
            agent = world.agents[0]
            for _ in range(args.n_rounds):
                world.new_round()
                world.user_input = "WAIT"
                while world.running:
                    alive = not agent.dead
                    if alive:
                        budgets.append(agent.available_think_time * 1000.0)
                    t0 = time.perf_counter()
                    world.do_step("WAIT")
                    if alive:
                        step_ms.append((time.perf_counter() - t0) * 1000.0)
            world.end()
    finally:
        rhine_callbacks.act = original_act

    n = min(len(act_ms), len(budgets))
    overruns = sum(1 for i in range(n) if act_ms[i] > budgets[i])
    forced_waits = sum(1 for i in range(n) if budgets[i] <= 0)
    buckets = {"<100ms": 0, "100-300ms": 0, "300-500ms": 0, ">500ms": 0}
    for d in act_ms:
        buckets[bucket_label(d)] += 1
    summarize("seed0", act_ms, step_ms, overruns, log_file)
    common.log_print(log_file, f"  buckets={buckets}")
    common.log_print(log_file, f"  forced_waits(available_think_time<=0)={forced_waits}")
    log_file.close()


if __name__ == "__main__":
    main()
