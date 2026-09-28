"""Per-step act() wall-clock timing for the Stage D checkpoints with the real
think-time limit enforced, against 3x coin_collector_agent.

Times only the real callbacks.act() call (perf_counter around it), like
diagnose_act_timing_distribution.py, which itself no longer runs on the
28-dim feature set. Whole do_step() time (including opponents' act() calls,
which do not count against our budget) is recorded for context.

Run alone on an idle machine; nothing is written under models/.

Usage:
  python -m agent_code.rhine.scripts.measure_act_timing_stage_d [--n-rounds 20]
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
OPPONENTS = ["coin_collector_agent"] * 3
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

    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True

    log_file, log_path = common.open_log_file("stage_d_act_timing")
    common.log_print(log_file, f"Stage D act() timing, real think-time enforced -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={args.n_rounds} breaker=on")

    all_act, all_step, all_overruns = [], [], 0
    for seed in (0, 1, 2):
        checkpoint = common.MODELS_DIR / f"task3_stage_d_seed{seed}.pt"
        common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint))
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
        buckets = {"<100ms": 0, "100-300ms": 0, "300-500ms": 0, ">500ms": 0}
        for d in act_ms:
            buckets[bucket_label(d)] += 1
        summarize(f"seed{seed}", act_ms, step_ms, overruns, log_file)
        common.log_print(log_file, f"  buckets={buckets}")
        all_act += act_ms
        all_step += step_ms
        all_overruns += overruns

    summarize("ALL", all_act, all_step, all_overruns, log_file)
    log_file.close()


if __name__ == "__main__":
    main()
