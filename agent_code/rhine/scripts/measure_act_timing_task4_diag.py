"""Diagnostic variant of measure_act_timing_task4.py: records both
time.time() (the wall clock the framework's overrun check uses) and
time.perf_counter() around the same act()/do_step() calls, plus every
agent's available_think_time per step. Shows whether a stall appears on both
clocks (a real pause) or only on the wall clock (a clock artifact), and
whether opponents are affected too.

Run conditions match measure_act_timing_task4.py (seed 0, real think-time
limit, no env overrides, 3x rule_based_agent, submission config).

Usage:
  python -m agent_code.rhine.scripts.measure_act_timing_task4_diag [--n-rounds 20]
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
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _pin_opponent_reseed

SCENARIO = "classic"
OPPONENTS = ["rule_based_agent"] * 3
EVAL_SEED = 1000
GAME_LOG_PATH = "logs/game.log"


def _stats(label, wall_ms, perf_ms, log_file):
    wall = np.asarray(wall_ms)
    perf = np.asarray(perf_ms)
    common.log_print(
        log_file,
        f"{label} wall(time.time) ms: mean={wall.mean():.3f} p95={np.percentile(wall, 95):.3f} max={wall.max():.3f} | "
        f"perf(perf_counter) ms: mean={perf.mean():.3f} p95={np.percentile(perf, 95):.3f} max={perf.max():.3f}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int, default=20)
    args = parser.parse_args()

    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False

    log_file, log_path = common.open_log_file("task4_act_timing_diag_seed0")
    common.log_print(log_file, f"Task 4 dual-clock timing diag (seed0) -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} n_rounds={args.n_rounds}")
    common.log_print(
        log_file,
        f"switches: ENABLE_OSCILLATION_BREAKER={cfg.ENABLE_OSCILLATION_BREAKER} "
        f"ENABLE_DEADLOCK_BOMB={cfg.ENABLE_DEADLOCK_BOMB} "
        f"ENABLE_NO_BOMB_WHEN_BOARD_CLEARED={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED} "
        f"n_features_active={cfg.n_features_active()} "
        f"expected_checkpoint={cfg.DEPLOYMENT_CHECKPOINT}",
    )

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)

    game_log_path = Path(GAME_LOG_PATH).resolve()
    offset_before = game_log_path.stat().st_size if game_log_path.is_file() else 0
    common.log_print(log_file, f"game.log byte offset before run: {offset_before} ({game_log_path})")

    act_wall_ms, act_perf_ms = [], []
    step_wall_ms, step_perf_ms = [], []
    # Every agent's available_think_time (name -> ms) right before each do_step().
    budget_records = []

    original_act = rhine_callbacks.act

    def timed_act(self, game_state):
        t0_wall, t0_perf = time.time(), time.perf_counter()
        result = original_act(self, game_state)
        act_wall_ms.append((time.time() - t0_wall) * 1000.0)
        act_perf_ms.append((time.perf_counter() - t0_perf) * 1000.0)
        return result

    rhine_callbacks.act = timed_act
    try:
        with _pin_opponent_reseed(EVAL_SEED):
            world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
            rhine_agent = world.agents[0]
            for round_idx in range(args.n_rounds):
                world.new_round()
                world.user_input = "WAIT"
                step_idx = 0
                while world.running:
                    step_idx += 1
                    budget_records.append({
                        "round": round_idx + 1, "step": step_idx,
                        "budgets": {a.name: a.available_think_time * 1000.0 for a in world.agents},
                        "alive": {a.name: not a.dead for a in world.agents},
                    })
                    rhine_alive = not rhine_agent.dead
                    t0_wall, t0_perf = time.time(), time.perf_counter()
                    world.do_step("WAIT")
                    if rhine_alive:
                        step_wall_ms.append((time.time() - t0_wall) * 1000.0)
                        step_perf_ms.append((time.perf_counter() - t0_perf) * 1000.0)
            world.end()
    finally:
        rhine_callbacks.act = original_act

    offset_after = game_log_path.stat().st_size
    with open(game_log_path, "r") as f:
        f.seek(offset_before)
        new_log_text = f.read()
    common.log_print(log_file, f"game.log byte offset after run: {offset_after} (new bytes: {offset_after - offset_before})")

    warning_lines = [ln for ln in new_log_text.splitlines() if "exceeded think time" in ln]
    skip_lines = [ln for ln in new_log_text.splitlines() if "Skipping agent" in ln]
    common.log_print(log_file, f"\nnew 'exceeded think time' warnings ({len(warning_lines)}):")
    for ln in warning_lines:
        common.log_print(log_file, f"  {ln}")
    common.log_print(log_file, f"new 'Skipping agent' (forced WAIT) lines ({len(skip_lines)}):")
    for ln in skip_lines:
        common.log_print(log_file, f"  {ln}")

    # Per-agent forced-WAIT count from these snapshots, independent of game.log.
    forced_wait_by_agent = {}
    for rec in budget_records:
        for name, budget in rec["budgets"].items():
            if rec["alive"].get(name) and budget <= 0:
                forced_wait_by_agent[name] = forced_wait_by_agent.get(name, 0) + 1
    common.log_print(log_file, f"\nforced_waits by agent (available_think_time<=0, alive only): {forced_wait_by_agent}")

    n = min(len(act_wall_ms), len(act_perf_ms))
    diffs = [act_wall_ms[i] - act_perf_ms[i] for i in range(n)]
    big_diff_idx = [i for i, d in enumerate(diffs) if abs(d) > 100]
    common.log_print(log_file, f"\nact() steps with |wall - perf| > 100ms: {len(big_diff_idx)} / {n}")
    for i in big_diff_idx:
        common.log_print(log_file, f"  step_call_index={i} wall={act_wall_ms[i]:.1f}ms perf={act_perf_ms[i]:.1f}ms diff={diffs[i]:.1f}ms")

    _stats("act()", act_wall_ms, act_perf_ms, log_file)
    _stats("do_step()", step_wall_ms, step_perf_ms, log_file)

    log_file.close()


if __name__ == "__main__":
    main()
