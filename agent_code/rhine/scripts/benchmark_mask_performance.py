"""Benchmark of act()'s full per-step cost against the 0.5s think-time budget.

Times the whole act() call (semantic extraction, features, mask, forward
pass) as the framework does, plus a mask-only breakdown, over a few hundred
real loot-crate states reached by playing the checkpoint. Each run appends
one CSV row so results can be compared across code changes.

Measures act() in isolation (run without other heavy load); think-time
timeouts caused by concurrent CPU contention are a separate issue.
"""
import csv
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine import callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.state_processing import extract_semantic_state
from agent_code.rhine.action_mask import mask_from_semantic

N_SAMPLES = 500
SCENARIO = "loot-crate"
SEED = 1000
# Final Task 2 checkpoint.
CHECKPOINT = common.MODELS_DIR / "task2_stage_a2_B2S2NORMSCORING_SEED4.pt"


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def collect_samples(n_samples: int):
    # The breaker is part of the deployed configuration, so its overhead is included.
    cfg.ENABLE_OSCILLATION_BREAKER = True
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(CHECKPOINT))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=SEED, train=False)
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self

    act_durations = []
    mask_durations = []
    while len(act_durations) < n_samples:
        world.new_round()
        world.user_input = "WAIT"
        while world.running and len(act_durations) < n_samples:
            state = world.get_state_for_agent(agent)
            if state is None:
                world.do_step("WAIT")
                continue

            t0 = time.perf_counter()
            action = callbacks.act(fake_self, state)
            act_durations.append(time.perf_counter() - t0)

            # Secondary breakdown: mask-only, recomputed independently so it
            # doesn't add to the act() timing above.
            t1 = time.perf_counter()
            semantic = extract_semantic_state(state)
            mask_from_semantic(semantic)
            mask_durations.append(time.perf_counter() - t1)

            world.do_step(action)  # Feed the real decision back for a realistic trajectory.
    world.end()
    return np.array(act_durations) * 1000.0, np.array(mask_durations) * 1000.0  # ms


def _stats(durations_ms):
    return {
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "max_ms": float(np.max(durations_ms)),
    }


def main():
    log_file, log_path = common.open_log_file("mask_performance_benchmark")
    cpu_model = get_cpu_model()
    common.log_print(log_file, f"act() performance benchmark -- log file: {log_path}")
    common.log_print(log_file, f"CPU: {cpu_model}")
    common.log_print(log_file, f"scenario={SCENARIO} seed={SEED} n_samples={N_SAMPLES} checkpoint={CHECKPOINT.name}")

    act_ms, mask_ms = collect_samples(N_SAMPLES)
    act_stats = _stats(act_ms)
    mask_stats = _stats(mask_ms)

    common.log_print(
        log_file,
        f"FULL act() (extractor + features + mask + model forward): "
        f"mean={act_stats['mean_ms']:.3f}ms median={act_stats['median_ms']:.3f}ms "
        f"p95={act_stats['p95_ms']:.3f}ms max={act_stats['max_ms']:.3f}ms "
        f"(budget: 500ms per act() call)",
    )
    common.log_print(
        log_file,
        f"  breakdown, mask-only (extract_semantic_state + mask_from_semantic): "
        f"mean={mask_stats['mean_ms']:.3f}ms median={mask_stats['median_ms']:.3f}ms "
        f"p95={mask_stats['p95_ms']:.3f}ms max={mask_stats['max_ms']:.3f}ms",
    )

    csv_path = common.LOGS_DIR / "mask_performance_benchmark_history.csv"
    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp", "cpu_model", "n_samples",
        "act_mean_ms", "act_median_ms", "act_p95_ms", "act_max_ms",
        "mask_mean_ms", "mask_median_ms", "mask_p95_ms", "mask_max_ms",
        "note",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cpu_model": cpu_model,
            "n_samples": N_SAMPLES,
            "act_mean_ms": round(act_stats["mean_ms"], 4),
            "act_median_ms": round(act_stats["median_ms"], 4),
            "act_p95_ms": round(act_stats["p95_ms"], 4),
            "act_max_ms": round(act_stats["max_ms"], 4),
            "mask_mean_ms": round(mask_stats["mean_ms"], 4),
            "mask_median_ms": round(mask_stats["median_ms"], 4),
            "mask_p95_ms": round(mask_stats["p95_ms"], 4),
            "mask_max_ms": round(mask_stats["max_ms"], 4),
            "note": "",
        })
    common.log_print(log_file, f"history CSV: {csv_path}")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
