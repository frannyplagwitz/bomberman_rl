"""Read-only diagnostic: per-step wall-clock timing distribution of act()
under real think-time enforcement, for task3_stage_a_seed0.pt.

act() is timed by a minimal monkeypatch around the original call, so this
script's own bookkeeping never counts against the budget. Context for
attributing spikes (bomb availability, opponent reachability, danger, crates
remaining, legal-action count) is captured before each world.do_step(),
outside the timed call.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import extract_semantic_state

SCENARIO = "classic"
OPPONENTS = ["peaceful_agent"]
EVAL_SEED = 1000
N_ROUNDS = 15
CHECKPOINT = common.MODELS_DIR / "task3_stage_a_seed0.pt"

BUCKETS = [100, 300, 500]  # Bucket edges in ms.


def bucket_label(ms):
    if ms < 100:
        return "<100ms"
    if ms < 300:
        return "100-300ms"
    if ms < 500:
        return "300-500ms"
    return ">500ms"


def main():
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(CHECKPOINT))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    original_act = rhine_callbacks.act
    timings = []  # One dict per real act() call.

    def timed_act(self, game_state):
        t0 = time.perf_counter()
        result = original_act(self, game_state)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append({"step": game_state["step"] if game_state else None, "duration_ms": dt_ms})
        return result

    rhine_callbacks.act = timed_act

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    agent = world.agents[0]

    contexts = []  # Index-aligned with timings.
    overrun_steps = 0

    for round_idx in range(N_ROUNDS):
        world.new_round()
        world.user_input = "WAIT"
        # Real timing: the think-time limit stays enabled.
        while world.running:
            state = world.get_state_for_agent(agent)
            budget_before = agent.available_think_time
            if state is not None:
                semantic = extract_semantic_state(state)
                pre_mask = mask_from_semantic(semantic)
                contexts.append({
                    "round": round_idx,
                    "step": state["step"],
                    "budget_before_ms": budget_before * 1000.0,
                    "bomb_available": semantic.bomb_available,
                    "has_reachable_opponent": semantic.has_reachable_opponent,
                    "has_bombing_target": semantic.has_bombing_target,
                    "current_tile_in_danger": semantic.current_tile_in_danger,
                    "nearest_threat_timer": semantic.nearest_threat_timer,
                    "n_legal_pre_mask": int(np.sum(pre_mask)),
                    "crates_remaining": int(np.sum(semantic.field_arr == 1)),
                })
            world.do_step("WAIT")

    world.end()
    rhine_callbacks.act = original_act

    # Both lists are appended once per non-None state, so indices align.
    n = min(len(timings), len(contexts))
    timings, contexts = timings[:n], contexts[:n]

    durations = [t["duration_ms"] for t in timings]
    log_file, log_path = common.open_log_file("task3_stage_a_act_timing_distribution")
    common.log_print(log_file, f"act() timing distribution -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={CHECKPOINT} scenario={SCENARIO} opponents={OPPONENTS} "
                                f"eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} real_timing=True")
    common.log_print(log_file, f"total steps recorded: {n}")

    bucket_counts = {"<100ms": 0, "100-300ms": 0, "300-500ms": 0, ">500ms": 0}
    for d in durations:
        bucket_counts[bucket_label(d)] += 1
    common.log_print(log_file, f"bucket counts: {bucket_counts}")
    common.log_print(
        log_file,
        f"duration stats (ms): mean={np.mean(durations):.2f} median={np.median(durations):.2f} "
        f"p95={np.percentile(durations, 95):.2f} p99={np.percentile(durations, 99):.2f} "
        f"max={np.max(durations):.2f}",
    )

    # Overrun: duration exceeded the budget available at call time, as in
    # environment.py's poll_and_run_agents().
    overrun_indices = [i for i in range(n) if durations[i] > contexts[i]["budget_before_ms"]]
    common.log_print(
        log_file,
        f"real overruns (duration > budget_before, i.e. forced WAIT this step): "
        f"{len(overrun_indices)}/{n}",
    )
    for i in overrun_indices:
        c = contexts[i]
        common.log_print(
            log_file,
            f"  round={c['round']:>2} step={c['step']:>4} duration_ms={durations[i]:.1f} "
            f"budget_before_ms={c['budget_before_ms']:.1f} bomb_available={c['bomb_available']} "
            f"has_reachable_opponent={c['has_reachable_opponent']} has_bombing_target={c['has_bombing_target']} "
            f"current_tile_in_danger={c['current_tile_in_danger']} nearest_threat_timer={c['nearest_threat_timer']} "
            f"n_legal_pre_mask={c['n_legal_pre_mask']} crates_remaining={c['crates_remaining']}",
        )

    spike_indices = [i for i in range(n) if durations[i] >= 300]
    common.log_print(log_file, f"\nsteps with duration>=300ms: {len(spike_indices)}/{n}")

    def rate(key, pred):
        matched = sum(1 for i in spike_indices if pred(contexts[i]))
        baseline = sum(1 for i in range(n) if pred(contexts[i]))
        return matched, len(spike_indices), baseline, n

    for label, pred in [
        ("bomb_available=True", lambda c: c["bomb_available"]),
        ("has_reachable_opponent=True", lambda c: c["has_reachable_opponent"]),
        ("has_bombing_target=True", lambda c: c["has_bombing_target"]),
        ("current_tile_in_danger=True", lambda c: c["current_tile_in_danger"]),
        ("nearest_threat_timer is not None", lambda c: c["nearest_threat_timer"] is not None),
        ("n_legal_pre_mask>=4", lambda c: c["n_legal_pre_mask"] >= 4),
    ]:
        m, sp_n, base, tot = rate(label, pred)
        common.log_print(
            log_file,
            f"  among spikes: {label} in {m}/{sp_n} ({100*m/sp_n if sp_n else float('nan'):.1f}%) "
            f"vs overall base rate {base}/{tot} ({100*base/tot:.1f}%)",
        )

    if spike_indices:
        common.log_print(log_file, "\n-- individual spike details (>=300ms) --")
        for i in spike_indices:
            c = contexts[i]
            common.log_print(
                log_file,
                f"  round={c['round']:>2} step={c['step']:>4} duration_ms={durations[i]:.1f} "
                f"bomb_available={c['bomb_available']} has_reachable_opponent={c['has_reachable_opponent']} "
                f"has_bombing_target={c['has_bombing_target']} current_tile_in_danger={c['current_tile_in_danger']} "
                f"nearest_threat_timer={c['nearest_threat_timer']} n_legal_pre_mask={c['n_legal_pre_mask']} "
                f"crates_remaining={c['crates_remaining']}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
