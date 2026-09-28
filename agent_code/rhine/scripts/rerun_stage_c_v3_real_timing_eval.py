"""Task 3 Stage C v3 (34-dim) real-timing final evaluation of
task3_stage_c_v3_seed{0,1,2}.pt, run strictly sequentially with nothing else
running concurrently.

  Step 1 (timing profile): TIMING_ROUNDS rounds/seed; act() duration
  mean/median/p95/p99/max.

  Step 2 (final eval): FINAL_ROUNDS rounds/seed; standard score/self-kill/
  oscillation metrics plus the real forced-WAIT count.

run_seed() reimplements run_full_evaluation()'s loop so each step's act()
duration can be paired with the budget available at call time; a step is a
forced WAIT iff the duration exceeds that budget, as in environment.py's
poll_and_run_agents().

Uses the Stage A-E training configuration for bombing targets plus the
standard evaluation flags (oscillation breaker on, no-bomb-when-cleared on).
Usage:
  python -m agent_code.rhine.scripts.rerun_stage_c_v3_real_timing_eval
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state

OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
TIMING_ROUNDS = 15
FINAL_ROUNDS = 100
SEEDS = [0, 1, 2]


def run_seed(checkpoint_path, n_rounds, log_file, label):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    original_act = rhine_callbacks.act
    durations_ms = []

    def timed_act(self, game_state):
        t0 = time.perf_counter()
        result = original_act(self, game_state)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)
        return result

    rhine_callbacks.act = timed_act

    world = common.build_world(scenario="classic", seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    agent = world.agents[0]

    episodes = []
    n_sustained = 0
    budgets_ms = []  # Index-aligned with durations_ms.

    try:
        for _ in range(n_rounds):
            world.new_round()
            world.user_input = "WAIT"
            round_index = world.round
            while world.running:
                state = world.get_state_for_agent(agent)
                if state is not None:
                    budgets_ms.append(agent.available_think_time * 1000.0)
                world.do_step("WAIT")

            actions = list(world.replay["actions"][agent.name])
            invalid_action_count = agent.statistics.get("invalid", 0)
            coins_collected = agent.statistics.get("coins", 0)
            crates_destroyed = agent.statistics.get("crates", 0)
            bombs_dropped = agent.statistics.get("bombs", 0)
            self_kill = agent.statistics.get("suicides", 0) > 0
            opponent_kills = agent.statistics.get("kills", 0)
            got_killed_by_opponent = agent.dead and not self_kill
            wait_count = actions.count("WAIT")
            reverse_count = sum(
                1 for prev, cur in zip(actions, actions[1:]) if common.OPPOSITE_DIRECTION.get(prev) == cur
            )
            completed = common.default_success_fn(world)
            episode = common.EpisodeMetrics(
                round_index=round_index, steps=world.step, coins_collected=coins_collected,
                completed=completed, invalid_action_count=invalid_action_count, wait_count=wait_count,
                immediate_reverse_count=reverse_count, crates_destroyed=crates_destroyed,
                bombs_dropped=bombs_dropped, self_kill=self_kill, opponent_kills=opponent_kills,
                got_killed_by_opponent=got_killed_by_opponent,
            )
            episodes.append(episode)

            run_len, _ = longest_oscillation_run(actions)
            if run_len >= OSCILLATION_THRESHOLD:
                n_sustained += 1
        world.end()
    finally:
        rhine_callbacks.act = original_act

    agg = common.aggregate_metrics(episodes)
    agg["oscillation_fraction"] = n_sustained / n_rounds if n_rounds else float("nan")

    n = min(len(durations_ms), len(budgets_ms))
    durations = np.array(durations_ms[:n])
    budgets = np.array(budgets_ms[:n])
    overrun_count = int(np.sum(durations > budgets))

    common.log_print(
        log_file,
        f"  [{label}] n_rounds={n_rounds} n_steps={n} "
        f"act()ms: mean={durations.mean():.2f} median={np.median(durations):.2f} "
        f"p95={np.percentile(durations, 95):.2f} p99={np.percentile(durations, 99):.2f} max={durations.max():.2f} "
        f"real_forced_wait_overruns={overrun_count}/{n}",
    )
    return agg, durations, overrun_count, n


def main():
    log_file, log_path = common.open_log_file("stage_c_v3_real_timing_final_eval")
    common.log_print(log_file, f"Stage C v3 (34-dim) real-timing final eval -- log: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} eval_seed={EVAL_SEED} "
                                f"timing_rounds={TIMING_ROUNDS} final_rounds={FINAL_ROUNDS} "
                                f"disable_think_time_limit=False (real timing), sequential (no concurrency)")

    common.log_print(log_file, "\n" + "=" * 78)
    common.log_print(log_file, f"STEP 1: quick act() timing profile ({TIMING_ROUNDS} rounds/seed)")
    common.log_print(log_file, "=" * 78)
    for seed in SEEDS:
        ckpt = common.MODELS_DIR / f"task3_stage_c_v3_seed{seed}.pt"
        run_seed(ckpt, TIMING_ROUNDS, log_file, f"seed{seed} timing-profile")

    common.log_print(log_file, "\n" + "=" * 78)
    common.log_print(log_file, f"STEP 2: official final evaluation ({FINAL_ROUNDS} rounds/seed)")
    common.log_print(log_file, "=" * 78)
    per_seed_summary = []
    for seed in SEEDS:
        ckpt = common.MODELS_DIR / f"task3_stage_c_v3_seed{seed}.pt"
        common.log_print(log_file, f"\n--- seed{seed} ({ckpt.name}) ---")
        agg, durations, overrun_count, n = run_seed(ckpt, FINAL_ROUNDS, log_file, f"seed{seed} final-eval")
        common.log_print(
            log_file,
            f"  score_mean={agg['score_mean']:.2f} score_total={agg['score_total']:.2f}\n"
            f"  opponent_kills_mean={agg['opponent_kills_mean']:.3f} "
            f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.3f} "
            f"self_kill_rate={agg['self_kill_rate']:.3f}\n"
            f"  completion_rate={agg['completion_rate']:.3f} "
            f"oscillation_fraction={agg['oscillation_fraction']:.3f} "
            f"({int(agg['oscillation_fraction'] * FINAL_ROUNDS)}/{FINAL_ROUNDS})\n"
            f"  episode_length_mean={agg['episode_length_mean']:.2f} "
            f"invalid_action_count_total={agg.get('invalid_action_count_total')}",
        )
        per_seed_summary.append({
            "seed": seed, "score_mean": agg["score_mean"], "score_total": agg["score_total"],
            "opponent_kills_mean": agg["opponent_kills_mean"],
            "got_killed_by_opponent_rate": agg["got_killed_by_opponent_rate"],
            "self_kill_rate": agg["self_kill_rate"], "completion_rate": agg["completion_rate"],
            "oscillation_fraction": agg["oscillation_fraction"],
            "real_forced_wait_overruns": overrun_count, "n_steps": n,
        })

    common.log_print(log_file, "\n" + "=" * 78)
    common.log_print(log_file, "3-seed mean +/- std (ddof=1)")
    common.log_print(log_file, "=" * 78)
    keys = ["score_mean", "score_total", "opponent_kills_mean", "got_killed_by_opponent_rate",
            "self_kill_rate", "completion_rate", "oscillation_fraction"]
    for key in keys:
        values = [s[key] for s in per_seed_summary]
        common.log_print(log_file, f"  {key}: {np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f} (values={values})")

    total_overruns = sum(s["real_forced_wait_overruns"] for s in per_seed_summary)
    total_steps = sum(s["n_steps"] for s in per_seed_summary)
    common.log_print(
        log_file,
        f"\n  total real forced-WAIT overruns (Step 2 only, pooled across 3 seeds): "
        f"{total_overruns}/{total_steps} "
        f"(per-seed: {[s['real_forced_wait_overruns'] for s in per_seed_summary]})",
    )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
