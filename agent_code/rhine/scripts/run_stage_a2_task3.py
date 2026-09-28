"""Task 3/4 training driver, structured like run_stage_a2_task2.py (training
bursts -> periodic eval -> final full evaluation), for the `classic` scenario
with opponents.

Training always runs to --total-timesteps. A snapshot is saved every
--eval-interval-rounds, and the delivered checkpoint is chosen from all
snapshots by _select_best_checkpoint().

Reward/PPO settings come from config.py defaults, which match Task 2's final
delivered configuration; no overrides are applied here. The Task 3 kill terms
(TRAINING_KILLED_OPPONENT_REWARD / TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY)
are training-only and independent of the real scoring ratio.

Opponent composition per --stage (curriculum table):
  A: 1x peaceful_agent
  B: 3x peaceful_agent
  C: 1x coin_collector_agent
  D: 3x coin_collector_agent
  E: 1x peaceful_agent + 2x coin_collector_agent (cancelled, kept for reference)
  TASK4: 3x rule_based_agent (trained from scratch)

run_full_evaluation() reports opponent_kills_mean but not
got_killed_by_opponent_rate, score_mean or score_total.

Usage (one process per stage):
    nohup python -m agent_code.rhine.scripts.run_stage_a2_task3 --stage A \
        > /dev/null 2>&1 &
"""
import argparse
import shutil
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_bombing_cost_benefit import (
    is_optimal_bombing_placement,
    placement_missed_opportunity,
)
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state

SCENARIO = "classic"

# Curriculum table: opponent composition per training stage.
STAGE_OPPONENTS = {
    "A": ["peaceful_agent"],
    "B": ["peaceful_agent", "peaceful_agent", "peaceful_agent"],
    "C": ["coin_collector_agent"],
    "D": ["coin_collector_agent", "coin_collector_agent", "coin_collector_agent"],
    "E": ["peaceful_agent", "coin_collector_agent", "coin_collector_agent"],
    "TASK4": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
}



# Window of the diagnostic-only _is_stabilized() check, logged to the CSV's
# window_* columns; it never stops training.
STABILIZATION_WINDOW = 5


def _is_stabilized(recent):
    if len(recent) < STABILIZATION_WINDOW:
        return False, {}
    window = recent[-STABILIZATION_WINDOW:]
    crates = [r["crates_destroyed_mean"] for r in window]
    waits = [r["wait_fraction_mean"] for r in window]
    completions = [r["completion_rate"] for r in window]

    crates_range = max(crates) - min(crates)
    crates_mean = sum(crates) / STABILIZATION_WINDOW
    stable_crates = crates_range <= max(5.0, 0.25 * crates_mean)

    wait_range = max(waits) - min(waits)
    stable_wait = wait_range <= 0.20

    completion_range = max(completions) - min(completions)
    stable_completion = completion_range <= 0.20

    detail = {
        "crates_range": crates_range, "crates_mean": crates_mean, "stable_crates": stable_crates,
        "wait_range": wait_range, "stable_wait": stable_wait,
        "completion_range": completion_range, "stable_completion": stable_completion,
    }
    return (stable_crates and stable_wait and stable_completion), detail


LOSS_STAT_KEYS = [
    "policy_loss_mean", "policy_loss_min", "policy_loss_max",
    "value_loss_mean", "value_loss_min", "value_loss_max",
    "entropy_mean", "entropy_min", "entropy_max",
    "gradient_norm_mean", "gradient_norm_min", "gradient_norm_max",
    "explained_variance_mean", "explained_variance_min", "explained_variance_max",
]


CHECKPOINT_SELECTION_SHORTLIST = 5  # Top-N eval points by score_mean considered.


def _select_best_checkpoint(history: list) -> dict:
    """Picks the delivered checkpoint from this run's periodic-eval history
    (one dict per eval point: round, snapshot_path, score_mean,
    self_kill_rate, got_killed_by_opponent_rate, oscillation_fraction,
    completion_rate).

    Shortlists the CHECKPOINT_SELECTION_SHORTLIST highest score_mean points,
    then drops those whose self_kill_rate, got_killed_by_opponent_rate or
    oscillation_fraction is a clear outlier against the run's median, so a
    noisy high score riding on degraded behavior is not picked.
    completion_rate is reported but not filtered on, since fewer completions
    can reflect legitimate hunting. Falls back to the full shortlist if every
    candidate is filtered out.
    """
    self_kill_med = float(np.median([h["self_kill_rate"] for h in history]))
    killed_by_opponent_med = float(np.median([h["got_killed_by_opponent_rate"] for h in history]))
    oscillation_med = float(np.median([h["oscillation_fraction"] for h in history]))
    completion_med = float(np.median([h["completion_rate"] for h in history]))

    self_kill_ceiling = self_kill_med + max(0.03, 0.5 * self_kill_med)
    killed_by_opponent_ceiling = killed_by_opponent_med + max(0.03, 0.5 * killed_by_opponent_med)
    oscillation_ceiling = oscillation_med + max(0.05, 0.5 * oscillation_med)

    shortlist = sorted(history, key=lambda h: h["score_mean"], reverse=True)
    shortlist = shortlist[:min(CHECKPOINT_SELECTION_SHORTLIST, len(shortlist))]

    def _failed_criteria(h: dict) -> list:
        failed = []
        if h["self_kill_rate"] > self_kill_ceiling:
            failed.append(f"self_kill_rate {h['self_kill_rate']:.3f} > ceiling {self_kill_ceiling:.3f}")
        if h["got_killed_by_opponent_rate"] > killed_by_opponent_ceiling:
            failed.append(
                f"got_killed_by_opponent_rate {h['got_killed_by_opponent_rate']:.3f} "
                f"> ceiling {killed_by_opponent_ceiling:.3f}"
            )
        if h["oscillation_fraction"] > oscillation_ceiling:
            failed.append(f"oscillation_fraction {h['oscillation_fraction']:.3f} > ceiling {oscillation_ceiling:.3f}")
        return failed

    shortlist_reasons = {h["round"]: _failed_criteria(h) for h in shortlist}
    qualified = [h for h in shortlist if not shortlist_reasons[h["round"]]]
    pool = qualified if qualified else shortlist
    chosen = max(pool, key=lambda h: h["score_mean"])
    return {
        "chosen": chosen, "shortlist": shortlist, "qualified": qualified,
        "shortlist_reasons": shortlist_reasons,
        "self_kill_med": self_kill_med, "oscillation_med": oscillation_med, "completion_med": completion_med,
        "killed_by_opponent_med": killed_by_opponent_med,
        "self_kill_ceiling": self_kill_ceiling, "oscillation_ceiling": oscillation_ceiling,
        "killed_by_opponent_ceiling": killed_by_opponent_ceiling,
    }


def run_full_evaluation(checkpoint_path, n_rounds: int, seed: int, opponents,
                         disable_think_time_limit: bool = False) -> dict:
    """Task 3 version of run_stage_a2_task2.run_full_evaluation(): same
    per-round trace/oscillation/bombing-quality logic, in the `classic`
    scenario with `opponents`.
    """
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False, opponents=opponents)
    agent = world.agents[0]

    episodes = []
    n_sustained = 0
    total_placements = 0
    flagged = 0
    extra_crates_when_flagged = []
    optimal_eligible = 0
    optimal_hits = 0
    self_kill_trace_paths = []

    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        round_index = world.round
        step_states = []
        trace = []
        recent_step_timings = deque(maxlen=5)
        while world.running:
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                step_states.append((
                    semantic.field_arr, semantic.self_pos, semantic.blocked, semantic.danger_offsets,
                    semantic.opponents,
                ))
                trace.append({
                    "step": state["step"],
                    "self_pos": semantic.self_pos,
                    "bomb_available": semantic.bomb_available,
                    "current_tile_in_danger": semantic.current_tile_in_danger,
                    "nearest_threat_timer": semantic.nearest_threat_timer,
                    "legal_actions": [cfg.ACTIONS[i] for i, m in enumerate(mask) if m],
                })
            step_number = state["step"] if state is not None else None
            if disable_think_time_limit:
                common._disable_think_time_limit(world)
            think_time_before_step = agent.available_think_time
            _t0 = time.perf_counter()
            world.do_step("WAIT")
            duration_ms = (time.perf_counter() - _t0) * 1000
            if step_number is not None:
                recent_step_timings.append((step_number, duration_ms, think_time_before_step))

        actions = list(world.replay["actions"][agent.name])
        for step_record, action in zip(trace, actions):
            step_record["chosen_action"] = action

        invalid_action_count = agent.statistics.get("invalid", 0)
        invalid_own_cause, invalid_contested_tile = common._invalid_action_breakdown(agent)
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
            completed=completed, invalid_action_count=invalid_action_count,
            invalid_action_own_cause_count=invalid_own_cause,
            invalid_action_contested_tile_count=invalid_contested_tile, wait_count=wait_count,
            immediate_reverse_count=reverse_count, crates_destroyed=crates_destroyed,
            bombs_dropped=bombs_dropped, self_kill=self_kill, opponent_kills=opponent_kills,
            got_killed_by_opponent=got_killed_by_opponent,
        )
        episodes.append(episode)

        if self_kill:
            context = {"scenario": SCENARIO, "seed": seed, "init_checkpoint": str(checkpoint_path), "opponents": opponents}
            self_kill_trace_paths.append(
                common.write_self_kill_trace(trace, episode, context, recent_step_timings)
            )

        run_len, _ = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            n_sustained += 1

        for (field_arr, self_pos, blocked, danger_offsets, opponents), action in zip(step_states, actions):
            if action != "BOMB":
                continue
            total_placements += 1
            extra = placement_missed_opportunity(field_arr, self_pos, blocked, cfg.BOMB_POWER)
            if extra is not None:
                flagged += 1
                extra_crates_when_flagged.append(extra)
            optimal = is_optimal_bombing_placement(
                field_arr, self_pos, blocked, cfg.BOMB_POWER, danger_offsets, opponents
            )
            if optimal is not None:
                optimal_eligible += 1
                if optimal:
                    optimal_hits += 1

    world.end()

    agg = common.aggregate_metrics(episodes)
    agg["oscillation_fraction"] = n_sustained / n_rounds if n_rounds else float("nan")
    agg["total_bomb_placements"] = total_placements
    agg["missed_opportunity_rate"] = (flagged / total_placements) if total_placements else float("nan")
    agg["missed_opportunity_avg_extra_crates"] = (
        float(np.mean(extra_crates_when_flagged)) if extra_crates_when_flagged else float("nan")
    )
    agg["bombing_target_optimal_eligible"] = optimal_eligible
    agg["bombing_target_optimal_rate"] = (
        (optimal_hits / optimal_eligible) if optimal_eligible else float("nan")
    )
    agg["self_kill_trace_paths"] = self_kill_trace_paths
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGE_OPPONENTS), default="A",
                         help="Opponent composition per the curriculum table above.")
    parser.add_argument("--total-timesteps", type=int, default=400_000,
                         help="Hard cap in environment steps; training always runs to this (no early "
                              "stop) and the delivered checkpoint is chosen from all periodic snapshots "
                              "by _select_best_checkpoint().")
    parser.add_argument("--eval-interval-rounds", type=int, default=100,
                         help="Raised from 50 (Stage A-C) to 100 as of the Stage D selection-protocol-v2 "
                              "change. Numbers from this "
                              "protocol are not directly comparable to Stage A-C's.")
    parser.add_argument("--eval-rounds", type=int, default=50,
                         help="Periodic in-loop evaluation round count driving both the diagnostic "
                              "_is_stabilized() window and checkpoint-selection score_mean/self_kill_rate/"
                              "oscillation_fraction/completion_rate. Raised from 20 to further reduce "
                              "per-checkpoint measurement noise now that checkpoint selection (not a "
                              "stop-training decision) depends directly on these per-eval-point values.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=3000,
                         help="Periodic in-loop evaluation seed, drives checkpoint selection only. Raised "
                              "from 1000 (Stage A-C) to 3000 as of the Stage D selection-protocol-v2 change "
                              "-- decoupled from "
                              "--final-eval-seed below; changing this does not affect this script's own "
                              "final evaluation.")
    parser.add_argument("--final-eval-rounds", type=int, default=100)
    parser.add_argument("--final-eval-seed", type=int, default=1000,
                         help="This script's own final evaluation (run_full_evaluation() below, skipped "
                              "entirely if --skip-final-eval is passed). Kept at 1000 regardless of "
                              "--eval-seed's value -- report data uses eval_seed=1000 with "
                              "an eval_seed=2000 cross-check, both run via run_full_evaluation() directly, "
                              "independent of this CLI flag.")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-steps", type=int, default=0)
    parser.add_argument("--resume-round", type=int, default=0)
    parser.add_argument("--rollout-steps-override", type=int, default=None,
                         help="Smoke-test-only override of PPOConfig.rollout_steps (config.py's stored "
                              "default, 4096, is left untouched for real training).")
    parser.add_argument(
        "--periodic-eval-disable-think-time-limit", action=argparse.BooleanOptionalAction, default=True,
        help="In-loop periodic eval (drives checkpoint-selection score_mean/self_kill_rate/"
             "oscillation_fraction). Default True: avoids CPU-contention "
             "timing noise from concurrent training/eval processes contaminating checkpoint selection.",
    )
    parser.add_argument(
        "--final-eval-disable-think-time-limit", action=argparse.BooleanOptionalAction, default=False,
        help="Final evaluation (the report data). Default False (real "
             "think-time limit); caller is responsible for running this pass "
             "without other training/eval processes concurrently contending for CPU.",
    )
    parser.add_argument("--checkpoint-path-override", type=str, default=None,
                         help="Any informal/smoke verification of this script must use this to point at a "
                              "scratch path, never the real per-stage default (cfg.STAGE_A_CHECKPOINT for "
                              "--stage A, models/task3_stage_<x>.pt otherwise). Must be an ABSOLUTE path: "
                              "train.py's _save_checkpoint() runs inside agents.py's chdir-into-agent-"
                              "directory window, while this script's own "
                              "shutil.copy() snapshot step runs at this process's own cwd -- a relative "
                              "path resolves differently in the two contexts and fails one of them.")
    parser.add_argument("--enable-oscillation-breaker", action="store_true", default=False,
                         help="Sets cfg.ENABLE_OSCILLATION_BREAKER=True for this run's evaluation passes "
                              "only (train=False path; never affects the training rollout regardless of "
                              "when this is set -- see config.py's docstring). Known limitation: validated "
                              "only against Task 2's static hazards; its trigger logic needs re-evaluation "
                              "once opponents are present, not yet done. Default off.")
    parser.add_argument("--enable-no-bomb-when-cleared", action="store_true", default=False,
                         help="Sets cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED=True: masks out BOMB whenever no "
                              "crates, no surviving opponents, and no collectable coins remain. Applies "
                              "wherever mask_from_semantic() runs, training rollout included. Default off.")
    parser.add_argument(
        "--skip-final-eval", action="store_true", default=False,
        help="Stop after checkpoint selection, without running the final full evaluation. Lets multiple "
             "seeds train concurrently while keeping their "
             "final evaluations -- which need real think-time and no concurrent CPU contention -- run "
             "separately afterward via run_full_evaluation() directly.",
    )
    parser.add_argument(
        "--disable-periodic-eval", action="store_true", default=False,
        help="Skip the in-loop periodic evaluation entirely: no eval-derived CSV row, no "
             "checkpoint_history entry, no in-process _select_best_checkpoint() call, no final "
             "evaluation (implies --skip-final-eval regardless of that flag's own value). "
             "Snapshot saving is unaffected (already independent of periodic eval -- see "
             "the training loop) and a reduced training-only CSV/log row "
             "(train_reward_mean + loss stats, computed purely from the training burst itself) is "
             "still written every --eval-interval-rounds. For use with a companion out-of-process "
             "evaluator (run_task4_async_eval.py) that evaluates each snapshot independently, and an "
             "offline selection script (select_task4_checkpoint.py) that reconstructs "
             "_select_best_checkpoint()'s input from the evaluator's CSV instead. Writes a "
             "'<stage>_seed<N>.done' sentinel file into the snapshot directory once the training loop "
             "exits, so the evaluator knows when to stop watching. The training loop itself "
             "(common.run_episodes(train=True, ...)) is completely unchanged either way.",
    )
    args = parser.parse_args()
    if args.resume_from:
        args.resume_from = str(Path(args.resume_from).resolve())
    cfg.ENABLE_OSCILLATION_BREAKER = args.enable_oscillation_breaker
    if args.enable_no_bomb_when_cleared:
        cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True

    # All reward/PPO settings come from config.py defaults.
    reward_override = {}
    ppo_override = {}

    opponents = STAGE_OPPONENTS[args.stage]

    label = f"Stage {args.stage} ({'+'.join(opponents) if opponents else 'no opponents'})"
    log_file, log_path = common.open_log_file(f"task3_stage_{args.stage.lower()}_seed{args.seed}")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Task 3 {label} -- log file: {log_path}")
    common.log_print(log_file, f"metrics CSV: {csv_path}")
    common.log_print(log_file, f"args: {vars(args)}")
    common.log_print(log_file, f"scenario: {SCENARIO}")
    common.log_print(log_file, f"opponents: {opponents}")
    common.log_print(
        log_file,
        "config sanity check (all read from config.py defaults, no script-level overrides): "
        f"n_features_active()={cfg.n_features_active()} "
        f"STAGE_A_CHECKPOINT={cfg.STAGE_A_CHECKPOINT} "
        f"ENABLE_BOMBING_TARGET_SAFETY_FILTER={cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER} "
        f"ENABLE_BOMBING_TARGET_SCORING_FORMULA={cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA} "
        f"ENABLE_BOMBING_PROGRESS_SHAPING={cfg.REWARD_CONFIG.ENABLE_BOMBING_PROGRESS_SHAPING} "
        f"ENABLE_STALL_PENALTY={cfg.REWARD_CONFIG.ENABLE_STALL_PENALTY} "
        f"ENABLE_STALL_PENALTY_V2={cfg.REWARD_CONFIG.ENABLE_STALL_PENALTY_V2} "
        f"normalize_returns={cfg.PPO_CONFIG.normalize_returns} "
        f"ENABLE_OSCILLATION_BREAKER={cfg.ENABLE_OSCILLATION_BREAKER} "
        f"ENABLE_NO_BOMB_WHEN_BOARD_CLEARED={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED}",
    )

    if args.checkpoint_path_override:
        # Absolute path: it is used both from this process's cwd and inside the
        # framework's chdir into the agent directory.
        checkpoint_path = Path(args.checkpoint_path_override).resolve()
    else:
        checkpoint_path = (
            cfg.STAGE_A_CHECKPOINT if args.stage == "A" else common.MODELS_DIR / f"task3_stage_{args.stage.lower()}.pt"
        )
    common.log_print(log_file, f"checkpoint_path: {checkpoint_path}")
    snapshot_dir = checkpoint_path.parent / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.resume_steps
    round_idx = args.resume_round
    burst_seed = args.seed + (args.resume_round // args.eval_interval_rounds if args.resume_round else 0)
    init_ckpt = args.resume_from

    if args.resume_from:
        common.log_print(log_file, f"Resuming from {args.resume_from} at round={round_idx}, total_steps={total_steps}")
    common.log_print(
        log_file,
        f"Training to hard cap --total-timesteps={args.total_timesteps} (no early stop); a snapshot is "
        f"saved every --eval-interval-rounds={args.eval_interval_rounds} rounds, and the delivered "
        f"checkpoint is chosen from all snapshots by _select_best_checkpoint() once the cap is hit.",
    )

    recent_metrics = []
    checkpoint_history = []
    stabilized = False

    while total_steps < args.total_timesteps:
        train_episodes, train_world = common.run_episodes(
            n_rounds=args.eval_interval_rounds, scenario=SCENARIO, seed=burst_seed,
            train=True, init_checkpoint=init_ckpt, save_checkpoint=str(checkpoint_path),
            reward_override=reward_override, ppo_override=ppo_override,
            steps_already_trained=total_steps, rollout_steps_override=args.rollout_steps_override,
            opponents=opponents,
        )
        train_reward_mean = common.training_reward_mean(train_world)
        loss_stats = common.training_loss_stats(train_world)
        total_steps += sum(ep.steps for ep in train_episodes)
        round_idx += args.eval_interval_rounds
        burst_seed += 1
        init_ckpt = str(checkpoint_path)

        snapshot_path = snapshot_dir / f"task3_stage_{args.stage.lower()}_seed{args.seed}_round{round_idx}.pt"
        shutil.copy(checkpoint_path, snapshot_path)

        if args.disable_periodic_eval:
            train_row = {
                "round": round_idx, "total_steps": total_steps,
                "train_reward_mean": train_reward_mean if train_reward_mean is not None else float("nan"),
                "invalid_action_own_cause_count_total": sum(
                    ep.invalid_action_own_cause_count for ep in train_episodes
                ),
            }
            for key in LOSS_STAT_KEYS:
                train_row[key] = loss_stats[key] if loss_stats is not None else float("nan")
            common.write_metrics_csv_row(csv_path, train_row)
            common.log_print(
                log_file,
                f"[train @ round {round_idx}, step {total_steps}] "
                f"train_reward_mean={train_row['train_reward_mean']} "
                f"value_loss_mean={train_row['value_loss_mean']} "
                f"entropy_mean={train_row['entropy_mean']} "
                f"gradient_norm_mean={train_row['gradient_norm_mean']} "
                f"invalid_action_own_cause_count_total={train_row['invalid_action_own_cause_count_total']} "
                f"snapshot={snapshot_path.name}",
            )
            continue

        eval_episodes, _, self_kill_trace_paths = common.run_evaluation_with_self_kill_tracing(
            n_rounds=args.eval_rounds, scenario=SCENARIO, seed=args.eval_seed,
            init_checkpoint=str(checkpoint_path),
            disable_think_time_limit=args.periodic_eval_disable_think_time_limit,
            opponents=opponents,
        )
        agg = common.aggregate_metrics(eval_episodes)
        agg["round"] = round_idx
        agg["total_steps"] = total_steps
        agg["train_reward_mean"] = train_reward_mean if train_reward_mean is not None else float("nan")
        n_sustained = sum(
            1 for ep in eval_episodes
            if ep.actions and longest_oscillation_run(ep.actions)[0] >= OSCILLATION_THRESHOLD
        )
        agg["oscillation_fraction"] = n_sustained / len(eval_episodes) if eval_episodes else float("nan")
        for key in LOSS_STAT_KEYS:
            agg[key] = loss_stats[key] if loss_stats is not None else float("nan")

        # Diagnostic windowed-range observation; not a stopping decision.
        recent_metrics.append({
            "round": round_idx,
            "crates_destroyed_mean": agg["crates_destroyed_mean"],
            "wait_fraction_mean": agg["wait_fraction_mean"],
            "completion_rate": agg["completion_rate"],
        })
        stabilized, detail = _is_stabilized(recent_metrics)
        agg["window_crates_range"] = detail.get("crates_range", float("nan"))
        agg["window_wait_range"] = detail.get("wait_range", float("nan"))
        agg["window_completion_range"] = detail.get("completion_range", float("nan"))
        common.write_metrics_csv_row(csv_path, agg)

        checkpoint_history.append({
            "round": round_idx,
            "snapshot_path": snapshot_path,
            "score_mean": agg["score_mean"],
            "self_kill_rate": agg["self_kill_rate"],
            "got_killed_by_opponent_rate": agg["got_killed_by_opponent_rate"],
            "oscillation_fraction": agg["oscillation_fraction"],
            "completion_rate": agg["completion_rate"],
        })

        common.log_print(
            log_file,
            f"[eval @ round {round_idx}, step {total_steps}] "
            f"score_mean={agg['score_mean']:.2f} "
            f"completion_rate={agg['completion_rate']:.2f} "
            f"self_kill_rate={agg['self_kill_rate']:.2f} "
            f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.2f} "
            f"opponent_kills_mean={agg['opponent_kills_mean']:.2f} "
            f"oscillation_fraction={agg['oscillation_fraction']:.3f} "
            f"crates_destroyed_mean={agg['crates_destroyed_mean']:.2f} "
            f"wait_fraction={agg['wait_fraction_mean']:.3f} "
            f"train_reward_mean={agg['train_reward_mean']} "
            f"value_loss_mean={agg['value_loss_mean']} "
            f"entropy_mean={agg['entropy_mean']} "
            f"snapshot={snapshot_path.name}",
        )
        if detail:
            common.log_print(
                log_file,
                f"  [8-pt window @ round {round_idx}] crates_range={detail['crates_range']:.2f} "
                f"(mean={detail['crates_mean']:.2f}, thresh={max(5.0, 0.25*detail['crates_mean']):.2f}) "
                f"wait_range={detail['wait_range']:.3f} completion_range={detail['completion_range']:.3f} "
                f"-> stabilized={stabilized} (informational only, does not stop training)",
            )
        if self_kill_trace_paths:
            common.log_print(
                log_file,
                f"  !! self-kill occurred ({len(self_kill_trace_paths)} round(s)) -- "
                f"trace(s): {[str(p) for p in self_kill_trace_paths]}, "
                f"checkpoint snapshot for replay: {snapshot_path}",
            )

    if args.disable_periodic_eval:
        done_marker = snapshot_dir / f"task3_stage_{args.stage.lower()}_seed{args.seed}.done"
        done_marker.write_text(f"total_steps={total_steps}\nlast_round={round_idx}\n")
        common.log_print(
            log_file,
            f"Reached hard cap (step {total_steps}). --disable-periodic-eval set: skipping in-process "
            "checkpoint selection and final evaluation -- run the companion async evaluator "
            "(run_task4_async_eval.py) and offline selection script (select_task4_checkpoint.py) "
            f"instead. Wrote done marker: {done_marker}",
        )
        log_file.close()
        common.ring_bell()
        return

    common.log_print(log_file, f"Reached hard cap (step {total_steps}). Selecting delivered checkpoint...")
    selection = _select_best_checkpoint(checkpoint_history)
    best = selection["chosen"]
    common.log_print(
        log_file,
        f"Checkpoint selection: shortlisted top {len(selection['shortlist'])} eval points by score_mean; "
        f"quality ceilings (this run's own median +/- tolerance): "
        f"self_kill_rate<={selection['self_kill_ceiling']:.3f} (med={selection['self_kill_med']:.3f}) "
        f"got_killed_by_opponent_rate<={selection['killed_by_opponent_ceiling']:.3f} "
        f"(med={selection['killed_by_opponent_med']:.3f}) "
        f"oscillation_fraction<={selection['oscillation_ceiling']:.3f} (med={selection['oscillation_med']:.3f}); "
        f"completion_rate not used as a filter criterion (med={selection['completion_med']:.3f}, reported only); "
        f"{len(selection['qualified'])}/{len(selection['shortlist'])} candidates passed the quality filter.",
    )
    for h in selection["shortlist"]:
        reasons = selection["shortlist_reasons"][h["round"]]
        status = "PASS" if not reasons else "FILTERED (" + "; ".join(reasons) + ")"
        common.log_print(
            log_file,
            f"  candidate round={h['round']} score_mean={h['score_mean']:.2f} "
            f"self_kill_rate={h['self_kill_rate']:.3f} "
            f"got_killed_by_opponent_rate={h['got_killed_by_opponent_rate']:.3f} "
            f"oscillation_fraction={h['oscillation_fraction']:.3f} -> {status}",
        )
    common.log_print(
        log_file,
        f"Selected round={best['round']} score_mean={best['score_mean']:.2f} "
        f"self_kill_rate={best['self_kill_rate']:.3f} "
        f"got_killed_by_opponent_rate={best['got_killed_by_opponent_rate']:.3f} "
        f"oscillation_fraction={best['oscillation_fraction']:.3f} "
        f"completion_rate={best['completion_rate']:.3f} (last trained round was {round_idx}).",
    )
    if best["snapshot_path"] != checkpoint_path:
        shutil.copy(best["snapshot_path"], checkpoint_path)
        common.log_print(log_file, f"Overwrote {checkpoint_path} with selected snapshot {best['snapshot_path']}")

    common.plot_training_curves(
        csv_path, png_path, x_col="total_steps",
        y_cols=[
            "score_mean", "completion_rate", "self_kill_rate", "got_killed_by_opponent_rate",
            "opponent_kills_mean", "oscillation_fraction", "crates_destroyed_mean",
            "train_reward_mean", "value_loss_mean", "entropy_mean",
        ],
        title=f"Task 3 {label}",
    )
    common.log_print(log_file, f"Task 3 {label} training loop finished. Chart: {png_path}")

    if args.skip_final_eval:
        common.log_print(
            log_file,
            "--skip-final-eval set: skipping final full evaluation. Run it separately, alone on the "
            "CPU, via run_full_evaluation() once no other training/eval process is running.",
        )
        log_file.close()
        common.ring_bell()
        return

    common.log_print(
        log_file,
        f"Running final full evaluation ({args.final_eval_rounds} rounds, seed={args.final_eval_seed})...",
    )
    final = run_full_evaluation(
        checkpoint_path, args.final_eval_rounds, args.final_eval_seed, opponents,
        disable_think_time_limit=args.final_eval_disable_think_time_limit,
    )
    common.log_print(
        log_file,
        f"=== FINAL ({label}, round {round_idx}, step {total_steps}) ===\n"
        f"  score_mean={final['score_mean']:.2f} score_total={final['score_total']:.2f}\n"
        f"  oscillation_fraction={final['oscillation_fraction']:.3f} "
        f"({int(final['oscillation_fraction'] * args.final_eval_rounds)}/{args.final_eval_rounds} rounds)\n"
        f"  missed_opportunity_rate={final['missed_opportunity_rate']:.3f} "
        f"(of {final['total_bomb_placements']} BOMB placements)\n"
        f"  crates_destroyed_mean={final['crates_destroyed_mean']:.2f} "
        f"bombing_target_optimal_rate={final['bombing_target_optimal_rate']:.3f}\n"
        f"  opponent_kills_mean={final['opponent_kills_mean']:.2f} "
        f"got_killed_by_opponent_rate={final['got_killed_by_opponent_rate']:.2f}\n"
        f"  wait_fraction_mean={final['wait_fraction_mean']:.3f} "
        f"completion_rate={final['completion_rate']:.2f} "
        f"self_kill_rate={final['self_kill_rate']:.2f}",
    )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
