"""Task 2 Stage A2: oscillation follow-up ablation with B2 (unified progress
shaping) as the baseline.

2x2 factorial over two independently toggleable reward mechanisms:
  B2    : ENABLE_BOMBING_PROGRESS_SHAPING=True, stall penalty off, crate-no-coin bonus off
  B2+S  : + V1 stall penalty (ENABLE_STALL_PENALTY=True)
  B2+C  : + CRATE_DESTROYED_REWARD_NO_COIN when no coin is reachable (ENABLE_CRATE_NO_COIN_BONUS=True)
  B2+SC : both

Follow-up single group:
  B2+S2 : ENABLE_BOMBING_PROGRESS_SHAPING=True + ENABLE_STALL_PENALTY_V2=True,
          V1 stall penalty and crate-no-coin bonus off.

Each group trains from a fresh random init until _is_stabilized() holds
(--stabilize), train seed 0, eval seed 1000. Periodic evaluation uses the
lightweight aggregate_metrics() pass; after stabilization (or the step cap),
run_full_evaluation() runs once, adding the diagnostic-only oscillation and
bombing-quality metrics in the same pass.

Usage (one process per group):
    nohup python -m agent_code.rhine.scripts.run_stage_a2_task2 --ablation B2 \
        --stabilize > /dev/null 2>&1 &
    (repeat for B2S, B2C, B2SC)
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

ABLATION_REWARD_OVERRIDES = {
    # V2 is set explicitly so these groups keep it off regardless of config defaults.
    "B2":   {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": False},
    "B2S":  {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": True,  "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": False},
    "B2C":  {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": True, "ENABLE_STALL_PENALTY_V2": False},
    "B2SC": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": True,  "ENABLE_CRATE_NO_COIN_BONUS": True, "ENABLE_STALL_PENALTY_V2": False},
    "B2S2": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2C": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": True, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config as "B2S2" under a separate key/checkpoint so both can
    # run in parallel; the safety filter and scoring formula are disabled via
    # CLI flags to isolate entropy/LR decay.
    "B2S2PURE": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; separate checkpoint for a short diagnostic re-run.
    "B2S2PUREDIAG": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; two seeds of the entropy_coef=0.05 check, each with
    # its own checkpoint.
    "B2S2ENT05SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2ENT05SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; two seeds of the normalize_returns=True check with the
    # default entropy_coef, so return normalization is the only change vs
    # B2S2PUREDIAG.
    "B2S2RETNORMSEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2RETNORMSEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; the normalize_returns=True run continued to
    # --stabilize convergence, with separate keys so the shorter runs'
    # checkpoints are kept.
    "B2S2RETNORMLONGSEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2RETNORMLONGSEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; short resumed continuations from a pre-collapse
    # snapshot with a narrower evaluation interval. Separate keys so the
    # source checkpoints are never touched.
    "B2S2RETNORMLONGSEED0_W500550": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2RETNORMLONGSEED1_W700750": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2RETNORMSEED1_W350400": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Same reward config; GAE lambda=0.90 check, otherwise as B2S2RETNORMSEED0/1.
    "B2S2RETNORMGAE90SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2RETNORMGAE90SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Multi-seed base-config selection plus the remaining safety-filter x
    # scoring-formula cells. All share this reward config; only the CLI flags
    # differ.
    **{f"B2S2NORMFINAL_SEED{i}": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True} for i in range(5)},
    "B2S2NORMFILTER_SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMFILTER_SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORING_SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORING_SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    # Additional scoring-only seeds, plus scoring-only with the stall-history
    # feature.
    "B2S2NORMSCORING_SEED2": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORING_SEED3": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORING_SEED4": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORINGH_SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMSCORINGH_SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMBOTH_SEED0": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
    "B2S2NORMBOTH_SEED1": {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": False, "ENABLE_CRATE_NO_COIN_BONUS": False, "ENABLE_STALL_PENALTY_V2": True},
}
ABLATION_LABELS = {
    "B2": "B2", "B2S": "B2+S", "B2C": "B2+C", "B2SC": "B2+SC", "B2S2": "B2+S2", "B2S2C": "B2+S2+C",
    "B2S2PURE": "B2+S2 (pure)", "B2S2PUREDIAG": "B2+S2 (pure, monitoring-metrics diag)",
    "B2S2ENT05SEED0": "B2+S2 (pure, entropy_coef=0.05, seed0)",
    "B2S2ENT05SEED1": "B2+S2 (pure, entropy_coef=0.05, seed1)",
    "B2S2RETNORMSEED0": "B2+S2 (pure, normalize_returns=True, seed0)",
    "B2S2RETNORMSEED1": "B2+S2 (pure, normalize_returns=True, seed1)",
    "B2S2RETNORMLONGSEED0": "B2+S2 (pure, normalize_returns=True, seed0, stabilize)",
    "B2S2RETNORMLONGSEED1": "B2+S2 (pure, normalize_returns=True, seed1, stabilize)",
    "B2S2RETNORMLONGSEED0_W500550": "B2+S2 (pure, normalize_returns=True, seed0, fine-grained window 500-560)",
    "B2S2RETNORMLONGSEED1_W700750": "B2+S2 (pure, normalize_returns=True, seed1, fine-grained window 700-760)",
    "B2S2RETNORMSEED1_W350400": "B2+S2 (pure, normalize_returns=True, seed1(200k), fine-grained window 350-410)",
    "B2S2RETNORMGAE90SEED0": "B2+S2 (pure, normalize_returns=True, gae_lambda=0.90, seed0)",
    "B2S2RETNORMGAE90SEED1": "B2+S2 (pure, normalize_returns=True, gae_lambda=0.90, seed1)",
    **{f"B2S2NORMFINAL_SEED{i}": f"B2+S2 (normalize_returns=True, filter=off/scoring=off, seed{i})" for i in range(5)},
    "B2S2NORMFILTER_SEED0": "B2+S2 (normalize_returns=True, filter=ON/scoring=off, seed0)",
    "B2S2NORMFILTER_SEED1": "B2+S2 (normalize_returns=True, filter=ON/scoring=off, seed1)",
    "B2S2NORMSCORING_SEED0": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, seed0)",
    "B2S2NORMSCORING_SEED1": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, seed1)",
    "B2S2NORMSCORING_SEED2": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, seed2)",
    "B2S2NORMSCORING_SEED3": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, seed3)",
    "B2S2NORMSCORING_SEED4": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, seed4)",
    "B2S2NORMSCORINGH_SEED0": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, H feature, seed0)",
    "B2S2NORMSCORINGH_SEED1": "B2+S2 (normalize_returns=True, filter=off/scoring=ON, H feature, seed1)",
    "B2S2NORMBOTH_SEED0": "B2+S2 (normalize_returns=True, filter=ON/scoring=ON, seed0)",
    "B2S2NORMBOTH_SEED1": "B2+S2 (normalize_returns=True, filter=ON/scoring=ON, seed1)",
}


# Long enough not to mistake the tail of a collapse/recovery cycle for convergence.
STABILIZATION_WINDOW = 8


def _is_stabilized(recent):
    """Same heuristic as run_stage_a_task2.py's _is_stabilized() with a wider
    window; copied rather than imported so the two scripts stay independent.
    """
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
    stable_wait = wait_range <= 0.15

    completion_range = max(completions) - min(completions)
    stable_completion = completion_range <= 0.15

    detail = {
        "crates_range": crates_range, "crates_mean": crates_mean, "stable_crates": stable_crates,
        "wait_range": wait_range, "stable_wait": stable_wait,
        "completion_range": completion_range, "stable_completion": stable_completion,
    }
    return (stable_crates and stable_wait and stable_completion), detail


def run_full_evaluation(checkpoint_path, n_rounds: int, seed: int, scenario: str = "loot-crate",
                         disable_think_time_limit: bool = False) -> dict:
    """One evaluation pass producing the standard metrics
    (common.aggregate_metrics), oscillation_fraction and the bombing-quality
    metrics, which all need the same per-round trace.

    Also records per-step legal/chosen actions and recent do_step() timings,
    and writes a self-kill trace (common.write_self_kill_trace) for every
    self-killing round. bombing_target_optimal_rate is reported in place of
    the deprecated bomb_efficiency (still in the returned dict).

    disable_think_time_limit: diagnostic opt-in, default False.
    """
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=scenario, seed=seed, train=False)
    agent = world.agents[0]

    episodes = []
    n_sustained = 0
    total_placements = 0
    flagged = 0
    extra_crates_when_flagged = []
    optimal_eligible = 0  # Placements where some crate-hitting candidate existed.
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
                step_states.append((semantic.field_arr, semantic.self_pos, semantic.blocked, semantic.danger_offsets))
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
                # Must be re-applied before every step.
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
        coins_collected = agent.statistics.get("coins", 0)
        crates_destroyed = agent.statistics.get("crates", 0)
        bombs_dropped = agent.statistics.get("bombs", 0)
        self_kill = agent.statistics.get("suicides", 0) > 0
        wait_count = actions.count("WAIT")
        reverse_count = sum(
            1 for prev, cur in zip(actions, actions[1:]) if common.OPPOSITE_DIRECTION.get(prev) == cur
        )
        completed = common.default_success_fn(world)
        episode = common.EpisodeMetrics(
            round_index=round_index, steps=world.step, coins_collected=coins_collected,
            completed=completed, invalid_action_count=invalid_action_count, wait_count=wait_count,
            immediate_reverse_count=reverse_count, crates_destroyed=crates_destroyed,
            bombs_dropped=bombs_dropped, self_kill=self_kill,
        )
        episodes.append(episode)

        if self_kill:
            context = {"scenario": scenario, "seed": seed, "init_checkpoint": str(checkpoint_path)}
            self_kill_trace_paths.append(
                common.write_self_kill_trace(trace, episode, context, recent_step_timings)
            )

        run_len, _ = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            n_sustained += 1

        for (field_arr, self_pos, blocked, danger_offsets), action in zip(step_states, actions):
            if action != "BOMB":
                continue
            total_placements += 1
            extra = placement_missed_opportunity(field_arr, self_pos, blocked, cfg.BOMB_POWER)
            if extra is not None:
                flagged += 1
                extra_crates_when_flagged.append(extra)
            optimal = is_optimal_bombing_placement(field_arr, self_pos, blocked, cfg.BOMB_POWER, danger_offsets)
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
    # Replaces the deprecated bomb_efficiency. The denominator counts only
    # placements with some crate target available, matching
    # is_optimal_bombing_placement()'s None convention.
    agg["bombing_target_optimal_eligible"] = optimal_eligible
    agg["bombing_target_optimal_rate"] = (
        (optimal_hits / optimal_eligible) if optimal_eligible else float("nan")
    )
    agg["self_kill_trace_paths"] = self_kill_trace_paths
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=list(ABLATION_REWARD_OVERRIDES), default="B2")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                         help="Hard safety cap in environment steps; --stabilize normally stops earlier.")
    parser.add_argument("--eval-interval-rounds", type=int, default=50)
    parser.add_argument("--eval-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--final-eval-rounds", type=int, default=100,
                         help="Round count for the one-time full evaluation (oscillation + "
                              "cost-benefit + standard metrics) run after stabilization.")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-steps", type=int, default=0)
    parser.add_argument("--resume-round", type=int, default=0)
    parser.add_argument("--stabilize", action="store_true", default=True,
                         help="Kept True by default for Stage A2: train each group to stabilization, "
                              "not a fixed step count.")
    parser.add_argument("--no-stabilize", dest="stabilize", action="store_false")
    parser.add_argument("--disable-bombing-target-safety-filter", action="store_true", default=False,
                         help="Reproduces the pre-fix candidate logic (bombing_target_info only checks "
                              "crates_in_blast, not escape safety) for a specific comparison run. Default "
                              "off (safety filter stays on, matching the fixed behavior).")
    parser.add_argument("--disable-bombing-target-scoring-formula", action="store_true", default=False,
                         help="Reproduces the nearest-distance-only candidate selection (bombing_target_info "
                              "picks the closest safety-filtered candidate, ignoring crate count) for a "
                              "specific comparison run. Default off (scoring formula stays on).")
    parser.add_argument("--enable-entropy-lr-decay", action="store_true", default=False,
                         help="Linear decay of entropy_coef (0.01->0.001) and learning_rate "
                              "(3e-4->3e-5) over training progress (steps_so_far / --total-timesteps). "
                              "Default off.")
    parser.add_argument("--enable-stall-history-feature", action="store_true", default=False,
                         help="Sets cfg.ENABLE_STALL_HISTORY_FEATURE=True for this run, switching the "
                              "model's input dimension from 20 to 21. Must be a fresh random-init run (no "
                              "--resume-from a 20-dim checkpoint) -- state_processing/features.py already "
                              "implement and test the 21st dim itself, this flag only toggles the existing "
                              "switch. Default off (config.py's stored default, unchanged).")
    parser.add_argument("--entropy-coef", type=float, default=None,
                         help="Diagnostic-only override of PPOConfig.entropy_coef via the existing "
                              "PPO_AGENT_PPO_OVERRIDE mechanism. Does not change config.py's stored "
                              "default (still 0.01) -- None (default) leaves it untouched.")
    parser.add_argument("--normalize-returns", action="store_true", default=False,
                         help="Diagnostic-only override of PPOConfig.normalize_returns via the existing "
                              "PPO_AGENT_PPO_OVERRIDE mechanism (return/value-loss running-mean/std "
                              "normalization). Does not change config.py's stored default (still False).")
    parser.add_argument("--gae-lambda", type=float, default=None,
                         help="Diagnostic-only override of PPOConfig.gae_lambda via the existing "
                              "PPO_AGENT_PPO_OVERRIDE mechanism. Does not change config.py's stored "
                              "default (still 0.95) -- None (default) leaves it untouched.")
    parser.add_argument("--disable-think-time-limit", action="store_true", default=False,
                         help="Training/diagnostic-only: overrides the framework's per-round think-time "
                              "budget to effectively unlimited during periodic and final evaluation, so "
                              "wait_fraction/oscillation_fraction/self_kill_rate aren't polluted by "
                              "CPU-load-dependent forced-WAIT overrides unrelated to actual policy "
                              "quality. Never touches settings.py/agents.py/environment.py and is never "
                              "exercised by the official submission path. Default off.")
    parser.add_argument("--enable-oscillation-breaker", action="store_true", default=False,
                         help="Sets cfg.ENABLE_OSCILLATION_BREAKER=True for this run: evaluation/"
                              "deployment-only action-mask refinement that masks 'return to previous tile' "
                              "when the agent has been confined to <=2 tiles for 4 steps and the current "
                              "tile is a through-passage (both the reverse direction and its opposite are "
                              "legal), leaving true dead ends untouched. Gated in callbacks.py's act() on "
                              "`not self.train`, so this only ever affects periodic/final evaluation "
                              "passes here, never the training rollout itself, regardless of when in a run "
                              "this flag is set. Task 2 (static-hazard) only -- see "
                              "config.ENABLE_OSCILLATION_BREAKER's docstring. Default off.")
    args = parser.parse_args()
    if args.resume_from:
        args.resume_from = str(Path(args.resume_from).resolve())
    if args.disable_bombing_target_safety_filter:
        cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER = False
    if args.disable_bombing_target_scoring_formula:
        cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = False
    if args.enable_stall_history_feature:
        cfg.ENABLE_STALL_HISTORY_FEATURE = True
    cfg.ENABLE_OSCILLATION_BREAKER = args.enable_oscillation_breaker
    ppo_override = {}
    if args.enable_entropy_lr_decay:
        ppo_override["enable_entropy_lr_decay"] = True
        ppo_override["total_timesteps"] = args.total_timesteps
    if args.entropy_coef is not None:
        ppo_override["entropy_coef"] = args.entropy_coef
    if args.normalize_returns:
        ppo_override["normalize_returns"] = True
    if args.gae_lambda is not None:
        ppo_override["gae_lambda"] = args.gae_lambda
    ppo_override = ppo_override or None

    label = ABLATION_LABELS[args.ablation]
    log_file, log_path = common.open_log_file(f"task2_stage_a2_{args.ablation}")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Task 2 Stage A2 ({label}) -- log file: {log_path}")
    common.log_print(log_file, f"metrics CSV: {csv_path}")
    common.log_print(log_file, f"args: {vars(args)}")
    common.log_print(log_file, f"reward_override: {ABLATION_REWARD_OVERRIDES[args.ablation]}")
    common.log_print(log_file, f"ppo_override: {ppo_override}")

    checkpoint_path = common.MODELS_DIR / f"task2_stage_a2_{args.ablation}.pt"
    snapshot_dir = common.MODELS_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    reward_override = ABLATION_REWARD_OVERRIDES[args.ablation]

    total_steps = args.resume_steps
    round_idx = args.resume_round
    burst_seed = args.seed + (args.resume_round // args.eval_interval_rounds if args.resume_round else 0)
    init_ckpt = args.resume_from

    if args.resume_from:
        common.log_print(log_file, f"Resuming from {args.resume_from} at round={round_idx}, total_steps={total_steps}")
    if args.stabilize:
        common.log_print(
            log_file,
            f"Stabilize mode: training until {STABILIZATION_WINDOW} consecutive eval points show small changes in "
            f"crates_destroyed_mean/wait_fraction_mean/completion_rate, --total-timesteps "
            f"({args.total_timesteps}) as a hard cap only.",
        )

    recent_metrics = []
    stabilized = False

    while total_steps < args.total_timesteps and (not args.stabilize or not stabilized):
        train_episodes, train_world = common.run_episodes(
            n_rounds=args.eval_interval_rounds, scenario="loot-crate", seed=burst_seed,
            train=True, init_checkpoint=init_ckpt, save_checkpoint=str(checkpoint_path),
            reward_override=reward_override,
            ppo_override=ppo_override, steps_already_trained=total_steps,
        )
        train_reward_mean = common.training_reward_mean(train_world)
        loss_stats = common.training_loss_stats(train_world)
        total_steps += sum(ep.steps for ep in train_episodes)
        round_idx += args.eval_interval_rounds
        burst_seed += 1
        init_ckpt = str(checkpoint_path)

        snapshot_path = snapshot_dir / f"task2_stage_a2_{args.ablation}_round{round_idx}.pt"
        shutil.copy(checkpoint_path, snapshot_path)

        eval_episodes, _, self_kill_trace_paths = common.run_evaluation_with_self_kill_tracing(
            n_rounds=args.eval_rounds, scenario="loot-crate", seed=args.eval_seed,
            init_checkpoint=str(checkpoint_path),
            disable_think_time_limit=args.disable_think_time_limit,
        )
        agg = common.aggregate_metrics(eval_episodes)
        agg["round"] = round_idx
        agg["total_steps"] = total_steps
        agg["train_reward_mean"] = train_reward_mean if train_reward_mean is not None else float("nan")
        # Derived from the episodes' action sequences, with the same oscillation
        # definition as run_full_evaluation() at the smaller periodic sample size.
        n_sustained = sum(
            1 for ep in eval_episodes
            if ep.actions and longest_oscillation_run(ep.actions)[0] >= OSCILLATION_THRESHOLD
        )
        agg["oscillation_fraction"] = n_sustained / len(eval_episodes) if eval_episodes else float("nan")
        loss_stat_keys = [
            "policy_loss_mean", "policy_loss_min", "policy_loss_max",
            "value_loss_mean", "value_loss_min", "value_loss_max",
            "entropy_mean", "entropy_min", "entropy_max",
            "gradient_norm_mean", "gradient_norm_min", "gradient_norm_max",
            "explained_variance_mean", "explained_variance_min", "explained_variance_max",
        ]
        for key in loss_stat_keys:
            agg[key] = loss_stats[key] if loss_stats is not None else float("nan")
        common.write_metrics_csv_row(csv_path, agg)

        common.log_print(
            log_file,
            f"[eval @ round {round_idx}, step {total_steps}] "
            f"completion_rate={agg['completion_rate']:.2f} "
            f"self_kill_rate={agg['self_kill_rate']:.2f} "
            f"oscillation_fraction={agg['oscillation_fraction']:.3f} "
            f"crates_destroyed_mean={agg['crates_destroyed_mean']:.2f} "
            f"bomb_efficiency={agg['bomb_efficiency']:.2f} "
            f"wait_fraction={agg['wait_fraction_mean']:.3f} "
            f"train_reward_mean={agg['train_reward_mean']} "
            f"policy_loss_mean={agg['policy_loss_mean']} (min={agg['policy_loss_min']}, max={agg['policy_loss_max']}) "
            f"value_loss_mean={agg['value_loss_mean']} (min={agg['value_loss_min']}, max={agg['value_loss_max']}) "
            f"entropy_mean={agg['entropy_mean']} (min={agg['entropy_min']}, max={agg['entropy_max']}) "
            f"gradient_norm_mean={agg['gradient_norm_mean']} (min={agg['gradient_norm_min']}, max={agg['gradient_norm_max']}) "
            f"explained_variance_mean={agg['explained_variance_mean']} "
            f"(min={agg['explained_variance_min']}, max={agg['explained_variance_max']}) "
            f"snapshot={snapshot_path.name}",
        )
        if self_kill_trace_paths:
            common.log_print(
                log_file,
                f"  !! self-kill occurred ({len(self_kill_trace_paths)} round(s)) -- "
                f"trace(s): {[str(p) for p in self_kill_trace_paths]}, "
                f"checkpoint snapshot for replay: {snapshot_path}",
            )

        if args.stabilize:
            recent_metrics.append({
                "round": round_idx,
                "crates_destroyed_mean": agg["crates_destroyed_mean"],
                "wait_fraction_mean": agg["wait_fraction_mean"],
                "completion_rate": agg["completion_rate"],
            })
            stabilized, detail = _is_stabilized(recent_metrics)
            if stabilized:
                common.log_print(
                    log_file,
                    f"STABILIZED at round {round_idx} (step {total_steps}): "
                    f"last 4 points -- crates_range={detail['crates_range']:.2f} "
                    f"(mean={detail['crates_mean']:.2f}), wait_range={detail['wait_range']:.3f}, "
                    f"completion_range={detail['completion_range']:.3f}. Stopping training loop.",
                )

    common.plot_training_curves(
        csv_path, png_path, x_col="total_steps",
        y_cols=[
            "completion_rate", "self_kill_rate", "oscillation_fraction", "crates_destroyed_mean",
            "bomb_efficiency", "train_reward_mean", "policy_loss_mean", "value_loss_mean",
            "entropy_mean", "gradient_norm_mean", "explained_variance_mean",
        ],
        title=f"Task 2 Stage A2 ({label})",
    )
    common.log_print(log_file, f"Task 2 Stage A2 ({label}) training loop finished. Chart: {png_path}")

    common.log_print(
        log_file,
        f"Running final full evaluation ({args.final_eval_rounds} rounds, seed={args.eval_seed}): "
        f"standard metrics + oscillation_fraction + cost-benefit bombing-quality metric...",
    )
    final = run_full_evaluation(
        checkpoint_path, args.final_eval_rounds, args.eval_seed,
        disable_think_time_limit=args.disable_think_time_limit,
    )
    common.log_print(
        log_file,
        f"=== FINAL ({label}, round {round_idx}, step {total_steps}) ===\n"
        f"  oscillation_fraction={final['oscillation_fraction']:.3f} "
        f"({int(final['oscillation_fraction'] * args.final_eval_rounds)}/{args.final_eval_rounds} rounds)\n"
        f"  missed_opportunity_rate={final['missed_opportunity_rate']:.3f} "
        f"(of {final['total_bomb_placements']} BOMB placements) "
        f"missed_opportunity_avg_extra_crates={final['missed_opportunity_avg_extra_crates']:.2f}\n"
        f"  crates_destroyed_mean={final['crates_destroyed_mean']:.2f} "
        f"bombing_target_optimal_rate={final['bombing_target_optimal_rate']:.3f} "
        f"(of {final['bombing_target_optimal_eligible']} eligible placements) "
        f"bombs_dropped_mean={final['bombs_dropped_mean']:.2f}\n"
        f"  wait_fraction_mean={final['wait_fraction_mean']:.3f} "
        f"completion_rate={final['completion_rate']:.2f} "
        f"self_kill_rate={final['self_kill_rate']:.2f}",
    )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
