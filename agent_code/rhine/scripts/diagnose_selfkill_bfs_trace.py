"""Read-only diagnostic: instrumented replay of Stage C self-kills, exposing
per-step mask internals (legal directions, N/required path counts,
per-direction conflicts, tie-break outcome).

Opponent randomness is pinned via _pin_opponent_reseed(), making a given
round reproducible across process launches. Runs from before this pin cannot
be reproduced, so a full census surveys an equivalent fresh population.

Usage:
  # Full census: every self-kill in a deterministic N-round eval.
  python -m agent_code.rhine.scripts.diagnose_selfkill_bfs_trace \\
      --checkpoint models/task3_stage_c_seed1.pt --breaker on \\
      --n-rounds 100 --eval-seed 1000

  # Re-inspect one specific round.
  python -m agent_code.rhine.scripts.diagnose_selfkill_bfs_trace \\
      --checkpoint models/task3_stage_c_seed1.pt --breaker on \\
      --target-round 9 --eval-seed 1000
"""
import argparse
import random as _random_module
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    DIRECTIONS,
    SAFETY_HORIZON,
    _has_sufficient_escape_directions,
    _opponent_distance_maps,
    _path_conflicts,
    extract_semantic_state,
    find_safe_path,
    is_free,
    neighbor_tile,
)

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"]


@contextmanager
def _pin_opponent_reseed(fixed_seed: int):
    """Context manager pinning framework opponents' randomness to
    `fixed_seed` across process launches, without editing their source.

    Redirects np.random.seed() calls (coin_collector_agent reseeds from OS
    entropy in setup()) to `fixed_seed`, and also seeds Python's global
    `random` state, which drives coin_collector_agent's shuffle() and is
    never reseeded by the agent. Both are restored on exit.
    """
    original_np_seed = np.random.seed
    original_random_state = _random_module.getstate()
    np.random.seed = lambda *args, **kwargs: original_np_seed(fixed_seed)
    _random_module.seed(fixed_seed)
    try:
        yield
    finally:
        np.random.seed = original_np_seed
        _random_module.setstate(original_random_state)


def _direction_detail(pos, field_arr, blocked, danger_offsets, opponents, can_move):
    """Per-direction detail from `pos`: the plain find_safe_path() result, and
    for directions with a path, its conflict tiles (as in
    _has_sufficient_escape_directions()).
    """
    opponent_dist_maps = _opponent_distance_maps(field_arr, blocked, opponents) if opponents else {}
    detail = {}
    candidate_paths = {}
    for d in DIRECTIONS:
        if not can_move[d]:
            detail[d] = {"base_safe": None, "path": None, "conflicts": {}, "robust": None}
            continue
        neighbor = neighbor_tile(pos, d)
        path = find_safe_path(neighbor, 0, field_arr, blocked, danger_offsets, SAFETY_HORIZON)
        base_safe = path is not None
        conflicts = _path_conflicts(path, opponent_dist_maps) if (path and opponent_dist_maps) else {}
        detail[d] = {"base_safe": base_safe, "path": path, "conflicts": conflicts, "robust": None}
        if base_safe:
            candidate_paths[d] = path

    all_conflicting_opponents = set()
    for d, info in detail.items():
        for opp_set in info["conflicts"].values():
            all_conflicting_opponents.update(opp_set)
    n_conflicting = len(all_conflicting_opponents)
    required = min(4, 1 + n_conflicting)

    for d, neighbor_path in candidate_paths.items():
        neighbor = neighbor_tile(pos, d)
        detail[d]["robust"] = _has_sufficient_escape_directions(
            neighbor, field_arr, blocked, danger_offsets, opponents, start_offset=0,
        )

    return detail, n_conflicting, required, len(candidate_paths)


def _instrument_step(state):
    semantic = extract_semantic_state(state)
    mask = mask_from_semantic(semantic)
    legal_actions = [cfg.ACTIONS[i] for i, m in enumerate(mask) if m]

    # Detail only while in danger, the only time self-kills and the
    # redundancy/tie-break logic apply; keeps a full census tractable.
    if semantic.current_tile_in_danger:
        detail, n_conflicting, required, n_candidates = _direction_detail(
            semantic.self_pos, semantic.field_arr, semantic.blocked, semantic.danger_offsets,
            semantic.opponents, semantic.can_move,
        )
    else:
        detail, n_conflicting, required, n_candidates = {}, 0, 1, None

    tie_break_uncontested = None
    if semantic.current_tile_in_danger and semantic.opponents:
        currently_legal_dirs = [d for d in DIRECTIONS if mask[cfg.ACTIONS.index(d)]]
        if len(currently_legal_dirs) > 1:
            opponent_dist_maps = _opponent_distance_maps(semantic.field_arr, semantic.blocked, semantic.opponents)
            tie_break_uncontested = [
                d for d in currently_legal_dirs
                if not any(
                    dmap.get(neighbor_tile(semantic.self_pos, d), float("inf")) <= 1
                    for dmap in opponent_dist_maps.values()
                )
            ]

    return {
        "step": state["step"],
        "self_pos": semantic.self_pos,
        "opponents": list(semantic.opponents),
        "bomb_available": semantic.bomb_available,
        "current_tile_in_danger": semantic.current_tile_in_danger,
        "nearest_threat_timer": semantic.nearest_threat_timer,
        "legal_actions": legal_actions,
        "detail": detail,
        "n_conflicting": n_conflicting,
        "required": required,
        "n_candidates": n_candidates,
        "tie_break_uncontested": tie_break_uncontested,
    }


def _dump_tail(log_file, step_log, actions, n):
    tail = step_log[-n:]
    offset0 = len(step_log) - len(tail)
    for i, rec in enumerate(tail):
        idx = offset0 + i
        chosen = actions[idx] if idx < len(actions) else None
        next_pos = step_log[idx + 1]["self_pos"] if idx + 1 < len(step_log) else None
        moved_flag = ""
        if chosen in DIRECTIONS and next_pos is not None:
            moved_flag = " <-- POSITION UNCHANGED after a movement action (INVALID_ACTION signature)" \
                if next_pos == rec["self_pos"] else ""
        common.log_print(
            log_file,
            f"\nstep={rec['step']} pos={rec['self_pos']} opponents(real)={rec['opponents']} "
            f"bomb_available={rec['bomb_available']} current_tile_in_danger={rec['current_tile_in_danger']} "
            f"nearest_threat_timer={rec['nearest_threat_timer']}\n"
            f"  mask legal_actions={rec['legal_actions']}  chosen_action={chosen}{moved_flag}\n"
            f"  N(conflicting opponents)={rec['n_conflicting']} required_paths={rec['required']} "
            f"candidate_paths_found={rec['n_candidates']}"
            + (f"  tie_break_uncontested={rec['tie_break_uncontested']}" if rec["tie_break_uncontested"] is not None else ""),
        )
        for d, info in rec["detail"].items():
            if info["base_safe"] is None:
                common.log_print(log_file, f"    {d}: BLOCKED")
                continue
            path_str = "->".join(f"{t}@{o}" for t, o in info["path"]) if info["path"] else "-"
            conflict_str = f" conflicts={info['conflicts']}" if info["conflicts"] else ""
            robust_str = f" robust={info['robust']}" if info["robust"] is not None else ""
            common.log_print(
                log_file,
                f"    {d}: base_safe={info['base_safe']} path={path_str}{conflict_str}{robust_str}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--breaker", choices=["on", "off"], required=True)
    parser.add_argument("--target-round", type=int, help="Replay up to this round, instrumenting only it.")
    parser.add_argument("--n-rounds", type=int,
                         help="Full-census mode: run this many fresh rounds (e.g. 100 to match the standard "
                              "eval protocol), instrumenting every round, and dump full detail for every round "
                              "that ends in self-kill -- not a sample, every occurrence in this deterministic run.")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--trace-last-n-steps", type=int, default=20)
    args = parser.parse_args()
    if (args.target_round is None) == (args.n_rounds is None):
        parser.error("exactly one of --target-round or --n-rounds is required")

    cfg.ENABLE_OSCILLATION_BREAKER = (args.breaker == "on")
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    mode = f"target_round={args.target_round}" if args.target_round else f"n_rounds={args.n_rounds}"
    ckpt_tag = Path(args.checkpoint).stem
    log_file, log_path = common.open_log_file(f"selfkill_census_{ckpt_tag}_{args.breaker}_{mode}")
    common.log_print(log_file, f"Instrumented self-kill census -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={args.checkpoint} breaker={args.breaker} {mode} eval_seed={args.eval_seed}")

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", args.checkpoint)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    with _pin_opponent_reseed(args.eval_seed):
        world = common.build_world(scenario=SCENARIO, seed=args.eval_seed, train=False, opponents=OPPONENTS)
        agent = world.agents[0]

        total_rounds = args.target_round if args.target_round else args.n_rounds
        n_selfkill = 0
        for round_idx in range(1, total_rounds + 1):
            world.new_round()
            world.user_input = "WAIT"
            instrument_this_round = bool(args.target_round) or bool(args.n_rounds)
            step_log = []
            while world.running:
                state = world.get_state_for_agent(agent)
                if instrument_this_round and state is not None:
                    step_log.append(_instrument_step(state))
                common._disable_think_time_limit(world)
                world.do_step("WAIT")

            self_kill = agent.statistics.get("suicides", 0) > 0
            actions = list(world.replay["actions"][agent.name])
            if args.target_round and round_idx == args.target_round:
                common.log_print(
                    log_file,
                    f"\nReplay of round {args.target_round}: agent.dead={agent.dead} self_kill={self_kill} "
                    f"steps_recorded={len(step_log)}",
                )
                _dump_tail(log_file, step_log, actions, args.trace_last_n_steps)
            elif args.n_rounds and self_kill:
                n_selfkill += 1
                common.log_print(
                    log_file,
                    f"\n{'='*20} self-kill at round {round_idx} (#{n_selfkill}) steps={len(step_log)} "
                    f"coins={agent.statistics.get('coins', 0)} crates={agent.statistics.get('crates', 0)} {'='*20}",
                )
                _dump_tail(log_file, step_log, actions, args.trace_last_n_steps)

        if args.n_rounds:
            common.log_print(log_file, f"\nCensus finished: {n_selfkill} self-kill(s) in {args.n_rounds} rounds")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
