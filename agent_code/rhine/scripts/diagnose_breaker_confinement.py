"""Read-only diagnostic: step-level confinement scan for
apply_oscillation_breaker().

oscillation_fraction only counts rounds with a long reversal run, so short
confinement episodes stay invisible. This classifies every confined step as:
  - confined_and_fired: the breaker filtered the retreat
  - confined_not_fired: by reason (dead_end_or_adjacency_edge /
    static_previous_action / cap_reached)
and flags "unresolved" fired steps whose next step is still confined.

The breaker decision is reconstructed like the live agent does (persistent
position deque and intervention streak through the real mask functions).

Usage:
  python -m agent_code.rhine.scripts.diagnose_breaker_confinement \\
      --checkpoint models/task3_stage_b_seed0.pt \\
      --opponents peaceful_agent peaceful_agent peaceful_agent \\
      --eval-seed 1000 --n-rounds 100

  # Several checkpoints, all against the same --opponents:
  python -m agent_code.rhine.scripts.diagnose_breaker_confinement \\
      --checkpoint models/task3_stage_b_seed0.pt models/task3_stage_b_seed1.pt \\
      --opponents peaceful_agent peaceful_agent peaceful_agent
"""
import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import (
    OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS,
    apply_oscillation_breaker,
    mask_from_semantic,
)
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import is_confined_to_small_range, extract_semantic_state

SCENARIO = "classic"
WINDOW_SIZE = 7  # Matches apply_oscillation_breaker()'s confinement window.


def run_for_checkpoint(checkpoint_path: str, opponents, eval_seed: int, n_rounds: int,
                        enable_no_bomb_when_cleared: bool = True):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = enable_no_bomb_when_cleared
    # Project-wide training configuration.
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", checkpoint_path)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=eval_seed, train=False, opponents=opponents)
    agent = world.agents[0]

    recent_positions = deque(maxlen=WINDOW_SIZE)
    streak = 0

    total_steps = 0
    confined_steps = 0
    confined_and_fired = 0
    not_fired_reasons = {"dead_end_or_adjacency_edge": 0, "static_previous_action": 0, "cap_reached": 0}
    fired_step_records = []  # (round_idx, step_idx_in_round), for the unresolved check.
    per_round_confined = []  # Per round, one bool per step.

    for round_idx in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        confined_flags = []
        fired_flags = []
        while world.running:
            common._disable_think_time_limit(world)
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                pre_mask = mask_from_semantic(semantic)
                recent_positions.append(semantic.self_pos)
                confined = is_confined_to_small_range(recent_positions, window_size=WINDOW_SIZE)
                post_mask, new_streak = apply_oscillation_breaker(pre_mask.copy(), recent_positions, streak)
                fired = bool(np.any(pre_mask != post_mask))
                streak = new_streak

                total_steps += 1
                confined_flags.append(confined)
                fired_flags.append(fired)
                if confined:
                    confined_steps += 1
                    if fired:
                        confined_and_fired += 1
                    else:
                        # Reconstruct why: cap reached, no movement last step, or dead end.
                        positions = list(recent_positions)
                        if len(positions) < 2 or positions[-1] == positions[-2]:
                            not_fired_reasons["static_previous_action"] += 1
                        elif streak >= OSCILLATION_BREAKER_MAX_CONSECUTIVE_INTERVENTIONS:
                            not_fired_reasons["cap_reached"] += 1
                        else:
                            # Dead end, or non-adjacent history at a round boundary.
                            not_fired_reasons["dead_end_or_adjacency_edge"] += 1
            world.do_step("WAIT")

        per_round_confined.append(confined_flags)
        for i, fired in enumerate(fired_flags):
            if fired:
                fired_step_records.append((round_idx, i))

    world.end()

    # "Unresolved": still confined at the next step in the same round.
    unresolved = 0
    for round_idx, step_i in fired_step_records:
        flags = per_round_confined[round_idx]
        if step_i + 1 < len(flags) and flags[step_i + 1]:
            unresolved += 1

    return {
        "total_steps": total_steps,
        "confined_steps": confined_steps,
        "confined_and_fired": confined_and_fired,
        "confined_not_fired": confined_steps - confined_and_fired,
        "not_fired_reasons": not_fired_reasons,
        "total_fired": len(fired_step_records),
        "unresolved_fired": unresolved,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                         help="One or more checkpoint paths to scan (each reported separately).")
    parser.add_argument("--opponents", nargs="*", default=[],
                         help="Opponent agent names, e.g. peaceful_agent peaceful_agent peaceful_agent.")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--disable-no-bomb-when-cleared", action="store_true",
                         help="Turn off ENABLE_NO_BOMB_WHEN_BOARD_CLEARED for this scan (default: on).")
    args = parser.parse_args()

    log_file, log_path = common.open_log_file("breaker_confinement_scan")
    common.log_print(log_file, f"Breaker confinement broad scan -- log: {log_path}")
    common.log_print(log_file, f"scenario={SCENARIO} opponents={args.opponents} eval_seed={args.eval_seed} "
                                f"n_rounds={args.n_rounds} window_size={WINDOW_SIZE}")

    for checkpoint in args.checkpoint:
        r = run_for_checkpoint(
            checkpoint, args.opponents, args.eval_seed, args.n_rounds,
            enable_no_bomb_when_cleared=not args.disable_no_bomb_when_cleared,
        )
        confined_rate = r["confined_steps"] / r["total_steps"] if r["total_steps"] else float("nan")
        unresolved_rate = r["unresolved_fired"] / r["total_fired"] if r["total_fired"] else float("nan")
        common.log_print(log_file, f"\n=== {checkpoint} ===")
        common.log_print(
            log_file,
            f"total_steps={r['total_steps']} confined_steps={r['confined_steps']} "
            f"({confined_rate:.4f})\n"
            f"  confined_and_fired={r['confined_and_fired']} confined_not_fired={r['confined_not_fired']}\n"
            f"  not_fired_reasons={r['not_fired_reasons']}\n"
            f"  total_fired={r['total_fired']} unresolved_fired={r['unresolved_fired']} "
            f"({unresolved_rate:.4f})",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
