"""Diagnostic-only bombing-position quality metrics (not used in rewards).

Missed opportunity: a BOMB placement is flagged if a BFS-reachable tile within
MAX_EXTRA_STEPS offered at least MIN_CRATE_ADVANTAGE more crates in blast.

Reports over a batch of evaluation rounds:
  missed_opportunity_rate             = flagged placements / all BOMB placements
  missed_opportunity_avg_extra_crates = mean crate advantage of the best
                                        alternative, over flagged placements

Candidate generation reuses the same helpers as
state_processing.bombing_target_info().
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fractions import Fraction

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    _bomb_positions,
    _to_coord,
    bfs_distances,
    crates_in_blast,
    has_safe_escape_after_bombing,
)

MAX_EXTRA_STEPS = 2
MIN_CRATE_ADVANTAGE = 1


def placement_missed_opportunity(field_arr, self_pos, blocked, power: int) -> Optional[int]:
    """Crate advantage of the best cheap alternative over self_pos, or None
    if no alternative qualifies.
    """
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    actual_crates = crates_in_blast(field_arr, self_pos, power)

    best_extra = None
    for tile, dist in dist_map.items():
        if tile == self_pos or dist > MAX_EXTRA_STEPS:
            continue
        extra = crates_in_blast(field_arr, tile, power) - actual_crates
        if extra >= MIN_CRATE_ADVANTAGE and (best_extra is None or extra > best_extra):
            best_extra = extra
    return best_extra


# Replaces the deprecated bomb_efficiency: checks whether each BOMB landed on a
# tile bombing_target_info()'s scoring would have chosen at that moment.
def bombing_target_best_by_score(field_arr, self_pos, blocked, power: int, danger_offsets, opponents=()) -> frozenset:
    """Best-by-score tile set under bombing_target_info()'s candidate
    generation and selection rule (all ties at the maximum score, before the
    distance tie-break). `opponents` is passed to the safety check as in the
    real agent.
    """
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    candidates = {}
    for tile, dist in dist_map.items():
        n = crates_in_blast(field_arr, tile, power)
        if n < 1:
            continue
        if cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER and not has_safe_escape_after_bombing(
            tile, field_arr, blocked, danger_offsets, power, opponents
        ):
            continue
        candidates[tile] = n

    if not candidates:
        return frozenset()

    if cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA:
        scores = {t: Fraction(n, dist_map[t] + 1) for t, n in candidates.items()}
        best_score = max(scores.values())
        return frozenset(t for t in candidates if scores[t] == best_score)
    else:
        best_distance = min(dist_map[t] for t in candidates)
        return frozenset(t for t in candidates if dist_map[t] == best_distance)


def is_optimal_bombing_placement(field_arr, self_pos, blocked, power: int, danger_offsets, opponents=()):
    """Whether the chosen placement (self_pos) is any of the tied best-by-score
    tiles; None when no crate-hitting candidate exists.
    """
    best_set = bombing_target_best_by_score(field_arr, self_pos, blocked, power, danger_offsets, opponents)
    if not best_set:
        return None
    return self_pos in best_set


def evaluate_cost_benefit(checkpoint_path, n_rounds: int, seed: int, scenario: str = "loot-crate") -> dict:
    """Deterministic evaluation that records (field, self_pos, blocked) per
    step and scores only the steps where BOMB was chosen.
    """
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=scenario, seed=seed, train=False)
    agent = world.agents[0]

    total_placements = 0
    flagged = 0
    extra_crates_when_flagged = []

    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        step_states = []
        while world.running:
            state = world.get_state_for_agent(agent)
            if state is not None:
                step_states.append((
                    state["field"], _to_coord(state["self"][3]), _bomb_positions(state)
                ))
            world.do_step("WAIT")

        actions = list(world.replay["actions"][agent.name])
        for (field_arr, self_pos, blocked), action in zip(step_states, actions):
            if action != "BOMB":
                continue
            total_placements += 1
            extra = placement_missed_opportunity(field_arr, self_pos, blocked, cfg.BOMB_POWER)
            if extra is not None:
                flagged += 1
                extra_crates_when_flagged.append(extra)

    world.end()

    return {
        "total_bomb_placements": total_placements,
        "missed_opportunity_rate": (flagged / total_placements) if total_placements else float("nan"),
        "missed_opportunity_avg_extra_crates": (
            float(np.mean(extra_crates_when_flagged)) if extra_crates_when_flagged else float("nan")
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    result = evaluate_cost_benefit(args.checkpoint, args.n_rounds, args.seed)
    print(result)
