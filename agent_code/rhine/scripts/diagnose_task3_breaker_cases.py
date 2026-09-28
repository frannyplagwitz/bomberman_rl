"""Read-only diagnostic: classifies every sustained-oscillation round in Task 3
Stage A's final 100-round evaluation (per seed) as a true dead end (breaker
correctly stays out) or a through-passage (breaker should intervene).

Reconstructs the breaker's per-step decisions with a local persistent
position deque fed through the unmodified mask_from_semantic() and
apply_oscillation_breaker(), on the same trajectory as the real evaluation.
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import settings as s
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import apply_oscillation_breaker, mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import DIRECTIONS, extract_semantic_state, is_confined_to_small_range, neighbor_tile

_OPPOSITE_DIRECTION = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

SCENARIO = "classic"
OPPONENTS = ["peaceful_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


def _onward_direction(current, previous):
    """Direction from `current` away from `previous`, as derived in
    apply_oscillation_breaker().
    """
    direction_to_previous = next(
        (d for d in DIRECTIONS if neighbor_tile(current, d) == previous), None
    )
    if direction_to_previous is None:
        return None
    return _OPPOSITE_DIRECTION[direction_to_previous]


def _structurally_free(field_arr, pos, direction):
    x, y = neighbor_tile(pos, direction)
    if x < 0 or y < 0 or x >= field_arr.shape[0] or y >= field_arr.shape[1]:
        return False
    return bool(field_arr[x, y] == 0)


def run_for_checkpoint(checkpoint_path: str):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", checkpoint_path)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    agent = world.agents[0]

    # One deque for the whole run, never reset across rounds, like the live agent's.
    recent_positions = deque(maxlen=8)

    cases = []
    n_oscillating_rounds = 0
    round_scores = []  # Index-aligned with round_idx.

    for round_idx in range(N_ROUNDS):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        trace = []
        while world.running:
            common._disable_think_time_limit(world)
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                pre_mask = mask_from_semantic(semantic)
                recent_positions.append(semantic.self_pos)
                post_mask = apply_oscillation_breaker(pre_mask.copy(), recent_positions)
                trace.append({
                    "step": state["step"],
                    "pos": semantic.self_pos,
                    "field_arr": semantic.field_arr,
                    "pre_legal": [cfg.ACTIONS[i] for i, m in enumerate(pre_mask) if m],
                    "post_legal": [cfg.ACTIONS[i] for i, m in enumerate(post_mask) if m],
                    "breaker_fired": bool(np.any(pre_mask != post_mask)),
                    "confined": is_confined_to_small_range(recent_positions, window_size=8),
                })
            world.do_step("WAIT")

        coins_collected = agent.statistics.get("coins", 0)
        opponent_kills = agent.statistics.get("kills", 0)
        round_scores.append(coins_collected * s.REWARD_COIN + opponent_kills * s.REWARD_KILL)

        actions = list(world.replay["actions"][agent.name])
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue
        n_oscillating_rounds += 1

        window = trace[start_idx:start_idx + run_len]
        if not window:
            continue
        positions = [r["pos"] for r in window]
        distinct_tiles = sorted(set(positions))

        onward_free_flags = []
        breaker_fired_flags = []
        detail_lines = []
        for i in range(1, len(window)):
            current, previous = window[i]["pos"], window[i - 1]["pos"]
            if current == previous:
                continue
            direction = _onward_direction(current, previous)
            if direction is None:
                continue
            back_direction = _OPPOSITE_DIRECTION[direction]
            onward_legal = direction in window[i]["pre_legal"]
            back_legal = back_direction in window[i]["pre_legal"]
            onward_free_flags.append(onward_legal)
            breaker_fired_flags.append(window[i]["breaker_fired"])
            detail_lines.append(
                f"    step={window[i]['step']:>4} pos={current} prev={previous} confined={window[i]['confined']} "
                f"back={back_direction}(legal={back_legal}) onward={direction}(legal={onward_legal}) "
                f"pre={window[i]['pre_legal']} post={window[i]['post_legal']} fired={window[i]['breaker_fired']}"
            )

        ever_onward_legal = any(onward_free_flags)
        ever_breaker_fired = any(breaker_fired_flags)
        classification = "corridor (breaker should intervene)" if ever_onward_legal else "true dead end"

        cases.append({
            "round": round_idx,
            "run_start_step": window[0]["step"],
            "run_length": run_len,
            "distinct_tiles": distinct_tiles,
            "classification": classification,
            "onward_legal_step_count": sum(onward_free_flags),
            "onward_checked_step_count": len(onward_free_flags),
            "breaker_fired_step_count": sum(breaker_fired_flags),
            "ever_breaker_fired": ever_breaker_fired,
            "runs_to_round_end": (start_idx + run_len) >= len(trace) - 1,
            "detail_lines": detail_lines,
            "score": round_scores[round_idx],
        })

    world.end()
    return n_oscillating_rounds, cases, round_scores


def main():
    log_file, log_path = common.open_log_file("task3_stage_a_breaker_case_classification")
    common.log_print(log_file, f"Task 3 Stage A oscillation-breaker case classification -- log: {log_path}")
    common.log_print(log_file, f"scenario={SCENARIO} opponents={OPPONENTS} eval_seed={EVAL_SEED} "
                                f"n_rounds={N_ROUNDS} OSCILLATION_THRESHOLD={OSCILLATION_THRESHOLD}")

    for seed_id in (0, 1, 2):
        checkpoint = str(common.MODELS_DIR / f"task3_stage_a_seed{seed_id}.pt")
        n_oscillating, cases, round_scores = run_for_checkpoint(checkpoint)
        score_mean = float(np.mean(round_scores))
        common.log_print(log_file, f"\n=== seed{seed_id} ({checkpoint}) ===")
        common.log_print(log_file, f"score_mean over all {N_ROUNDS} rounds this run: {score_mean:.2f}")
        common.log_print(log_file, f"oscillating rounds (longest_run>={OSCILLATION_THRESHOLD}): "
                                    f"{n_oscillating}/{N_ROUNDS}")
        dead_end = [c for c in cases if c["classification"] == "true dead end"]
        corridor = [c for c in cases if c["classification"].startswith("corridor")]
        common.log_print(log_file, f"classified: dead_end={len(dead_end)} corridor={len(corridor)}")
        for c in cases:
            common.log_print(
                log_file,
                f"  round={c['round']:>3} start_step={c['run_start_step']:>4} run_length={c['run_length']:>3} "
                f"tiles={c['distinct_tiles']} class={c['classification']} score={c['score']:.1f} "
                f"onward_legal={c['onward_legal_step_count']}/{c['onward_checked_step_count']} "
                f"breaker_fired_steps={c['breaker_fired_step_count']} "
                f"ever_fired={c['ever_breaker_fired']} runs_to_round_end={c['runs_to_round_end']}",
            )
        if corridor:
            corridor_scores = [c["score"] for c in corridor]
            common.log_print(
                log_file,
                f"corridor-case scores: {corridor_scores} vs seed score_mean={score_mean:.2f} "
                f"(corridor mean={np.mean(corridor_scores):.2f})",
            )

        first_corridor = next((c for c in corridor if c["onward_legal_step_count"] > 0), None)
        if first_corridor is not None:
            common.log_print(
                log_file,
                f"\n  -- per-step detail for round={first_corridor['round']} (first corridor case) --",
            )
            for line in first_corridor["detail_lines"]:
                common.log_print(log_file, line)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()


def debug_dump_round(checkpoint_path: str, target_round: int):
    """Dumps one round's per-step trace (pos/confined/pre_legal/post_legal)
    for manual inspection of breaker decisions. Not used by main().
    """
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", checkpoint_path)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    agent = world.agents[0]
    recent_positions = deque(maxlen=8)

    for round_idx in range(N_ROUNDS):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        trace = []
        while world.running:
            common._disable_think_time_limit(world)
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                pre_mask = mask_from_semantic(semantic)
                recent_positions.append(semantic.self_pos)
                confined = is_confined_to_small_range(recent_positions, window_size=8)
                post_mask = apply_oscillation_breaker(pre_mask.copy(), recent_positions)
                trace.append({
                    "step": state["step"], "pos": semantic.self_pos,
                    "pre_legal": [cfg.ACTIONS[i] for i, m in enumerate(pre_mask) if m],
                    "post_legal": [cfg.ACTIONS[i] for i, m in enumerate(post_mask) if m],
                    "confined": confined,
                    "history": list(recent_positions),
                })
            world.do_step("WAIT")

        actions = list(world.replay["actions"][agent.name])
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            print(f"round={round_idx} run_len={run_len} start_idx={start_idx} total_steps={len(actions)}")
            lo = max(0, (start_idx or 0) - 3)
            hi = min(len(trace), (start_idx or 0) + run_len + 3) if start_idx is not None else len(trace)
            for i in range(lo, hi):
                r = trace[i]
                a = actions[i] if i < len(actions) else None
                print(f"  i={i:>3} step={r['step']:>4} pos={r['pos']} action={a} confined={r['confined']} "
                      f"pre={r['pre_legal']} post={r['post_legal']} hist={r['history']}")
            world.end()
            return

    world.end()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--debug-round":
    ckpt = sys.argv[2]
    rnd = int(sys.argv[3])
    debug_dump_round(ckpt, rnd)
