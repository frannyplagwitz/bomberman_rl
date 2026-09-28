"""Read-only diagnostic: for Task 3 Stage B's breaker-on checkpoints,
characterizes every sustained-oscillation round (the oscillation_fraction
criterion): bouncing between exactly two tiles vs drifting through more, and
whether the run lasts to the round's end or the agent breaks out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state

SCENARIO = "classic"
OPPONENTS = ["peaceful_agent", "peaceful_agent", "peaceful_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100


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

    cases = []
    for round_idx in range(N_ROUNDS):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        positions = []
        while world.running:
            common._disable_think_time_limit(world)
            state = world.get_state_for_agent(agent)
            if state is not None:
                positions.append(extract_semantic_state(state).self_pos)
            world.do_step("WAIT")

        actions = list(world.replay["actions"][agent.name])
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue

        total_steps = len(actions)
        run_positions = positions[start_idx:start_idx + run_len + 1]
        distinct_tiles = sorted(set(run_positions))
        run_ends_at = start_idx + run_len
        runs_to_round_end = run_ends_at >= total_steps - 1

        # If the agent breaks out, record what it does right afterwards.
        post_break_actions = actions[run_ends_at:run_ends_at + 10] if not runs_to_round_end else []

        cases.append({
            "round": round_idx,
            "total_steps": total_steps,
            "run_start_step": start_idx,
            "run_length": run_len,
            "distinct_tile_count": len(distinct_tiles),
            "distinct_tiles": distinct_tiles,
            "runs_to_round_end": runs_to_round_end,
            "post_break_actions": post_break_actions,
        })

    world.end()
    return cases


def main():
    log_file, log_path = common.open_log_file("task3_stage_b_oscillation_patterns")
    common.log_print(log_file, f"Stage B oscillation pattern scan (breaker-ON, delivered checkpoints) -- log: {log_path}")
    common.log_print(log_file, f"scenario={SCENARIO} opponents={OPPONENTS} eval_seed={EVAL_SEED} "
                                f"n_rounds={N_ROUNDS} OSCILLATION_THRESHOLD={OSCILLATION_THRESHOLD}")

    for seed_id in (0, 1, 2):
        checkpoint = str(common.MODELS_DIR / f"task3_stage_b_seed{seed_id}.pt")
        cases = run_for_checkpoint(checkpoint)
        common.log_print(log_file, f"\n=== seed{seed_id} ({checkpoint}) -- {len(cases)} sustained-oscillation round(s) ===")
        for c in cases:
            common.log_print(
                log_file,
                f"  round={c['round']:>3} total_steps={c['total_steps']:>3} "
                f"run_start={c['run_start_step']:>3} run_length={c['run_length']:>3} "
                f"distinct_tiles={c['distinct_tile_count']} {c['distinct_tiles']} "
                f"runs_to_round_end={c['runs_to_round_end']} "
                f"post_break_actions={c['post_break_actions']}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
