"""Read-only diagnostic: per-episode detail for rounds classified as sustained
oscillation (diagnose_oscillation.longest_oscillation_run).

For each such round, records the run's action and position sequences, the
tiles it alternates between, the surrounding map, whether a bombing position
or reachable coin existed during the run, and bomb availability.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state


def _local_map_description(field_arr, pos):
    x, y = pos
    desc = {}
    for name, (dx, dy) in {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.items():
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= field_arr.shape[0] or ny >= field_arr.shape[1]:
            desc[name] = "out-of-bounds"
            continue
        v = field_arr[nx, ny]
        desc[name] = {-1: "wall", 1: "crate", 0: "free"}.get(v, f"unknown({v})")
    return desc


def collect_case_studies(checkpoint_path: str, n_rounds: int, seed: int, scenario: str = "loot-crate"):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", checkpoint_path)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_PPO_OVERRIDE", None)

    world = common.build_world(scenario=scenario, seed=seed, train=False)
    agent = world.agents[0]
    cases = []

    for round_num in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        trace = []
        while world.running:
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                trace.append({
                    "step": state["step"],
                    "self_pos": semantic.self_pos,
                    "field_arr": semantic.field_arr,
                    "bomb_available": semantic.bomb_available,
                    "has_reachable_coin": semantic.has_reachable_coin,
                    "has_bombing_target": semantic.has_bombing_target,
                    "nearest_bombing_distance": semantic.nearest_bombing_distance,
                })
            common._disable_think_time_limit(world)
            world.do_step("WAIT")

        actions = list(world.replay["actions"][agent.name])
        run_len, run_start = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue

        window = trace[run_start:run_start + run_len]
        positions_in_run = [r["self_pos"] for r in window]
        distinct_tiles = sorted(set(positions_in_run))
        pre_window = trace[max(0, run_start - 5):run_start]

        cases.append({
            "round": round_num,
            "episode_length": len(actions),
            "run_start_step": window[0]["step"] if window else None,
            "run_length": run_len,
            "distinct_tiles": distinct_tiles,
            "local_map_per_tile": {
                pos: _local_map_description(window[0]["field_arr"], pos) for pos in distinct_tiles
            } if window else {},
            "bomb_available_during_run": [r["bomb_available"] for r in window],
            "has_reachable_coin_during_run": [r["has_reachable_coin"] for r in window],
            "has_bombing_target_during_run": [r["has_bombing_target"] for r in window],
            "nearest_bombing_distance_during_run": [r["nearest_bombing_distance"] for r in window],
            "actions_before_run": [r for r in actions[max(0, run_start - 5):run_start]],
            "actions_during_run": actions[run_start:run_start + run_len],
        })

    world.end()
    return cases


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file(f"task2_oscillation_case_studies_{args.label}")
    cases = collect_case_studies(args.checkpoint, args.n_rounds, args.seed)
    common.log_print(log_file, f"[{args.label}] checkpoint={args.checkpoint} n_rounds={args.n_rounds} seed={args.seed}")
    common.log_print(log_file, f"Found {len(cases)} oscillating rounds out of {args.n_rounds}")

    for case in cases[:args.max_cases]:
        common.log_print(log_file, "\n" + "=" * 70)
        common.log_print(log_file, f"round={case['round']} episode_length={case['episode_length']} "
                                    f"run_start_step={case['run_start_step']} run_length={case['run_length']}")
        common.log_print(log_file, f"distinct_tiles={case['distinct_tiles']}")
        for pos, desc in case["local_map_per_tile"].items():
            common.log_print(log_file, f"  local_map@{pos}: {desc}")
        common.log_print(log_file, f"bomb_available_during_run={case['bomb_available_during_run']}")
        common.log_print(log_file, f"has_reachable_coin_during_run={case['has_reachable_coin_during_run']}")
        common.log_print(log_file, f"has_bombing_target_during_run={case['has_bombing_target_during_run']}")
        common.log_print(log_file, f"nearest_bombing_distance_during_run={case['nearest_bombing_distance_during_run']}")
        common.log_print(log_file, f"actions_before_run={case['actions_before_run']}")
        common.log_print(log_file, f"actions_during_run={case['actions_during_run']}")

    common.ring_bell()


if __name__ == "__main__":
    main()
