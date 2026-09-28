"""Task 2 diagnostic: does the bombing-target tile set change right when a
sustained oscillation starts, e.g. because the agent's own new bomb blocks
the former shortest path?

Mirrors bombing_target_info()'s nearest-distance candidate selection but
exposes the tied target tiles themselves. For each oscillating episode (same
detector as diagnose_oscillation.py), prints a window around the onset:
position, own bombs, tied target tiles and whether that set changed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine import config as cfg
from agent_code.rhine.state_processing import (
    _bomb_positions,
    bfs_distances,
    crates_in_blast,
    extract_semantic_state,
)
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts.diagnose_oscillation import longest_oscillation_run, OSCILLATION_THRESHOLD

N_ROUNDS = 30
SEED = 1000
SCENARIO = "loot-crate"
CHECKPOINT = common.MODELS_DIR / "task2_stage_a.pt"
WINDOW_BEFORE = 6
WINDOW_AFTER = 14
MAX_EXAMPLES = 5


def tied_bombing_targets(field_arr, self_pos, blocked, power):
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    candidates = {}
    for tile, dist in dist_map.items():
        n = crates_in_blast(field_arr, tile, power)
        if n >= 1:
            candidates[tile] = (dist, n)
    if not candidates:
        return None, frozenset(), 0
    nearest_distance = min(d for d, n in candidates.values())
    tied = frozenset(t for t, (d, n) in candidates.items() if d == nearest_distance)
    max_crates = max(candidates[t][1] for t in tied)
    return nearest_distance, tied, max_crates


def run_and_diagnose():
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(CHECKPOINT))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=SEED, train=False)
    agent = world.agents[0]

    examples = []
    for round_idx in range(1, N_ROUNDS + 1):
        world.new_round()
        world.user_input = "WAIT"
        trace = []
        while world.running:
            state = world.get_state_for_agent(agent)
            record = None
            if state is not None:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                legal = [cfg.ACTIONS[i] for i, m in enumerate(mask) if m]
                blocked = _bomb_positions(state)
                dist, tied, max_crates = tied_bombing_targets(state["field"], semantic.self_pos, blocked, cfg.BOMB_POWER)
                record = {
                    "step": state["step"], "pos": semantic.self_pos, "blocked": blocked,
                    "bomb_available": semantic.bomb_available, "legal": legal,
                    "target_dist": dist, "tied_targets": tied, "max_crates": max_crates,
                }
            world.do_step("WAIT")
            if record is not None:
                trace.append(record)

        actions = list(world.replay["actions"][agent.name])
        for rec, a in zip(trace, actions):
            rec["chosen"] = a

        run_len, start_idx = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD and start_idx is not None and len(examples) < MAX_EXAMPLES:
            lo = max(0, start_idx - WINDOW_BEFORE)
            hi = min(len(trace), start_idx + WINDOW_AFTER)
            examples.append((round_idx, start_idx, trace[lo:hi]))

    world.end()
    return examples


def main():
    log_file, log_path = common.open_log_file("task2_bombing_target_stability")
    common.log_print(log_file, f"Bombing-target stability diagnostic -- log file: {log_path}")

    examples = run_and_diagnose()
    common.log_print(log_file, f"found {len(examples)} oscillating episode(s) to inspect\n")

    for round_idx, start_idx, window in examples:
        common.log_print(log_file, f"=== round {round_idx}, oscillation starts at trace-index {start_idx} ===")
        prev_tied = None
        for rec in window:
            flip = "" if prev_tied is None else (" <== TARGET IDENTITY CHANGED" if rec["tied_targets"] != prev_tied else "")
            marker = " <<< OSCILLATION STARTS HERE" if rec is window[min(start_idx - max(0, start_idx - WINDOW_BEFORE), len(window)-1)] else ""
            common.log_print(
                log_file,
                f"  step={rec['step']:>4} pos={rec['pos']} bomb_available={rec['bomb_available']} "
                f"own_bombs_blocking={sorted(rec['blocked'])} "
                f"target_dist={rec['target_dist']} n_tied_targets={len(rec['tied_targets'])} "
                f"tied_targets={sorted(rec['tied_targets'])[:6]}{'...' if len(rec['tied_targets'])>6 else ''} "
                f"legal={rec['legal']} chosen={rec['chosen']}{flip}",
            )
            prev_tied = rec["tied_targets"]
        common.log_print(log_file, "")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
