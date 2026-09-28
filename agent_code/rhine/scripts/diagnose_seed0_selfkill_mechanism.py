"""One-off: reproduces one Stage B seed0 self-kill (task3_stage_b_seed0_round100.pt,
eval_seed=1000) and dumps field/blocked/danger_offsets plus direct calls into
has_safe_escape_after_bombing()/exists_safe_path() at the bombing step, to
verify why the fatal direction was reported as a legal escape.
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    extract_semantic_state, exists_safe_path, has_safe_escape_after_bombing,
    blast_coords, DIRECTIONS, neighbor_tile, SAFETY_HORIZON,
)
from agent_code.rhine.action_mask import mask_from_semantic

SCENARIO = "classic"
OPPONENTS = ["peaceful_agent", "peaceful_agent", "peaceful_agent"]
EVAL_SEED = 1000
TARGET_ROUND = 32
CHECKPOINT = str(common.MODELS_DIR / "snapshots" / "task3_stage_b_seed0_round100.pt")


def main():
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", CHECKPOINT)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    agent = world.agents[0]

    for round_idx in range(TARGET_ROUND + 1):
        world.new_round()
        world.user_input = "WAIT"
        common._disable_think_time_limit(world)
        if round_idx < TARGET_ROUND:
            while world.running:
                common._disable_think_time_limit(world)
                world.do_step("WAIT")
            continue

        # Target round: dump full detail from shortly before the bombing step.
        step_count = 0
        while world.running:
            common._disable_think_time_limit(world)
            state = world.get_state_for_agent(agent)
            if state is not None and state["step"] >= 215:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                legal = [cfg.ACTIONS[i] for i, m in enumerate(mask) if m]
                print(f"step={state['step']} pos={semantic.self_pos} bomb_available={semantic.bomb_available} "
                      f"current_tile_in_danger={semantic.current_tile_in_danger} "
                      f"nearest_threat_timer={semantic.nearest_threat_timer} legal={legal}")
                if semantic.bomb_available:
                    # Re-derive has_safe_escape_after_bombing()'s decision
                    # here, with a per-direction breakdown.
                    print(f"  [would-place-bomb-here check] blast={blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER)}")
                for d in DIRECTIONS:
                    neighbor = neighbor_tile(semantic.self_pos, d)
                    if not semantic.can_move[d]:
                        print(f"  dir={d} neighbor={neighbor} blocked(wall/crate/bomb/opponent)")
                        continue
                    safe = exists_safe_path(
                        neighbor, 0, semantic.field_arr, semantic.blocked, semantic.danger_offsets, SAFETY_HORIZON
                    )
                    danger_at_neighbor = semantic.danger_offsets.get(neighbor, set())
                    print(f"  dir={d} neighbor={neighbor} exists_safe_path={safe} "
                          f"danger_offsets_at_neighbor={sorted(danger_at_neighbor)}")
                if state["step"] >= 224:
                    break
            world.do_step("WAIT")
            step_count += 1
            if step_count > 500:
                break

    world.end()


if __name__ == "__main__":
    main()
