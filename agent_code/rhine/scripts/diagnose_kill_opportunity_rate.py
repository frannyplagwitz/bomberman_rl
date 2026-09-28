"""Read-only diagnostic: kill-opportunity steps (a reachable opponent inside
the own blast range and BOMB mask-legal) and whether the agent bombed.

For bombed opportunities, checks whether the opponent disappears within the
bomb's lifetime; for skipped ones, records the policy's action
probabilities.

Usage:
  python -m agent_code.rhine.scripts.diagnose_kill_opportunity_rate \\
      --checkpoint models/task3_stage_c_seed0.pt --breaker on \\
      --n-rounds 100 --eval-seed 1000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.distributions import Categorical

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.model import masked_logits
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import blast_coords, extract_semantic_state

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"]
KILL_CHECK_WINDOW = cfg.BOMB_TIMER + 1


def _action_probabilities(model, features, mask):
    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        logits = masked_logits(logits, mask_t)
        probs = Categorical(logits=logits).probs.squeeze(0).tolist()
    return {cfg.ACTIONS[i]: round(p, 3) for i, p in enumerate(probs) if mask[i]}


def is_opportunity(semantic):
    if not semantic.bomb_available or not semantic.opponents:
        return False
    blast = set(blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER))
    if not any(opp in blast for opp in semantic.opponents):
        return False
    mask = mask_from_semantic(semantic)
    return bool(mask[cfg.ACTIONS.index("BOMB")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--breaker", choices=["on", "off"], default="on")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000)
    args = parser.parse_args()

    cfg.ENABLE_OSCILLATION_BREAKER = (args.breaker == "on")
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    ckpt_tag = Path(args.checkpoint).stem
    log_file, log_path = common.open_log_file(f"kill_opportunity_{ckpt_tag}_{args.breaker}")
    common.log_print(log_file, f"Kill-opportunity diagnostic -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={args.checkpoint} breaker={args.breaker} "
                                f"n_rounds={args.n_rounds} eval_seed={args.eval_seed}")

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", args.checkpoint)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=args.eval_seed, train=False, opponents=OPPONENTS)
    agent = world.agents[0]

    original_act = rhine_callbacks.act
    all_opportunities = []
    round_opponents_by_step = {}

    def instrumented_act(self, game_state):
        chosen = original_act(self, game_state)
        if game_state is not None:
            semantic = extract_semantic_state(game_state)
            round_opponents_by_step.setdefault(game_state["round"], {})[game_state["step"]] = list(semantic.opponents)
            if is_opportunity(semantic):
                entry = {
                    "round": game_state["round"], "step": game_state["step"],
                    "self_pos": semantic.self_pos, "opponents": list(semantic.opponents),
                    "chosen_action": chosen,
                }
                if chosen != "BOMB":
                    features = features_from_semantic(semantic)
                    mask = mask_from_semantic(semantic)
                    entry["probs"] = _action_probabilities(self.model, features, mask)
                all_opportunities.append(entry)
        return chosen

    rhine_callbacks.act = instrumented_act
    try:
        for round_idx in range(1, args.n_rounds + 1):
            world.new_round()
            world.user_input = "WAIT"
            while world.running:
                common._disable_think_time_limit(world)
                world.do_step("WAIT")
    finally:
        rhine_callbacks.act = original_act

    for opp in all_opportunities:
        if opp["chosen_action"] != "BOMB":
            continue
        steps_this_round = round_opponents_by_step.get(opp["round"], {})
        window_steps = [s for s in steps_this_round if opp["step"] < s <= opp["step"] + KILL_CHECK_WINDOW]
        opp["hit"] = any(len(steps_this_round[s]) == 0 for s in window_steps)

    n_total = len(all_opportunities)
    n_bomb_chosen = [o for o in all_opportunities if o["chosen_action"] == "BOMB"]
    n_hit = sum(1 for o in n_bomb_chosen if o.get("hit"))
    n_not_chosen = [o for o in all_opportunities if o["chosen_action"] != "BOMB"]

    common.log_print(log_file, f"\ntotal opportunities: {n_total}  per-round avg: {n_total / args.n_rounds:.3f}")
    common.log_print(
        log_file,
        f"BOMB chosen at opportunity: {len(n_bomb_chosen)}/{n_total} "
        f"({100 * len(n_bomb_chosen) / n_total if n_total else float('nan'):.1f}%)",
    )
    common.log_print(
        log_file,
        f"of those BOMB-chosen opportunities, opponent confirmed gone within "
        f"{KILL_CHECK_WINDOW} ticks: {n_hit}/{len(n_bomb_chosen)} "
        f"({100 * n_hit / len(n_bomb_chosen) if n_bomb_chosen else float('nan'):.1f}%)",
    )

    common.log_print(log_file, f"\n--- opportunities where BOMB was NOT chosen ({len(n_not_chosen)}) ---")
    for o in n_not_chosen:
        common.log_print(
            log_file,
            f"round={o['round']:>3} step={o['step']:>4} self_pos={o['self_pos']} opponents={o['opponents']} "
            f"chosen={o['chosen_action']} probs={o['probs']}",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
