"""Read-only diagnostic over the kill-opportunity steps defined in
diagnose_kill_opportunity_rate.py (bomb available, an opponent inside the own
blast, BOMB mask-legal), in three parts:

1. expected_kill_value_at_target for "opportunity, not bombed" vs "bombed
   and kill confirmed".
2. Cause split of "opportunity, not bombed":
     a) bomb unavailable (impossible by definition; reported as a check)
     b) BOMB illegal or self_pos not the kill target (also impossible, since
        the mask and kill_target_info() share the same safety check;
        reported to verify that coupling)
     c) BOMB legal and self_pos is the target, but the policy chose
        otherwise; reports kill value and BOMB probability
3. Wasteful-bomb vs kill-target mismatch: among BOMB_DROPPED steps with
   bomb_threatens_reachable_opponent() True, the share whose covered
   opponent had a low escape-difficulty score.

Opponent randomness is pinned via _pin_opponent_reseed() for determinism.
Reports data only.

Usage:
  python -m agent_code.rhine.scripts.diagnose_kill_target_discrimination \\
      --checkpoint <abs path> --breaker on --n-rounds 100 --eval-seed 1000
"""
import argparse
import random as _random_module
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.distributions import Categorical

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.model import masked_logits
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    _opponent_escape_difficulty,
    blast_coords,
    bomb_threatens_reachable_opponent,
    crates_in_blast,
    extract_semantic_state,
)

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"]
KILL_CHECK_WINDOW = cfg.BOMB_TIMER + 1
WASTEFUL_MISMATCH_THRESHOLD = 0.25


@contextmanager
def _pin_opponent_reseed(fixed_seed: int):
    """Pins opponents' Python `random` reseeding; see
    diagnose_selfkill_bfs_trace.py's identical helper."""
    original_np_seed = np.random.seed
    original_random_state = _random_module.getstate()
    np.random.seed = lambda *args, **kwargs: original_np_seed(fixed_seed)
    _random_module.seed(fixed_seed)
    try:
        yield
    finally:
        np.random.seed = original_np_seed
        _random_module.setstate(original_random_state)


def is_opportunity(semantic, mask):
    if not semantic.bomb_available or not semantic.opponents:
        return False
    blast = set(blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER))
    if not any(opp in blast for opp in semantic.opponents):
        return False
    return bool(mask[cfg.ACTIONS.index("BOMB")])


def _action_probabilities(model, features, mask):
    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        logits = masked_logits(logits, mask_t)
        probs = Categorical(logits=logits).probs.squeeze(0).tolist()
    return {cfg.ACTIONS[i]: round(p, 3) for i, p in enumerate(probs) if mask[i]}


def _bomb_wasteful_mismatch_check(semantic, game_state):
    """For a BOMB_DROPPED step: the wasteful-bomb verdict plus the escape
    difficulty of every opponent in the blast.
    """
    field_arr = semantic.field_arr
    pos = semantic.self_pos
    power = cfg.BOMB_POWER
    crates = crates_in_blast(field_arr, pos, power)
    threatens = bomb_threatens_reachable_opponent(game_state, pos, power)
    currently_wasteful = (crates == 0) and (not threatens)

    blast = set(blast_coords(field_arr, pos, power))
    hit_opponents = [opp for opp in semantic.opponents if opp in blast]
    difficulties = {}
    if hit_opponents:
        hypothetical_danger = {t: set(offsets) for t, offsets in semantic.danger_offsets.items()}
        for blast_tile in blast:
            hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
        for opp in hit_opponents:
            other_opponents = [o for o in semantic.opponents if o != opp]
            difficulties[opp] = float(_opponent_escape_difficulty(
                opp, field_arr, semantic.blocked, hypothetical_danger, pos, other_opponents,
            ))

    return {
        "crates": crates, "threatens_reachable_opponent": threatens,
        "currently_wasteful": currently_wasteful, "hit_opponents": hit_opponents,
        "difficulties": difficulties,
    }


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
    log_file, log_path = common.open_log_file(f"kill_target_discrimination_{ckpt_tag}_{args.breaker}")
    common.log_print(log_file, f"Kill-target discrimination diagnostic -- log: {log_path}")
    common.log_print(log_file, f"checkpoint={args.checkpoint} breaker={args.breaker} "
                                f"n_rounds={args.n_rounds} eval_seed={args.eval_seed}")

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", args.checkpoint)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    not_bombed = []  # Opportunity, BOMB not chosen.
    bombed = []  # Opportunity, BOMB chosen (with kill value at decision time).
    wasteful_checks = []  # Every BOMB_DROPPED step.
    round_opponents_by_step = {}

    with _pin_opponent_reseed(args.eval_seed):
        world = common.build_world(scenario=SCENARIO, seed=args.eval_seed, train=False, opponents=OPPONENTS)
        agent = world.agents[0]

        original_act = rhine_callbacks.act

        def instrumented_act(self, game_state):
            chosen = original_act(self, game_state)
            if game_state is not None:
                semantic = extract_semantic_state(game_state)
                mask = mask_from_semantic(semantic)
                round_opponents_by_step.setdefault(game_state["round"], {})[game_state["step"]] = \
                    list(semantic.opponents)

                if chosen == "BOMB":
                    wasteful_checks.append({
                        "round": game_state["round"], "step": game_state["step"],
                        **_bomb_wasteful_mismatch_check(semantic, game_state),
                    })

                if is_opportunity(semantic, mask):
                    entry = {
                        "round": game_state["round"], "step": game_state["step"],
                        "self_pos": semantic.self_pos, "opponents": list(semantic.opponents),
                        "bomb_available": semantic.bomb_available,
                        "mask_bomb_legal": bool(mask[cfg.ACTIONS.index("BOMB")]),
                        "has_kill_target": semantic.has_kill_target,
                        "nearest_kill_distance": semantic.nearest_kill_distance,
                        "expected_kill_value_at_target": semantic.expected_kill_value_at_target,
                        "chosen_action": chosen,
                    }
                    if chosen == "BOMB":
                        bombed.append(entry)
                    else:
                        features = features_from_semantic(semantic)
                        entry["probs"] = _action_probabilities(self.model, features, mask)
                        not_bombed.append(entry)
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

    for entry in bombed:
        steps_this_round = round_opponents_by_step.get(entry["round"], {})
        window_steps = [s for s in steps_this_round if entry["step"] < s <= entry["step"] + KILL_CHECK_WINDOW]
        entry["hit"] = any(len(steps_this_round[s]) == 0 for s in window_steps)

    # --- Part 1: expected_kill_value_at_target distribution ---
    not_bombed_values = [e["expected_kill_value_at_target"] for e in not_bombed]
    hit_values = [e["expected_kill_value_at_target"] for e in bombed if e["hit"]]

    def _dist_summary(values):
        if not values:
            return "n=0"
        arr = np.array(values, dtype=np.float64)
        buckets = [
            ("[0,0.25)", np.mean((arr >= 0) & (arr < 0.25))),
            ("[0.25,0.5)", np.mean((arr >= 0.25) & (arr < 0.5))),
            ("[0.5,0.75)", np.mean((arr >= 0.5) & (arr < 0.75))),
            ("[0.75,1.0]", np.mean((arr >= 0.75) & (arr <= 1.0))),
            (">1.0 (multi-opponent)", np.mean(arr > 1.0)),
        ]
        bucket_str = ", ".join(f"{name}={frac*100:.1f}%" for name, frac in buckets)
        return (f"n={len(arr)} mean={arr.mean():.3f} median={np.median(arr):.3f} "
                f"min={arr.min():.3f} max={arr.max():.3f}\n    buckets: {bucket_str}")

    common.log_print(log_file, "\n=== Part 1: expected_kill_value_at_target distribution ===")
    common.log_print(log_file, f"  opportunity-not-bombed: {_dist_summary(not_bombed_values)}")
    common.log_print(log_file, f"  bombed-and-hit:         {_dist_summary(hit_values)}")

    # --- Part 2: root-cause split of opportunity-not-bombed ---
    cat_a = [e for e in not_bombed if not e["bomb_available"]]
    cat_b = [
        e for e in not_bombed
        if e["bomb_available"] and (not e["mask_bomb_legal"] or not e["has_kill_target"] or e["nearest_kill_distance"] != 0)
    ]
    cat_c = [e for e in not_bombed if e not in cat_a and e not in cat_b]

    common.log_print(log_file, "\n=== Part 2: root-cause split of opportunity-but-not-bombed ===")
    n_total = len(not_bombed)
    common.log_print(
        log_file,
        f"  total={n_total}\n"
        f"  (a) bomb_available=False: {len(cat_a)} ({100*len(cat_a)/n_total if n_total else float('nan'):.1f}%)\n"
        f"  (b) mask-illegal or self_pos not recognized as kill target: {len(cat_b)} "
        f"({100*len(cat_b)/n_total if n_total else float('nan'):.1f}%)\n"
        f"  (c) legal + recognized, policy chose otherwise: {len(cat_c)} "
        f"({100*len(cat_c)/n_total if n_total else float('nan'):.1f}%)",
    )
    if cat_b:
        common.log_print(log_file, "\n  -- category (b) cases (should be empty; dumped for inspection if not) --")
        for e in cat_b[:20]:
            common.log_print(log_file, f"    {e}")
    common.log_print(log_file, "\n  -- category (c) sample (expected_kill_value_at_target + BOMB probability) --")
    for e in cat_c:
        common.log_print(
            log_file,
            f"    round={e['round']:>3} step={e['step']:>4} self_pos={e['self_pos']} opponents={e['opponents']} "
            f"expected_kill_value_at_target={e['expected_kill_value_at_target']:.3f} "
            f"chosen={e['chosen_action']} probs={e['probs']}",
        )

    # --- Part 3: wasteful-bomb-penalty / kill-target definition mismatch ---
    threatens_true = [w for w in wasteful_checks if w["threatens_reachable_opponent"]]
    low_difficulty_mismatches = [
        w for w in threatens_true
        if w["difficulties"] and max(w["difficulties"].values()) < WASTEFUL_MISMATCH_THRESHOLD
    ]
    common.log_print(log_file, "\n=== Part 3: wasteful-bomb-penalty / kill-target definition mismatch ===")
    common.log_print(
        log_file,
        f"  total BOMB_DROPPED steps: {len(wasteful_checks)}\n"
        f"  of those, threatens_reachable_opponent=True (not penalized on opponent-value grounds): "
        f"{len(threatens_true)}\n"
        f"  of THOSE, max hit-opponent escape-difficulty < {WASTEFUL_MISMATCH_THRESHOLD}: "
        f"{len(low_difficulty_mismatches)} "
        f"({100*len(low_difficulty_mismatches)/len(threatens_true) if threatens_true else float('nan'):.1f}%)",
    )
    for w in low_difficulty_mismatches[:20]:
        common.log_print(
            log_file,
            f"    round={w['round']:>3} step={w['step']:>4} crates={w['crates']} "
            f"hit_opponents={w['hit_opponents']} difficulties={ {k: round(v, 3) for k, v in w['difficulties'].items()} }",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
