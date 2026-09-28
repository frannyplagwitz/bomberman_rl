"""Task 3 self-kill replay verification: deterministically re-runs a
100-round evaluation on the current code and diagnoses every self-kill.

The think-time limit is disabled before every world.do_step(), since
environment.py resets the budget each step.

Opponent RNG: framework opponents call np.random.seed() without an argument
in setup(), so their movement is not reproducible. For this process only,
peaceful_agent's setup() is monkeypatched to reseed with a fixed constant
(nothing on disk changes). Self-kills found here are therefore new
reproducible cases, not reproductions of earlier traces.

Additional diagnostics (never used to drive the mask or regular metrics):
- bombs_placed_mean, bomb_attempt_denial_rate and missed_opportunity_rate,
  the latter against _old_has_safe_escape_after_bombing(), a frozen
  Manhattan-distance version of the bombing gate.
- the per-step redundancy trigger under both the new (BFS + fuse window) and
  old (Manhattan) formulas on the same trajectory: fire rates,
  disagreements, whether a firing actually masks a direction, and how much
  earlier the new formula fires.
- prediction misjudge rate: decisions denied only by the opponent-movement
  prediction layer record the predicted (tile, offset) pairs; after the
  round these are checked against opponents' actual positions. misjudge_rate
  = false positives / checkable denials; denials whose window extends past
  the round end are reported separately.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    DIRECTIONS,
    SAFETY_HORIZON,
    _augment_danger_with_opponent_prediction,
    _effective_opponent_proximity,
    _has_sufficient_escape_directions,
    blast_coords,
    exists_safe_path,
    extract_semantic_state,
    is_free,
    neighbor_tile,
)

import agent_code.peaceful_agent.callbacks as _peaceful_callbacks

DIAGNOSTIC_OPPONENT_SEED = 424242


def _deterministic_peaceful_setup(self):
    np.random.seed(DIAGNOSTIC_OPPONENT_SEED)


_peaceful_callbacks.setup = _deterministic_peaceful_setup

SEED = 1000
N_ROUNDS = 100
SCENARIO = "classic"

CHECKPOINTS = {
    "seed0": common.MODELS_DIR / "task3_stage_a_seed0.pt",
    "seed1": common.MODELS_DIR / "task3_stage_a_seed1.pt",
    "seed2": common.MODELS_DIR / "task3_stage_a_seed2.pt",
}
OPPONENTS = ["peaceful_agent"]  # Stage A curriculum.


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _old_has_sufficient_escape_directions(pos, field_arr, blocked, danger_offsets, opponents, start_offset):
    """Frozen Manhattan-distance escape-direction standard, used only for the
    old-vs-new comparison.
    """
    required = 2 if any(_manhattan(pos, opp) <= 2 for opp in opponents) else 1
    count = 0
    for d in DIRECTIONS:
        neighbor = neighbor_tile(pos, d)
        if not is_free(field_arr, *neighbor, blocked=blocked):
            continue
        if exists_safe_path(neighbor, start_offset + 1, field_arr, blocked, danger_offsets, SAFETY_HORIZON):
            count += 1
            if count >= required:
                return True
    return False


def _old_has_safe_escape_after_bombing(tile, field_arr, blocked, danger_offsets, power, opponents):
    """Frozen has_safe_escape_after_bombing() from before the BFS-distance +
    fuse-window fix, used only to measure opportunities the new gate denies.
    """
    hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
    for blast_tile in blast_coords(field_arr, tile, power):
        hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    blocked_with_new_bomb = blocked | {tile}
    if 0 in hypothetical_danger.get(tile, ()):
        return False
    hypothetical_danger = _augment_danger_with_opponent_prediction(
        hypothetical_danger, field_arr, blocked, opponents, tile, cfg.BOMB_TIMER,
    )
    return _old_has_sufficient_escape_directions(
        tile, field_arr, blocked_with_new_bomb, hypothetical_danger, opponents, start_offset=0
    )


def _predicted_only_pairs(base_danger, augmented_danger):
    """(tile, offset) pairs added only by the opponent-movement prediction
    layer, not by real bombs/explosions.
    """
    pairs = set()
    for tile, offsets in augmented_danger.items():
        for offset in offsets - base_danger.get(tile, set()):
            pairs.add((tile, offset))
    return pairs


def replay_checkpoint(checkpoint_path, opponents, n_rounds, seed, label=""):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False, opponents=opponents)
    agent = world.agents[0]
    self_kills = []
    stats = {
        "n_rounds": n_rounds,
        "bombs_placed": 0,
        "bomb_available_steps": 0,
        "bomb_denied_steps": 0,
        "old_legal_steps": 0,
        "missed_opportunity_steps": 0,
        "danger_steps": 0,
        "new_trigger_steps": 0,
        "old_trigger_steps": 0,
        "both_trigger_steps": 0,
        "new_only_trigger_steps": 0,
        "old_only_trigger_steps": 0,
        "trigger_effective_steps": 0,
        "earlier_by_steps": [],
        "prediction_denials": 0,
        "prediction_false_positives": 0,
        "prediction_confirmed": 0,
        "prediction_inconclusive": 0,
    }

    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        round_index = world.round
        trace = []
        round_prediction_denials = []
        first_new_trigger_step = None
        first_old_trigger_step = None
        while world.running:
            state = world.get_state_for_agent(agent)
            if state is not None:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                nearest_opp = min(
                    (_manhattan(semantic.self_pos, o) for o in semantic.opponents), default=None,
                )
                old_redundancy_trigger = semantic.current_tile_in_danger and any(
                    _manhattan(semantic.self_pos, o) <= 2 for o in semantic.opponents
                )
                new_redundancy_trigger = semantic.current_tile_in_danger and _effective_opponent_proximity(
                    semantic.self_pos, semantic.field_arr, semantic.blocked, semantic.opponents,
                    semantic.nearest_threat_timer,
                )
                trigger_masked_direction = False
                if new_redundancy_trigger:
                    pre_redundancy_legal = {
                        d: exists_safe_path(
                            neighbor_tile(semantic.self_pos, d), 0, semantic.field_arr, semantic.blocked,
                            semantic.danger_offsets, SAFETY_HORIZON,
                        )
                        for d in DIRECTIONS if semantic.can_move[d]
                    }
                    trigger_masked_direction = any(
                        legal and not mask[cfg.ACTIONS.index(d)] for d, legal in pre_redundancy_legal.items()
                    )
                    # Misjudge-rate diagnostic: robust without prediction but
                    # not with it means the direction was denied by prediction alone.
                    augmented_danger = _augment_danger_with_opponent_prediction(
                        semantic.danger_offsets, semantic.field_arr, semantic.blocked, semantic.opponents,
                        semantic.self_pos, semantic.nearest_threat_timer,
                    )
                    for d, legal in pre_redundancy_legal.items():
                        if not legal or mask[cfg.ACTIONS.index(d)]:
                            continue
                        neighbor = neighbor_tile(semantic.self_pos, d)
                        robust_without_prediction = _has_sufficient_escape_directions(
                            neighbor, semantic.field_arr, semantic.blocked, semantic.danger_offsets,
                            semantic.opponents, start_offset=0, fuse_remaining_ticks=semantic.nearest_threat_timer,
                        )
                        if robust_without_prediction:
                            round_prediction_denials.append({
                                "step": state["step"], "type": f"DIRECTION:{d}",
                                "pairs": _predicted_only_pairs(semantic.danger_offsets, augmented_danger),
                            })
                if semantic.current_tile_in_danger:
                    stats["danger_steps"] += 1
                    if new_redundancy_trigger:
                        stats["new_trigger_steps"] += 1
                        if first_new_trigger_step is None:
                            first_new_trigger_step = state["step"]
                    if old_redundancy_trigger:
                        stats["old_trigger_steps"] += 1
                        if first_old_trigger_step is None:
                            first_old_trigger_step = state["step"]
                    if new_redundancy_trigger and old_redundancy_trigger:
                        stats["both_trigger_steps"] += 1
                    elif new_redundancy_trigger:
                        stats["new_only_trigger_steps"] += 1
                    elif old_redundancy_trigger:
                        stats["old_only_trigger_steps"] += 1
                    if trigger_masked_direction:
                        stats["trigger_effective_steps"] += 1
                new_bomb_legal = bool(mask[cfg.ACTIONS.index("BOMB")])
                if semantic.bomb_available:
                    old_bomb_legal = _old_has_safe_escape_after_bombing(
                        semantic.self_pos, semantic.field_arr, semantic.blocked, semantic.danger_offsets,
                        cfg.BOMB_POWER, semantic.opponents,
                    )
                    stats["bomb_available_steps"] += 1
                    if not new_bomb_legal:
                        stats["bomb_denied_steps"] += 1
                    if old_bomb_legal:
                        stats["old_legal_steps"] += 1
                        if not new_bomb_legal:
                            stats["missed_opportunity_steps"] += 1

                    # Same with/without-prediction comparison for BOMB.
                    if not new_bomb_legal:
                        hyp_real = {t: set(o) for t, o in semantic.danger_offsets.items()}
                        for blast_tile in blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER):
                            hyp_real.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
                        blocked_with_new_bomb = semantic.blocked | {semantic.self_pos}
                        bomb_legal_without_prediction = (
                            0 not in hyp_real.get(semantic.self_pos, ())
                            and _has_sufficient_escape_directions(
                                semantic.self_pos, semantic.field_arr, blocked_with_new_bomb, hyp_real,
                                semantic.opponents, start_offset=0, fuse_remaining_ticks=cfg.BOMB_TIMER,
                            )
                        )
                        if bomb_legal_without_prediction:
                            hyp_augmented = _augment_danger_with_opponent_prediction(
                                hyp_real, semantic.field_arr, semantic.blocked, semantic.opponents,
                                semantic.self_pos, cfg.BOMB_TIMER,
                            )
                            round_prediction_denials.append({
                                "step": state["step"], "type": "BOMB",
                                "pairs": _predicted_only_pairs(hyp_real, hyp_augmented),
                            })
                trace.append({
                    "step": state["step"],
                    "self_pos": semantic.self_pos,
                    "opponents": list(semantic.opponents),
                    "current_tile_in_danger": semantic.current_tile_in_danger,
                    "nearest_threat_timer": semantic.nearest_threat_timer,
                    "nearest_opp_manhattan": nearest_opp,
                    "redundancy_trigger": new_redundancy_trigger,
                    "old_redundancy_trigger": old_redundancy_trigger,
                    "legal_actions": [cfg.ACTIONS[i] for i, m in enumerate(mask) if m],
                })
            # Must be re-applied every step.
            common._disable_think_time_limit(world)
            world.do_step("WAIT")

        if first_new_trigger_step is not None and first_old_trigger_step is not None:
            stats["earlier_by_steps"].append(first_old_trigger_step - first_new_trigger_step)

        # Check each prediction-only denial against the opponents' actual positions.
        step_to_opponents = {rec["step"]: rec["opponents"] for rec in trace}
        last_step = trace[-1]["step"] if trace else None
        for denial in round_prediction_denials:
            stats["prediction_denials"] += 1
            max_offset = max((o for _, o in denial["pairs"]), default=0)
            if last_step is None or denial["step"] + max_offset > last_step:
                stats["prediction_inconclusive"] += 1
                continue
            confirmed = any(
                tile in step_to_opponents.get(denial["step"] + offset, ())
                for tile, offset in denial["pairs"]
            )
            if confirmed:
                stats["prediction_confirmed"] += 1
            else:
                stats["prediction_false_positives"] += 1

        actions = list(world.replay["actions"][agent.name])
        for rec, act in zip(trace, actions):
            rec["chosen_action"] = act
        stats["bombs_placed"] += sum(1 for act in actions if act == "BOMB")

        self_kill = agent.statistics.get("suicides", 0) > 0
        if self_kill:
            self_kills.append({"round": round_index, "trace": trace})

    world.end()
    return self_kills, stats


def _print_trace_tail(trace, n=8):
    for rec in trace[-n:]:
        print(
            f"    step={rec['step']:>3} pos={rec['self_pos']} opponents={rec['opponents']} "
            f"nearest_opp_manhattan={rec['nearest_opp_manhattan']} "
            f"current_tile_in_danger={rec['current_tile_in_danger']} "
            f"nearest_threat_timer={rec['nearest_threat_timer']} "
            f"redundancy_trigger={rec['redundancy_trigger']} "
            f"legal_actions={rec['legal_actions']} chosen_action={rec['chosen_action']}"
        )


def smoke_test(label, checkpoint_path, opponents, n_rounds=5, seed=SEED):
    """Runs the same short rollout twice from fresh worlds and checks for
    identical self-kill rounds and an identical trace for the first shared
    one, confirming determinism.
    """
    print(f"\n=== SMOKE TEST ({label}, {n_rounds} rounds) ===")
    run1, _ = replay_checkpoint(checkpoint_path, opponents, n_rounds, seed, label)
    run2, _ = replay_checkpoint(checkpoint_path, opponents, n_rounds, seed, label)
    rounds1 = [r["round"] for r in run1]
    rounds2 = [r["round"] for r in run2]
    print(f"  run1 self-kill rounds: {rounds1}")
    print(f"  run2 self-kill rounds: {rounds2}")
    if rounds1 != rounds2:
        print("  FAIL: self-kill round sets differ between two independent replays -- not deterministic yet")
        return False
    if not rounds1:
        print("  no self-kill in either run within this short window -- inconclusive, not a failure")
        return True
    r1 = run1[0]["trace"]
    r2 = run2[0]["trace"]
    if len(r1) != len(r2):
        print(f"  FAIL: round {run1[0]['round']} trace length differs: {len(r1)} vs {len(r2)}")
        return False
    mismatches = [
        i for i, (a, b) in enumerate(zip(r1, r2))
        if a["self_pos"] != b["self_pos"] or a["chosen_action"] != b["chosen_action"]
    ]
    if mismatches:
        print(f"  FAIL: round {run1[0]['round']} diverges at step index {mismatches[0]}")
        return False
    print(f"  PASS: round {run1[0]['round']} reproduced identically across two independent replays ({len(r1)} steps)")
    return True


def _diagnose(trace):
    """Prints why the redundancy check did or did not prevent this self-kill:
    when danger and a nearby opponent first co-occurred, whether the fatal
    action was ever outside legal_actions, and when the opponent's Manhattan
    distance dropped into range.
    """
    first_trigger = next((r for r in trace if r["redundancy_trigger"]), None)
    bypass = [r for r in trace if r["chosen_action"] not in r["legal_actions"]]
    crossings = [
        (a, b) for a, b in zip(trace, trace[1:])
        if a["nearest_opp_manhattan"] is not None and b["nearest_opp_manhattan"] is not None
        and a["nearest_opp_manhattan"] > 2 >= b["nearest_opp_manhattan"]
    ]
    print(f"    redundancy_trigger first True at step={first_trigger['step'] if first_trigger else 'never'}")
    print(f"    mask bypass (chosen_action outside legal_actions): {len(bypass)} step(s)"
          + (f" -- {[b['step'] for b in bypass]}" if bypass else ""))
    if crossings:
        for a, b in crossings:
            print(
                f"    opponent distance crossed >2 -> <=2 at step {a['step']}->{b['step']} "
                f"(pos {a['self_pos']}->{b['self_pos']}, dist {a['nearest_opp_manhattan']}->{b['nearest_opp_manhattan']}, "
                f"current_tile_in_danger at crossing step: {b['current_tile_in_danger']})"
            )
    else:
        print("    no >2 -> <=2 opponent-distance crossing observed in this trace "
              "(opponent was already <=2, or never got that close)")


def main():
    all_smoke_ok = True
    for label, ckpt in CHECKPOINTS.items():
        all_smoke_ok &= smoke_test(label, ckpt, OPPONENTS, n_rounds=20)

    if not all_smoke_ok:
        print("\nSmoke test failed for at least one checkpoint -- stopping before the full case replay.")
        sys.exit(1)

    print("\nAll smoke tests passed (reproducible opponent). Running full replay (100 rounds/checkpoint)...")

    for label, ckpt in CHECKPOINTS.items():
        print(f"\n{'=' * 70}\n=== {label} ({ckpt.name}) full replay, seed={SEED}, n_rounds={N_ROUNDS} ===\n{'=' * 70}")
        self_kills, stats = replay_checkpoint(ckpt, OPPONENTS, N_ROUNDS, SEED, label)
        now_rounds = sorted(r["round"] for r in self_kills)

        print(f"self_kill_rate (deterministic-opponent replay) = {len(now_rounds)}/{N_ROUNDS} = {len(now_rounds)/N_ROUNDS:.3f}")
        print(f"self-kill rounds: {now_rounds}")
        bombs_placed_mean = stats["bombs_placed"] / stats["n_rounds"]
        denial_rate = (
            stats["bomb_denied_steps"] / stats["bomb_available_steps"] if stats["bomb_available_steps"] else 0.0
        )
        missed_opportunity_rate = (
            stats["missed_opportunity_steps"] / stats["old_legal_steps"] if stats["old_legal_steps"] else 0.0
        )
        print(f"bombs_placed_mean = {bombs_placed_mean:.3f} ({stats['bombs_placed']}/{stats['n_rounds']})")
        print(
            f"bomb_attempt_denial_rate = {denial_rate:.3f} "
            f"({stats['bomb_denied_steps']}/{stats['bomb_available_steps']} bomb_available steps)"
        )
        print(
            f"missed_opportunity_rate = {missed_opportunity_rate:.3f} "
            f"({stats['missed_opportunity_steps']}/{stats['old_legal_steps']} steps the old Manhattan gate allowed)"
        )

        danger_steps = stats["danger_steps"]
        new_trigger_rate = stats["new_trigger_steps"] / danger_steps if danger_steps else 0.0
        old_trigger_rate = stats["old_trigger_steps"] / danger_steps if danger_steps else 0.0
        trigger_effective_rate = (
            stats["trigger_effective_steps"] / stats["new_trigger_steps"] if stats["new_trigger_steps"] else 0.0
        )
        earlier_by = stats["earlier_by_steps"]
        mean_earlier_by = float(np.mean(earlier_by)) if earlier_by else float("nan")
        print(
            f"redundancy_trigger fire rate (of {danger_steps} danger_steps): "
            f"new={new_trigger_rate:.3f} ({stats['new_trigger_steps']}), old={old_trigger_rate:.3f} ({stats['old_trigger_steps']})"
        )
        print(
            f"  new_only={stats['new_only_trigger_steps']} old_only={stats['old_only_trigger_steps']} "
            f"both={stats['both_trigger_steps']}"
        )
        print(
            f"  trigger_effective_rate (new fires AND masks >=1 direction, of new_trigger_steps) = "
            f"{trigger_effective_rate:.3f} ({stats['trigger_effective_steps']}/{stats['new_trigger_steps']})"
        )
        print(
            f"  rounds where both old and new eventually fired: {len(earlier_by)}/{N_ROUNDS}; "
            f"new fires {mean_earlier_by:.2f} steps earlier than old would have on average "
            "(same trajectory, old is observational only here)"
        )

        conclusive = stats["prediction_confirmed"] + stats["prediction_false_positives"]
        misjudge_rate = stats["prediction_false_positives"] / conclusive if conclusive else float("nan")
        print(
            f"prediction misjudge_rate = {misjudge_rate:.3f} "
            f"({stats['prediction_false_positives']}/{conclusive} conclusively-checkable prediction-only denials) "
            f"[{stats['prediction_denials']} total denials, {stats['prediction_inconclusive']} inconclusive "
            f"(fuse window ran past round end)]"
        )

        for entry in self_kills:
            print(f"\n  --- {label} round {entry['round']} -- last steps ---")
            _print_trace_tail(entry["trace"])
            _diagnose(entry["trace"])

    common.ring_bell()


if __name__ == "__main__":
    main()
