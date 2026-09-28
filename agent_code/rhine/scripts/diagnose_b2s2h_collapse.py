"""Read-only diagnostic: periodic WAIT-collapse in the B2+S2+H training run.

For named "trough" (collapsed) and "peak"/"good" checkpoints, replays the
periodic-evaluation protocol through the real extraction/mask pipeline and
reconstructs the stall_history feature from the same persistent position
history act() keeps.

Reports per checkpoint:
  - standard metrics (wait_fraction, crates_destroyed, completion,
    oscillation_fraction)
  - fraction of steps with stall_history == 1
  - WAIT steps split by stall_history
  - stall_history == 1 steps split by BOMB mask-illegal vs legal-but-unused
  - mean of each base feature, for a descriptive cross-checkpoint comparison
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.distributions import Categorical

from agent_code.rhine import config as cfg

cfg.ENABLE_STALL_HISTORY_FEATURE = True

from agent_code.rhine import config as cfg  # noqa: E402
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.model import masked_logits
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state, is_confined_to_small_range

EVAL_ROUNDS = 15
EVAL_SEED = 1000
SCENARIO = "loot-crate"

CHECKPOINTS = [
    ("round500 (trough)", "round500"),
    ("round550 (trough)", "round550"),
    ("round600 (recovered)", "round600"),
    ("round1100 (peak)", "round1100"),
    ("round1150 (declining, self-kill burst)", "round1150"),
    ("round1200 (trough, final/hard-cap)", "round1200"),
]


def replay_with_feature_trace(checkpoint_path):
    """Runs EVAL_ROUNDS deterministic rounds and returns per-step records,
    replaying the stall-history position deque act() keeps (never reset
    between rounds) independently of the live agent.
    """
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False)
    agent = world.agents[0]
    model = agent.backend.runner.fake_self.model

    history = deque(maxlen=4)  # Mirrors callbacks.setup()'s recent_positions_for_feature.
    records = []
    round_traces = []  # (round_idx, actions), for oscillation_fraction.

    for round_idx in range(1, EVAL_ROUNDS + 1):
        world.new_round()
        world.user_input = "WAIT"
        trace = []
        while world.running:
            state = world.get_state_for_agent(agent)
            rec = None
            if state is not None:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                history.append(semantic.self_pos)
                stall_history = is_confined_to_small_range(history)
                features = features_from_semantic(semantic, stall_history=stall_history)

                with torch.no_grad():
                    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
                    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                    logits, _ = model(state_t)
                    logits = masked_logits(logits, mask_t)
                    entropy = float(Categorical(logits=logits).entropy().item())

                rec = {
                    "round": round_idx, "step": state["step"], "pos": semantic.self_pos,
                    "stall_history": stall_history, "bomb_available": semantic.bomb_available,
                    "bomb_legal": bool(mask[cfg.ACTIONS.index("BOMB")]),
                    "features_0_20": features[:20].copy(),
                    "policy_entropy": entropy,
                }
            world.do_step("WAIT")
            if rec is not None:
                trace.append(rec)

        actions = list(world.replay["actions"][agent.name])
        for rec, a in zip(trace, actions):
            rec["chosen"] = a
        records.extend(trace)
        round_traces.append(actions)

    n_sustained = sum(
        1 for actions in round_traces if longest_oscillation_run(actions)[0] >= OSCILLATION_THRESHOLD
    )

    invalid = sum(agent.statistics.get("invalid", 0) for _ in [0])
    world.end()
    return records, n_sustained


def main():
    log_file, log_path = common.open_log_file("task2_diagnose_b2s2h_collapse")
    common.log_print(log_file, f"B2+S2+H collapse diagnosis -- log file: {log_path}")
    common.log_print(
        log_file,
        f"Protocol: {EVAL_ROUNDS} rounds, seed={EVAL_SEED}, deterministic policy, "
        f"scenario={SCENARIO} (matches the periodic-eval protocol used during training).",
    )

    snapshot_dir = common.MODELS_DIR / "snapshots"

    for label, round_tag in CHECKPOINTS:
        ckpt = snapshot_dir / f"task2_stage_a2_B2S2H_{round_tag}.pt"
        common.log_print(log_file, f"\n{'=' * 70}\n{label} -- {ckpt.name}\n{'=' * 70}")

        records, n_sustained = replay_with_feature_trace(ckpt)
        n_steps = len(records)

        wait_count = sum(1 for r in records if r["chosen"] == "WAIT")
        stall1_count = sum(1 for r in records if r["stall_history"])
        wait_and_stall1 = sum(1 for r in records if r["chosen"] == "WAIT" and r["stall_history"])
        wait_and_stall0 = sum(1 for r in records if r["chosen"] == "WAIT" and not r["stall_history"])

        stall1_bomb_illegal = sum(1 for r in records if r["stall_history"] and not r["bomb_legal"])
        stall1_bomb_legal = sum(1 for r in records if r["stall_history"] and r["bomb_legal"])

        mean_entropy_overall = float(np.mean([r["policy_entropy"] for r in records]))
        mean_entropy_stall1 = float(np.mean([r["policy_entropy"] for r in records if r["stall_history"]])) if stall1_count else float("nan")
        mean_entropy_stall0 = float(np.mean([r["policy_entropy"] for r in records if not r["stall_history"]]))

        common.log_print(
            log_file,
            f"  n_steps={n_steps}, oscillation_fraction={n_sustained}/{EVAL_ROUNDS}={n_sustained / EVAL_ROUNDS:.3f}\n"
            f"  wait_fraction(overall)={wait_count / n_steps:.3f}\n"
            f"  mean masked policy entropy (nats, max=ln(n_legal_actions)): overall={mean_entropy_overall:.4f}, "
            f"stall_history=1 steps={mean_entropy_stall1:.4f}, stall_history=0 steps={mean_entropy_stall0:.4f}\n"
            f"  dim21(stall_history)==1 fraction of all steps: {stall1_count / n_steps:.3f}\n"
            f"  of WAIT steps: stall_history=1 in {wait_and_stall1}/{wait_count if wait_count else 1} "
            f"({(wait_and_stall1 / wait_count if wait_count else float('nan')):.3f}), "
            f"stall_history=0 in {wait_and_stall0}/{wait_count if wait_count else 1}\n"
            f"  of stall_history==1 steps ({stall1_count}): BOMB mask-illegal in {stall1_bomb_illegal} "
            f"({(stall1_bomb_illegal / stall1_count if stall1_count else float('nan')):.3f}), "
            f"BOMB mask-legal(but not necessarily chosen) in {stall1_bomb_legal}",
        )

        feats = np.array([r["features_0_20"] for r in records])
        means = feats.mean(axis=0)
        stds = feats.std(axis=0)
        names = [
            "can_move_up", "can_move_down", "can_move_left", "can_move_right",
            "has_reachable_coin", "nearest_coin_distance",
            "coin_path_up", "coin_path_down", "coin_path_left", "coin_path_right",
            "bomb_available", "has_bombing_target", "nearest_bombing_distance",
            "bombing_path_up", "bombing_path_down", "bombing_path_left", "bombing_path_right",
            "crates_destructible_at_target", "current_tile_in_danger", "nearest_threat_timer",
        ]
        common.log_print(log_file, "  dims 1-20 mean(std) across this eval:")
        for i, name in enumerate(names):
            common.log_print(log_file, f"    #{i + 1:>2} {name:<32} mean={means[i]:.3f} std={stds[i]:.3f}")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
