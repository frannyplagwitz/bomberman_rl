"""Tests whether "all coins collected, crates remaining" is a rare state in
PPO's training data, which could explain why the policy never learned to
handle it.

Samples stochastic (train=True, as in training) rollouts from B2 snapshots
across its training history, under B2's reward config, and classifies every
step as:
  - "coins_left>0"                   -- coins still available
  - "coins_left==0, crates_left>0"   -- the state implicated in the oscillation
  - "coins_left==0, crates_left==0"  -- fully cleared

All checkpoints are written to a scratch path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from collections import Counter

from agent_code.rhine.scripts import common

SNAPSHOT_ROUNDS = [550, 700, 850, 1000, 1150, 1300, 1500]  # Spread across B2's training history.
STEPS_PER_SNAPSHOT = 8000
SCENARIO = "loot-crate"
TRAIN_SEED = 777  # Distinct from the evaluation seed used elsewhere.
SCRATCH_CKPT = common.LOGS_DIR / "verify_state_dist_scratch.pt"


def sample_from_snapshot(snapshot_path, step_budget, seed):
    counts = Counter()
    total = 0
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(snapshot_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", str(SCRATCH_CKPT))
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", '{"ENABLE_BOMBING_PROGRESS_SHAPING": true, "ENABLE_STALL_PENALTY": true, "ENABLE_STALL_PENALTY_V2": false}')

    world = common.build_world(scenario=SCENARIO, seed=seed, train=True)
    while total < step_budget:
        world.new_round()
        world.user_input = "WAIT"
        while world.running and total < step_budget:
            coins_left = sum(1 for c in world.coins if c.collectable)
            crates_left = int((world.arena == 1).sum())
            if coins_left > 0:
                counts["coins_left>0"] += 1
            elif crates_left > 0:
                counts["coins_left==0,crates_left>0"] += 1
            else:
                counts["coins_left==0,crates_left==0"] += 1
            total += 1
            world.do_step("WAIT")
    world.end()
    return counts, total


def main():
    log_file, log_path = common.open_log_file("task2_verify_state_distribution")
    common.log_print(log_file, f"State distribution verification (B2 training-mode rollouts) -- log file: {log_path}")
    common.log_print(log_file, f"snapshot_rounds={SNAPSHOT_ROUNDS} steps_per_snapshot={STEPS_PER_SNAPSHOT}")

    snapshot_dir = common.MODELS_DIR / "snapshots"
    grand_total = Counter()

    for round_num in SNAPSHOT_ROUNDS:
        snap = snapshot_dir / f"task2_stage_a_B2_round{round_num}.pt"
        counts, total = sample_from_snapshot(snap, STEPS_PER_SNAPSHOT, TRAIN_SEED + round_num)
        for k, v in counts.items():
            grand_total[k] += v
        pct_zero_coin = counts["coins_left==0,crates_left>0"] / total * 100
        common.log_print(
            log_file,
            f"  round={round_num:>4}: n={total} "
            f"coins_left>0={counts['coins_left>0']} ({counts['coins_left>0']/total*100:.2f}%) "
            f"coins==0&crates>0={counts['coins_left==0,crates_left>0']} ({pct_zero_coin:.2f}%) "
            f"fully_cleared={counts['coins_left==0,crates_left==0']} "
            f"({counts['coins_left==0,crates_left==0']/total*100:.2f}%)",
        )

    grand_n = sum(grand_total.values())
    common.log_print(log_file, f"\n=== aggregate across all {len(SNAPSHOT_ROUNDS)} snapshots, n={grand_n} ===")
    for k, v in grand_total.items():
        common.log_print(log_file, f"  {k}: {v} ({v/grand_n*100:.2f}%)")

    if SCRATCH_CKPT.exists():
        SCRATCH_CKPT.unlink()

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
