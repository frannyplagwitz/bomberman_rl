"""Checkpoint 0 smoke test.

Runs a few rounds of solo rhine training on `coin-heaven` and checks:
  - no exceptions raised
  - invalid_action_count == 0
  - reward values are finite and within a sane magnitude

Uses a small rollout_steps override so the PPO update path is exercised too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common


def main():
    log_file, log_path = common.open_log_file("checkpoint0")
    common.log_print(log_file, f"Checkpoint 0 smoke test -- log file: {log_path}")

    n_rounds = 5
    checkpoint_path = common.MODELS_DIR / "checkpoint0_smoke.pt"

    episodes, world = common.run_episodes(
        n_rounds=n_rounds,
        scenario="coin-heaven",
        seed=123,
        train=True,
        init_checkpoint=None,          # fresh random init
        save_checkpoint=str(checkpoint_path),
        rollout_steps_override=64,     # small, so a PPO update fires
    )

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)

    common.log_print(log_file, f"Rounds run: {len(episodes)}")
    common.log_print(log_file, f"Total steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"completed={ep.completed} invalid={ep.invalid_action_count} "
            f"wait_fraction={ep.wait_fraction:.3f} reverse={ep.immediate_reverse_count}",
        )

    fake_self = world.agents[0].backend.runner.fake_self
    reward_history = getattr(fake_self, "reward_history", [])
    rewards = np.array(reward_history, dtype=np.float64)

    common.log_print(log_file, f"\nreward count = {len(rewards)}")
    if len(rewards) > 0:
        common.log_print(
            log_file,
            f"reward min/mean/max = {rewards.min():.4f} / {rewards.mean():.4f} / {rewards.max():.4f}",
        )
    n_nonfinite = int(np.sum(~np.isfinite(rewards))) if len(rewards) else 0
    common.log_print(log_file, f"non-finite rewards = {n_nonfinite}")
    common.log_print(log_file, f"invalid_action_count (total) = {total_invalid}")

    ok = True
    if total_invalid != 0:
        common.log_print(log_file, "FAIL: invalid_action_count != 0")
        ok = False
    if n_nonfinite != 0:
        common.log_print(log_file, "FAIL: non-finite reward encountered")
        ok = False
    if len(rewards) and (rewards.max() >= 2.0 or rewards.min() <= -2.0):
        # Per-step rewards should stay well below this bound under the default reward config.
        common.log_print(log_file, f"FAIL: reward magnitude looks anomalous (min={rewards.min()}, max={rewards.max()})")
        ok = False

    common.log_print(log_file, "\nCheckpoint 0: " + ("PASS" if ok else "FAIL"))
    log_file.close()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
