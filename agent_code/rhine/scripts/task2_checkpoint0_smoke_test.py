"""Task 2 checkpoint 0 smoke test.

Runs a few rounds of solo rhine training on `loot-crate` and checks:
  - no exceptions raised
  - invalid_action_count == 0 (mask correct for movement/BOMB/WAIT)
  - self_kill_rate == 0 (BOMB escape-route judgment correct)
  - reward values are finite and within a sane magnitude

Uses a small rollout_steps override so the PPO update path is exercised.
B1 (shaping off) and B2 (shaping on) both run with the historical Task 2
baseline pinned explicitly (V1 stall penalty on, V2 off).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common


def run_burst(log_file, label, reward_override, n_rounds=30):
    checkpoint_path = common.MODELS_DIR / f"checkpoint0_smoke_task2_{label}.pt"

    episodes, world = common.run_episodes(
        n_rounds=n_rounds,
        scenario="loot-crate",
        seed=123,
        train=True,
        init_checkpoint=None,          # fresh random init
        save_checkpoint=str(checkpoint_path),
        rollout_steps_override=64,     # small, so a PPO update fires
        reward_override=reward_override,
    )

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)
    # environment.py already sums per-round statistics into round_statistics.
    total_suicides = sum(world.round_statistics[r]["suicides"] for r in world.round_statistics)

    common.log_print(log_file, f"\n=== {label} ===")
    common.log_print(log_file, f"Rounds run: {len(episodes)}")
    common.log_print(log_file, f"Total steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"invalid={ep.invalid_action_count} wait_fraction={ep.wait_fraction:.3f} "
            f"reverse={ep.immediate_reverse_count}",
        )

    fake_self = world.agents[0].backend.runner.fake_self
    reward_history = getattr(fake_self, "reward_history", [])
    rewards = np.array(reward_history, dtype=np.float64)

    common.log_print(log_file, f"reward count = {len(rewards)}")
    if len(rewards) > 0:
        common.log_print(
            log_file,
            f"reward min/mean/max = {rewards.min():.4f} / {rewards.mean():.4f} / {rewards.max():.4f}",
        )
    n_nonfinite = int(np.sum(~np.isfinite(rewards))) if len(rewards) else 0
    common.log_print(log_file, f"non-finite rewards = {n_nonfinite}")
    common.log_print(log_file, f"invalid_action_count (total) = {total_invalid}")
    common.log_print(log_file, f"self_kill_rate = {total_suicides}/{len(episodes)}")

    ok = True
    if total_invalid != 0:
        common.log_print(log_file, "FAIL: invalid_action_count != 0")
        ok = False
    if total_suicides != 0:
        common.log_print(log_file, "FAIL: self_kill_rate != 0")
        ok = False
    if n_nonfinite != 0:
        common.log_print(log_file, "FAIL: non-finite reward encountered")
        ok = False
    if len(rewards) and (rewards.max() >= 2.0 or rewards.min() <= -2.0):
        # Per-step rewards should stay well below this bound under this reward config.
        common.log_print(log_file, f"FAIL: reward magnitude looks anomalous (min={rewards.min()}, max={rewards.max()})")
        ok = False

    common.log_print(log_file, f"{label}: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    log_file, log_path = common.open_log_file("task2_checkpoint0")
    common.log_print(log_file, f"Task 2 checkpoint 0 smoke test -- log file: {log_path}")

    plain_baseline = {"ENABLE_BOMBING_PROGRESS_SHAPING": False, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False}
    ok_b1 = run_burst(log_file, "B1", reward_override=plain_baseline)
    ok_b2 = run_burst(log_file, "B2", reward_override={**plain_baseline, "ENABLE_BOMBING_PROGRESS_SHAPING": True})

    overall_ok = ok_b1 and ok_b2
    common.log_print(log_file, "\nTask 2 Checkpoint 0 (B1+B2): " + ("PASS" if overall_ok else "FAIL"))
    log_file.close()

    common.ring_bell()

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
