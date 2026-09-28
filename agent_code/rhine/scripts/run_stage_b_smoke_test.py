"""Stage B smoke test: 3x peaceful_agent, the first curriculum stage with
several simultaneous opponents.

A short random-init run (structured like task3_checkpoint0_smoke_test.py) that
sanity-checks environment/mask/reward mechanics before real training:
  - no exceptions
  - invalid_action_count: reported, not asserted zero, since the known
    opponent-collision race makes some invalid actions expected; flags only
    an abnormal spike
  - self_kill_rate: reported next to Stage A's real-timing baseline scale
  - reward values: no NaN/inf, magnitude sanity check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.scripts import common

OPPONENTS = ["peaceful_agent", "peaceful_agent", "peaceful_agent"]
N_ROUNDS = 20
SEED = 123


def _install_reward_recorder():
    recorded = []
    original = train_module.compute_reward

    def wrapped(old_game_state, self_action, new_game_state, events, *args, **kwargs):
        reward = original(old_game_state, self_action, new_game_state, events, *args, **kwargs)
        recorded.append((list(events), reward))
        return reward

    train_module.compute_reward = wrapped

    def restore():
        train_module.compute_reward = original

    return recorded, restore


def main():
    log_file, log_path = common.open_log_file("stage_b_smoke_test")
    common.log_print(log_file, f"Stage B smoke test -- log file: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} n_rounds={N_ROUNDS} seed={SEED}, fresh random init")

    recorded_rewards, restore = _install_reward_recorder()
    scratch_checkpoint = common.LOGS_DIR / "stage_b_smoke_scratch.pt"
    ok = True

    try:
        episodes, _ = common.run_episodes(
            n_rounds=N_ROUNDS,
            scenario="classic",
            seed=SEED,
            train=True,
            init_checkpoint=None,
            save_checkpoint=str(scratch_checkpoint),
            rollout_steps_override=512,
            opponents=OPPONENTS,
        )
    finally:
        restore()

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)
    total_self_kill = sum(1 for ep in episodes if ep.self_kill)
    self_kill_rate = total_self_kill / len(episodes) if episodes else float("nan")
    invalid_rate_per_step = total_invalid / total_steps if total_steps else float("nan")

    common.log_print(log_file, f"\nRounds run: {len(episodes)}  total_steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"crates={ep.crates_destroyed} invalid={ep.invalid_action_count} "
            f"self_kill={ep.self_kill} opponent_kills={ep.opponent_kills}",
        )

    common.log_print(
        log_file,
        f"\ninvalid_action_count: total={total_invalid} rate_per_step={invalid_rate_per_step:.4f} "
        f"(Stage A real-timing baseline: ~0.002-0.004 per step, 1 opponent)",
    )
    common.log_print(
        log_file,
        f"self_kill_rate: {self_kill_rate:.3f} ({total_self_kill}/{len(episodes)}) "
        f"(Stage A real-timing baseline: ~0.01)",
    )

    all_rewards = np.array([r for _, r in recorded_rewards], dtype=np.float64)
    n_nonfinite = int(np.sum(~np.isfinite(all_rewards))) if len(all_rewards) else 0
    common.log_print(log_file, f"\nreward calls total: {len(recorded_rewards)}  non-finite: {n_nonfinite}")
    if len(all_rewards):
        common.log_print(
            log_file,
            f"reward min/mean/max = {all_rewards.min():.4f} / {all_rewards.mean():.4f} / {all_rewards.max():.4f}",
        )

    if total_steps < 1000:
        common.log_print(log_file, f"FLAG: total_steps={total_steps} lower than expected for {N_ROUNDS} rounds")
        ok = False
    if n_nonfinite != 0:
        common.log_print(log_file, "FAIL: non-finite reward encountered")
        ok = False
    # Spike thresholds: well above what the known collision race produces, so
    # exceeding them points to a mask/escape-route regression.
    if invalid_rate_per_step > 0.20:
        common.log_print(log_file, f"FLAG: invalid_action rate {invalid_rate_per_step:.4f} looks abnormally high")
        ok = False
    if self_kill_rate > 0.20:
        common.log_print(log_file, f"FLAG: self_kill_rate {self_kill_rate:.3f} looks abnormally high")
        ok = False

    common.log_print(log_file, "\nStage B smoke test: " + ("PASS" if ok else "FLAGGED -- see above"))
    log_file.close()
    common.ring_bell()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
