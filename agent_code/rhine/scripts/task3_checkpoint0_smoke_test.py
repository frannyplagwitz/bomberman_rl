"""Task 3 checkpoint 0 smoke test (the Task 3 counterpart of
task2_checkpoint0_smoke_test.py).

Runs a short random-init training burst on `classic` against one
peaceful_agent and two coin_collector_agent, so kill/death events are likely
to occur. Uses a small rollout_steps override and a scratch checkpoint path
under logs/.

Checks:
  - no exceptions raised
  - invalid_action_own_cause_count == 0; contested_tile is reported only
    (see state_processing.classify_invalid_action())
  - self_kill_rate == 0 (escape BFS with opponents as obstacles)
  - every recorded coin_contested value and 28-dim feature vector is finite
    and in range
  - rewards from real KILLED_OPPONENT/GOT_KILLED/KILLED_SELF events match the
    expected magnitudes
  - checkpoint save to the scratch path and a fresh load succeed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.scripts import common

N_FEATURES = cfg.n_features_active()
assert N_FEATURES == 28, f"expected the Task 3 base 28-dim vector, got {N_FEATURES}"


def _install_recorders():
    """Patches the names bound inside train.py's namespace, since train.py's
    `from .x import y` bindings are unaffected by patching the source modules.
    Returns (recorded_semantics, recorded_reward_calls, restore_fn).
    """
    recorded_semantics = []
    recorded_reward_calls = []

    original_extract = train_module.extract_semantic_state
    original_compute_reward = train_module.compute_reward

    def wrapped_extract(game_state):
        semantic = original_extract(game_state)
        recorded_semantics.append(semantic)
        return semantic

    def wrapped_compute_reward(old_game_state, self_action, new_game_state, events, *args, **kwargs):
        reward = original_compute_reward(old_game_state, self_action, new_game_state, events, *args, **kwargs)
        recorded_reward_calls.append((list(events), reward))
        return reward

    train_module.extract_semantic_state = wrapped_extract
    train_module.compute_reward = wrapped_compute_reward

    def restore():
        train_module.extract_semantic_state = original_extract
        train_module.compute_reward = original_compute_reward

    return recorded_semantics, recorded_reward_calls, restore


def main():
    log_file, log_path = common.open_log_file("task3_checkpoint0")
    common.log_print(log_file, f"Task 3 checkpoint 0 smoke test -- log file: {log_path}")
    common.log_print(log_file, f"n_features_active() = {N_FEATURES}")

    recorded_semantics, recorded_reward_calls, restore = _install_recorders()
    ok = True

    scratch_checkpoint = common.LOGS_DIR / "task3_checkpoint0_smoke_scratch.pt"
    try:
        episodes, _ = common.run_episodes(
            n_rounds=15,
            scenario="classic",
            seed=123,
            train=True,
            init_checkpoint=None,  # fresh random init
            save_checkpoint=str(scratch_checkpoint),
            rollout_steps_override=512,  # small, so PPO updates fire
            opponents=["peaceful_agent", "coin_collector_agent", "coin_collector_agent"],
        )
    finally:
        restore()

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)
    # own_cause is a mask failure (must be 0); contested_tile is the known
    # same-tick race, reported only.
    total_invalid_own_cause = sum(ep.invalid_action_own_cause_count for ep in episodes)
    total_invalid_contested_tile = sum(ep.invalid_action_contested_tile_count for ep in episodes)
    # Uses our agent's own self_kill flag; round_statistics["suicides"] also
    # counts opponents.
    total_suicides = sum(1 for ep in episodes if ep.self_kill)

    common.log_print(log_file, f"\nRounds run: {len(episodes)}")
    common.log_print(log_file, f"Total steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"crates={ep.crates_destroyed} invalid={ep.invalid_action_count} "
            f"(own_cause={ep.invalid_action_own_cause_count} contested_tile={ep.invalid_action_contested_tile_count}) "
            f"wait_fraction={ep.wait_fraction:.3f} self_kill={ep.self_kill}",
        )

    if total_steps < 1000:
        common.log_print(log_file, f"FAIL: total_steps={total_steps} is far short of the requested few-thousand-step scale")
        ok = False
    common.log_print(
        log_file,
        f"\ninvalid_action total={total_invalid} own_cause={total_invalid_own_cause} "
        f"contested_tile={total_invalid_contested_tile}",
    )
    if total_invalid_own_cause != 0:
        common.log_print(log_file, f"FAIL: invalid_action_own_cause_count != 0 (total={total_invalid_own_cause})")
        ok = False
    if total_suicides != 0:
        common.log_print(log_file, f"FAIL: self_kill_rate != 0 (total={total_suicides})")
        ok = False

    # Feature sanity: every recorded 28-dim vector must be finite and in [0, 1].
    common.log_print(log_file, f"\nSemanticState samples recorded: {len(recorded_semantics)}")
    n_contested = sum(1 for s in recorded_semantics if s.coin_contested)
    common.log_print(log_file, f"  coin_contested=True: {n_contested}/{len(recorded_semantics)}")

    n_bad_feature_vectors = 0
    for semantic in recorded_semantics:
        features = features_from_semantic(semantic)
        if features.shape != (28,) or not np.all(np.isfinite(features)) or np.any(features < 0.0) or np.any(features > 1.0):
            n_bad_feature_vectors += 1

    common.log_print(log_file, f"Malformed 28-dim feature vectors (NaN/inf/out-of-[0,1]): {n_bad_feature_vectors}")
    if n_bad_feature_vectors != 0:
        common.log_print(log_file, "FAIL: malformed feature vector encountered")
        ok = False

    # Reward wiring: rewards from real kill/death events must match the expected magnitudes.
    import events as e

    death_related = [
        (events, reward) for events, reward in recorded_reward_calls
        if e.KILLED_OPPONENT in events or e.GOT_KILLED in events or e.KILLED_SELF in events
    ]
    common.log_print(log_file, f"\nReward calls total: {len(recorded_reward_calls)}")
    common.log_print(log_file, f"Reward calls with a death-related event: {len(death_related)}")

    n_reward_mismatches = 0
    for events, reward in death_related:
        n_killed_opponent = events.count(e.KILLED_OPPONENT)
        expected = cfg.REWARD_CONFIG.STEP_COST + n_killed_opponent * cfg.REWARD_CONFIG.TRAINING_KILLED_OPPONENT_REWARD
        if e.KILLED_SELF in events:
            expected += cfg.REWARD_CONFIG.SELF_KILL_PENALTY
        elif e.GOT_KILLED in events:
            expected += cfg.REWARD_CONFIG.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY
        # Real transitions may include shaping/stall terms, so bound the
        # unexplained residual instead of asserting equality.
        residual = reward - expected
        if abs(residual) > 0.5:  # Generous bound for all other terms combined.
            n_reward_mismatches += 1
            common.log_print(log_file, f"  MISMATCH events={events} reward={reward:.4f} expected~={expected:.4f} residual={residual:.4f}")
        else:
            common.log_print(log_file, f"  ok events={events} reward={reward:.4f} expected~={expected:.4f}")

    if n_reward_mismatches != 0:
        common.log_print(log_file, f"FAIL: {n_reward_mismatches} death-related reward(s) deviate from the expected magnitude by more than the residual bound")
        ok = False

    all_rewards = np.array([r for _, r in recorded_reward_calls], dtype=np.float64)
    n_nonfinite = int(np.sum(~np.isfinite(all_rewards))) if len(all_rewards) else 0
    common.log_print(log_file, f"\nnon-finite rewards (all calls) = {n_nonfinite}")
    if n_nonfinite != 0:
        common.log_print(log_file, "FAIL: non-finite reward encountered")
        ok = False
    if len(all_rewards):
        common.log_print(log_file, f"reward min/mean/max = {all_rewards.min():.4f} / {all_rewards.mean():.4f} / {all_rewards.max():.4f}")

    # --- Checkpoint save/load check (scratch path only). ---
    import torch
    if not scratch_checkpoint.is_file():
        common.log_print(log_file, f"FAIL: scratch checkpoint was not written to {scratch_checkpoint}")
        ok = False
    else:
        checkpoint = torch.load(scratch_checkpoint, map_location="cpu")
        saved_n_features = checkpoint["model_state_dict"]["trunk.0.weight"].shape[1]
        common.log_print(log_file, f"Scratch checkpoint written: {scratch_checkpoint} (saved n_features={saved_n_features})")
        if saved_n_features != 28:
            common.log_print(log_file, f"FAIL: saved checkpoint's input dim is {saved_n_features}, expected 28")
            ok = False

        # Load it back through a second fresh burst to test callbacks.setup()'s load path.
        try:
            episodes2, world2 = common.run_episodes(
                n_rounds=1,
                scenario="classic",
                seed=124,
                train=False,
                init_checkpoint=str(scratch_checkpoint),
                opponents=["peaceful_agent", "coin_collector_agent", "coin_collector_agent"],
            )
            common.log_print(log_file, f"Reload check: 1 round ran successfully from the scratch checkpoint, steps={episodes2[0].steps}")
        except Exception as exc:
            common.log_print(log_file, f"FAIL: reloading the scratch checkpoint raised: {exc!r}")
            ok = False

    common.log_print(log_file, f"\nDefault checkpoint path (cfg.STAGE_A_CHECKPOINT, NOT written by this smoke test) = {cfg.STAGE_A_CHECKPOINT}")

    common.log_print(log_file, "\nTask 3 Checkpoint 0: " + ("PASS" if ok else "FAIL"))
    log_file.close()

    common.ring_bell()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
