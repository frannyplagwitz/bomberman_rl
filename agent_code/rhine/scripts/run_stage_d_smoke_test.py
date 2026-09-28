"""Stage D smoke test: 3x coin_collector_agent, the first stage with several
independent opponent bomb sources (danger aggregation and the N+1-path escape
requirement with N > 1).

A short random-init run (structured like run_stage_c_smoke_test.py) with a
scratch checkpoint path; no canonical checkpoint is loaded or written. Checks:
  - key switches match their expected Stage D values
  - feature vector: 28-dim, no NaN/inf, all values in [0, 1]
  - invalid_action_own_cause_count must be 0; contested_tile (the known
    same-tick collision race) is reported only
  - self_kill_rate: reported
  - reward values: no NaN/inf, magnitude sanity check
  - coin_contested / kill_target occurrence rates: reported
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import events as e
from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.scripts import common

OPPONENTS = ["coin_collector_agent", "coin_collector_agent", "coin_collector_agent"]
N_ROUNDS = 30
SEED = 789

EXPECTED_FLAGS = {
    "ENABLE_BOMBING_PROGRESS_SHAPING": True,
    "ENABLE_STALL_PENALTY": False,
    "ENABLE_STALL_PENALTY_V2": True,
    "normalize_returns": True,
    "ENABLE_BOMBING_TARGET_SAFETY_FILTER": False,
    "ENABLE_BOMBING_TARGET_SCORING_FORMULA": True,
    "ENABLE_NO_BOMB_WHEN_BOARD_CLEARED": True,  # Explicit opt-in for this run.
}


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


def _install_semantic_recorder():
    """Records (coin_contested, has_kill_target, expected_kill_value_at_target,
    full_feature_vector) for every extract_semantic_state() call."""
    recorded = []
    original = train_module.extract_semantic_state

    def wrapped(game_state):
        semantic = original(game_state)
        from agent_code.rhine.features import features_from_semantic
        recorded.append((
            semantic.coin_contested,
            semantic.has_kill_target,
            semantic.expected_kill_value_at_target,
            features_from_semantic(semantic),
        ))
        return semantic

    train_module.extract_semantic_state = wrapped

    def restore():
        train_module.extract_semantic_state = original

    return recorded, restore


def main():
    log_file, log_path = common.open_log_file("stage_d_smoke_test")
    common.log_print(log_file, f"Stage D smoke test -- log file: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} n_rounds={N_ROUNDS} seed={SEED}, fresh random init")

    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True  # Explicit opt-in, as in Stage D training.

    actual_flags = {
        "ENABLE_BOMBING_PROGRESS_SHAPING": cfg.REWARD_CONFIG.ENABLE_BOMBING_PROGRESS_SHAPING,
        "ENABLE_STALL_PENALTY": cfg.REWARD_CONFIG.ENABLE_STALL_PENALTY,
        "ENABLE_STALL_PENALTY_V2": cfg.REWARD_CONFIG.ENABLE_STALL_PENALTY_V2,
        "normalize_returns": cfg.PPO_CONFIG.normalize_returns,
        "ENABLE_BOMBING_TARGET_SAFETY_FILTER": cfg.ENABLE_BOMBING_TARGET_SAFETY_FILTER,
        "ENABLE_BOMBING_TARGET_SCORING_FORMULA": cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA,
        "ENABLE_NO_BOMB_WHEN_BOARD_CLEARED": cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED,
    }
    common.log_print(log_file, f"\nswitch check: {actual_flags}")
    mismatches = {k: (v, EXPECTED_FLAGS[k]) for k, v in actual_flags.items() if v != EXPECTED_FLAGS[k]}
    ok = True
    if mismatches:
        common.log_print(log_file, f"FAIL: switch mismatch (actual, expected): {mismatches}")
        ok = False
    else:
        common.log_print(log_file, "switch check: all match expected Stage D values")

    recorded_rewards, restore_rewards = _install_reward_recorder()
    recorded_semantic, restore_semantic = _install_semantic_recorder()
    scratch_checkpoint = common.LOGS_DIR / "stage_d_smoke_scratch.pt"

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
        restore_rewards()
        restore_semantic()

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)
    total_invalid_own = sum(ep.invalid_action_own_cause_count for ep in episodes)
    total_invalid_contested = sum(ep.invalid_action_contested_tile_count for ep in episodes)
    total_self_kill = sum(1 for ep in episodes if ep.self_kill)
    total_killed_by_opponent = sum(1 for ep in episodes if ep.got_killed_by_opponent)
    self_kill_rate = total_self_kill / len(episodes) if episodes else float("nan")
    killed_by_opponent_rate = total_killed_by_opponent / len(episodes) if episodes else float("nan")

    common.log_print(log_file, f"\nRounds run: {len(episodes)}  total_steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"crates={ep.crates_destroyed} invalid={ep.invalid_action_count} "
            f"(own={ep.invalid_action_own_cause_count} contested={ep.invalid_action_contested_tile_count}) "
            f"self_kill={ep.self_kill} got_killed_by_opponent={ep.got_killed_by_opponent} "
            f"opponent_kills={ep.opponent_kills}",
        )

    common.log_print(
        log_file,
        f"\ninvalid_action_own_cause_count: total={total_invalid_own} "
        f"invalid_action_contested_tile_count: total={total_invalid_contested} "
        "(contested_tile is the known same-tick multi-agent collision race, reported not judged -- "
        "own_cause must be 0)",
    )
    if total_invalid_own != 0:
        common.log_print(log_file, "FAIL: invalid_action_own_cause_count != 0")
        ok = False

    common.log_print(
        log_file,
        f"self_kill_rate: {self_kill_rate:.3f} ({total_self_kill}/{len(episodes)}) "
        "(Stage C real-timing baseline: single coin_collector_agent)",
    )
    common.log_print(
        log_file,
        f"got_killed_by_opponent_rate: {killed_by_opponent_rate:.3f} ({total_killed_by_opponent}/{len(episodes)})",
    )

    all_rewards = np.array([r for _, r in recorded_rewards], dtype=np.float64)
    n_nonfinite_reward = int(np.sum(~np.isfinite(all_rewards))) if len(all_rewards) else 0
    common.log_print(log_file, f"\nreward calls total: {len(recorded_rewards)}  non-finite: {n_nonfinite_reward}")
    if len(all_rewards):
        common.log_print(
            log_file,
            f"reward min/mean/max = {all_rewards.min():.4f} / {all_rewards.mean():.4f} / {all_rewards.max():.4f}",
        )
    if n_nonfinite_reward:
        common.log_print(log_file, "FAIL: non-finite reward encountered")
        ok = False

    killed_by_opponent_rewards = [
        r for events_list, r in recorded_rewards
        if e.GOT_KILLED in events_list and e.KILLED_SELF not in events_list
    ]
    common.log_print(
        log_file,
        f"\nGOT_KILLED_BY_OPPONENT_PENALTY triggers: {len(killed_by_opponent_rewards)} "
        f"(expected constant contribution: {cfg.REWARD_CONFIG.TRAINING_GOT_KILLED_BY_OPPONENT_PENALTY})",
    )
    for r in killed_by_opponent_rewards:
        if not np.isfinite(r):
            common.log_print(log_file, "FAIL: GOT_KILLED_BY_OPPONENT_PENALTY step produced non-finite reward")
            ok = False

    coin_contested_flags = [v[0] for v in recorded_semantic]
    kill_target_flags = [v[1] for v in recorded_semantic]
    kill_values = np.array([v[2] for v in recorded_semantic], dtype=np.float64)
    feature_vectors = np.stack([v[3] for v in recorded_semantic]) if recorded_semantic else np.zeros((0, 28))

    n_semantic_calls = len(recorded_semantic)
    n_coin_contested = sum(1 for v in coin_contested_flags if v)
    n_has_kill_target = sum(1 for v in kill_target_flags if v)
    common.log_print(
        log_file,
        f"\ncoin_contested: True on {n_coin_contested}/{n_semantic_calls} semantic-extraction calls "
        f"({n_coin_contested / n_semantic_calls if n_semantic_calls else float('nan'):.4f})",
    )
    common.log_print(
        log_file,
        f"has_kill_target: True on {n_has_kill_target}/{n_semantic_calls} semantic-extraction calls "
        f"({n_has_kill_target / n_semantic_calls if n_semantic_calls else float('nan'):.4f})",
    )

    n_kill_nonfinite = int(np.sum(~np.isfinite(kill_values))) if len(kill_values) else 0
    n_kill_out_of_range = int(np.sum((kill_values < 0.0) | (kill_values > 1.0))) if len(kill_values) else 0
    common.log_print(
        log_file,
        f"expected_kill_value_at_target range [{kill_values.min():.3f}, {kill_values.max():.3f}] "
        f"(non-finite={n_kill_nonfinite}, out-of-[0,1]={n_kill_out_of_range})",
    )
    if n_kill_nonfinite or n_kill_out_of_range:
        common.log_print(log_file, "FAIL: expected_kill_value_at_target out of the expected [0,1] range")
        ok = False

    n_dim = feature_vectors.shape[1] if feature_vectors.size else -1
    n_feature_nonfinite = int(np.sum(~np.isfinite(feature_vectors))) if feature_vectors.size else 0
    n_feature_out_of_range = (
        int(np.sum((feature_vectors < -1e-6) | (feature_vectors > 1.0 + 1e-6))) if feature_vectors.size else 0
    )
    common.log_print(
        log_file,
        f"\nfeature vector: dim={n_dim} (expected 28), non-finite={n_feature_nonfinite}, "
        f"out-of-[0,1]={n_feature_out_of_range}, n_calls={feature_vectors.shape[0] if feature_vectors.size else 0}",
    )
    if n_dim != 28:
        common.log_print(log_file, f"FAIL: feature dim {n_dim} != 28")
        ok = False
    if n_feature_nonfinite or n_feature_out_of_range:
        common.log_print(log_file, "FAIL: feature vector has non-finite or out-of-[0,1] values")
        ok = False

    if total_steps < 1000:
        common.log_print(log_file, f"FLAG: total_steps={total_steps} lower than expected for {N_ROUNDS} rounds")
        ok = False
    if self_kill_rate > 0.20:
        common.log_print(log_file, f"FLAG: self_kill_rate {self_kill_rate:.3f} looks abnormally high")
        ok = False

    common.log_print(log_file, "\nStage D smoke test: " + ("PASS" if ok else "FLAGGED -- see above"))
    log_file.close()
    common.ring_bell()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
