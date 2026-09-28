"""Task 4 smoke test: 3x rule_based_agent on classic, the first opponent that
bombs and pursues tactically. Sanity-checks environment/mask/reward/feature
mechanics (31-dim) before real training; structured like
run_stage_d_smoke_test.py.

A short random-init run with a scratch checkpoint path; no canonical
checkpoint is loaded or written.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.scripts import common

OPPONENTS = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]
N_ROUNDS = 30
SEED = 789

EXPECTED_FLAGS = {
    "ENABLE_BOMBING_PROGRESS_SHAPING": True,
    "ENABLE_STALL_PENALTY": False,
    "ENABLE_STALL_PENALTY_V2": True,
    "normalize_returns": True,
    "ENABLE_BOMBING_TARGET_SAFETY_FILTER": False,
    "ENABLE_BOMBING_TARGET_SCORING_FORMULA": True,
    # Task 4 keeps training, evaluation and submission config identical.
    "ENABLE_NO_BOMB_WHEN_BOARD_CLEARED": False,
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
    """Records (nearest_alive_opponent_distance, opponents_within_3,
    reachable_space, full_feature_vector) for every extract_semantic_state()
    call."""
    recorded = []
    original = train_module.extract_semantic_state

    def wrapped(game_state):
        semantic = original(game_state)
        from agent_code.rhine.features import features_from_semantic
        recorded.append((
            semantic.nearest_alive_opponent_distance,
            semantic.opponents_within_3,
            semantic.reachable_space,
            features_from_semantic(semantic),
        ))
        return semantic

    train_module.extract_semantic_state = wrapped

    def restore():
        train_module.extract_semantic_state = original

    return recorded, restore


def main():
    log_file, log_path = common.open_log_file("task4_smoke_test")
    common.log_print(log_file, f"Task 4 smoke test -- log file: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} n_rounds={N_ROUNDS} seed={SEED}, fresh random init")

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
        common.log_print(log_file, "switch check: all match expected Task 4 values")

    recorded_rewards, restore_rewards = _install_reward_recorder()
    recorded_semantic, restore_semantic = _install_semantic_recorder()
    scratch_checkpoint = common.LOGS_DIR / "task4_smoke_scratch.pt"

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
        "(contested_tile is the known same-tick multi-agent collision race, reported not judged; "
        "own_cause must be 0)",
    )
    if total_invalid_own != 0:
        common.log_print(log_file, "FAIL: invalid_action_own_cause_count != 0")
        ok = False

    common.log_print(
        log_file,
        f"self_kill_rate: {self_kill_rate:.3f} ({total_self_kill}/{len(episodes)}) "
        f"got_killed_by_opponent_rate: {killed_by_opponent_rate:.3f} ({total_killed_by_opponent}/{len(episodes)}) "
        "(rule_based_agent zero-shot reference: score_mean=5.08, got_killed=0.140, self_kill=0.150 -- "
        "not directly comparable, random-init here)",
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

    nearest_opp_dists = [v[0] for v in recorded_semantic]
    within_3_counts = [v[1] for v in recorded_semantic]
    reachable_spaces = np.array([v[2] for v in recorded_semantic], dtype=np.float64)
    feature_vectors = np.stack([v[3] for v in recorded_semantic]) if recorded_semantic else np.zeros((0, 31))

    n_semantic_calls = len(recorded_semantic)
    n_has_reachable_opp = sum(1 for d in nearest_opp_dists if d is not None)
    common.log_print(
        log_file,
        f"\nnearest_alive_opponent_distance: reachable on {n_has_reachable_opp}/{n_semantic_calls} calls",
    )
    reachable_dists = np.array([d for d in nearest_opp_dists if d is not None], dtype=np.float64)
    if len(reachable_dists):
        qs = np.quantile(reachable_dists, [0.0, 0.25, 0.5, 0.75, 1.0])
        common.log_print(
            log_file,
            f"  distribution (reachable only) min/p25/median/p75/max = "
            f"{qs[0]:.1f}/{qs[1]:.1f}/{qs[2]:.1f}/{qs[3]:.1f}/{qs[4]:.1f}",
        )

    within_3_arr = np.array(within_3_counts, dtype=np.float64)
    common.log_print(
        log_file,
        f"opponents_within_3: mean={within_3_arr.mean():.3f} "
        f"distribution of counts: {dict(zip(*np.unique(within_3_arr, return_counts=True)))}",
    )

    qs_rs = np.quantile(reachable_spaces, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    n_saturated = int(np.sum(reachable_spaces >= cfg.REACHABLE_SPACE_NORM))
    common.log_print(
        log_file,
        f"reachable_space: min/p25/median/p75/p90/max = "
        f"{qs_rs[0]:.0f}/{qs_rs[1]:.0f}/{qs_rs[2]:.0f}/{qs_rs[3]:.0f}/{qs_rs[4]:.0f}/{qs_rs[5]:.0f}; "
        f"saturated at 1.0 (>={cfg.REACHABLE_SPACE_NORM}): {n_saturated}/{n_semantic_calls} "
        f"({n_saturated / n_semantic_calls:.4f})",
    )

    n_dim = feature_vectors.shape[1] if feature_vectors.size else -1
    n_feature_nonfinite = int(np.sum(~np.isfinite(feature_vectors))) if feature_vectors.size else 0
    n_feature_out_of_range = (
        int(np.sum((feature_vectors < -1e-6) | (feature_vectors > 1.0 + 1e-6))) if feature_vectors.size else 0
    )
    common.log_print(
        log_file,
        f"\nfeature vector: dim={n_dim} (expected {cfg.MODEL_CONFIG.n_features}), "
        f"non-finite={n_feature_nonfinite}, out-of-[0,1]={n_feature_out_of_range}, "
        f"n_calls={feature_vectors.shape[0] if feature_vectors.size else 0}",
    )
    if n_dim != cfg.MODEL_CONFIG.n_features:
        common.log_print(log_file, f"FAIL: feature dim {n_dim} != {cfg.MODEL_CONFIG.n_features}")
        ok = False
    if n_feature_nonfinite or n_feature_out_of_range:
        common.log_print(log_file, "FAIL: feature vector has non-finite or out-of-[0,1] values")
        ok = False

    if total_steps < 1000:
        common.log_print(log_file, f"FLAG: total_steps={total_steps} lower than expected for {N_ROUNDS} rounds")
        ok = False

    common.log_print(log_file, "\nTask 4 smoke test: " + ("PASS" if ok else "FLAGGED -- see above"))
    log_file.close()
    common.ring_bell()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
