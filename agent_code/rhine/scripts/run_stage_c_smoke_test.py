"""Stage C smoke test: 1x coin_collector_agent, the first opponent that moves
purposefully, drops bombs and competes for coins.

A short random-init run (structured like run_stage_b_smoke_test.py) that
sanity-checks environment/mask/reward mechanics before real training:
  - no exceptions
  - invalid_action_count: reported; the known opponent-collision race allows
    a small nonzero count, so only an abnormal spike is flagged
  - self_kill_rate: reported; first test of danger aggregation across two
    independent bomb sources
  - GOT_KILLED_BY_OPPONENT_PENALTY: first stage where it can fire; reported
    separately
  - coin_contested: occurrence rate reported
  - reward values: no NaN/inf, magnitude sanity check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import events as e
from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.scripts import common

OPPONENTS = ["coin_collector_agent"]
N_ROUNDS = 30
SEED = 456


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


def _install_coin_contested_recorder():
    recorded = []
    original = train_module.extract_semantic_state

    def wrapped(game_state):
        semantic = original(game_state)
        recorded.append(semantic.coin_contested)
        return semantic

    train_module.extract_semantic_state = wrapped

    def restore():
        train_module.extract_semantic_state = original

    return recorded, restore


def _install_kill_target_recorder():
    """Records (has_kill_target, expected_kill_value_at_target) for every
    extract_semantic_state() call, to sanity-check that kill values stay in
    [0, 1] (see kill_target_info()).
    """
    recorded = []
    original = train_module.extract_semantic_state

    def wrapped(game_state):
        semantic = original(game_state)
        recorded.append((semantic.has_kill_target, semantic.expected_kill_value_at_target))
        return semantic

    train_module.extract_semantic_state = wrapped

    def restore():
        train_module.extract_semantic_state = original

    return recorded, restore


def main():
    log_file, log_path = common.open_log_file("stage_c_smoke_test")
    common.log_print(log_file, f"Stage C smoke test -- log file: {log_path}")
    common.log_print(log_file, f"opponents={OPPONENTS} n_rounds={N_ROUNDS} seed={SEED}, fresh random init")
    common.log_print(
        log_file,
        f"key flags: ENABLE_NO_BOMB_WHEN_BOARD_CLEARED={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED} "
        f"ENABLE_OSCILLATION_BREAKER={cfg.ENABLE_OSCILLATION_BREAKER} "
        f"ENABLE_WASTEFUL_BOMB_PENALTY={cfg.REWARD_CONFIG.ENABLE_WASTEFUL_BOMB_PENALTY} "
        "(breaker is eval-only, not exercised in this train=True dry run; board-cleared "
        "flag stays at default False -- neither is being enabled for this smoke test)",
    )

    recorded_rewards, restore_rewards = _install_reward_recorder()
    recorded_coin_contested, restore_coin_contested = _install_coin_contested_recorder()
    recorded_kill_target, restore_kill_target = _install_kill_target_recorder()
    scratch_checkpoint = common.LOGS_DIR / "stage_c_smoke_scratch.pt"
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
        restore_rewards()
        restore_kill_target()
        restore_coin_contested()

    total_steps = sum(ep.steps for ep in episodes)
    total_invalid = sum(ep.invalid_action_count for ep in episodes)
    total_self_kill = sum(1 for ep in episodes if ep.self_kill)
    total_killed_by_opponent = sum(1 for ep in episodes if ep.got_killed_by_opponent)
    self_kill_rate = total_self_kill / len(episodes) if episodes else float("nan")
    killed_by_opponent_rate = total_killed_by_opponent / len(episodes) if episodes else float("nan")
    invalid_rate_per_step = total_invalid / total_steps if total_steps else float("nan")

    common.log_print(log_file, f"\nRounds run: {len(episodes)}  total_steps: {total_steps}")
    for ep in episodes:
        common.log_print(
            log_file,
            f"  round {ep.round_index}: steps={ep.steps} coins={ep.coins_collected} "
            f"crates={ep.crates_destroyed} invalid={ep.invalid_action_count} "
            f"self_kill={ep.self_kill} got_killed_by_opponent={ep.got_killed_by_opponent} "
            f"opponent_kills={ep.opponent_kills}",
        )

    common.log_print(
        log_file,
        f"\ninvalid_action_count: total={total_invalid} rate_per_step={invalid_rate_per_step:.4f} "
        "(opponent now moves/bombs purposefully; nonzero count consistent with the documented "
        "opponent-collision race condition would not by itself indicate a mask regression -- "
        "it is the known framework race)",
    )
    common.log_print(
        log_file,
        f"self_kill_rate: {self_kill_rate:.3f} ({total_self_kill}/{len(episodes)}) "
        f"(Stage A real-timing baseline: ~0.01, single peaceful opponent)",
    )
    common.log_print(
        log_file,
        f"got_killed_by_opponent_rate: {killed_by_opponent_rate:.3f} "
        f"({total_killed_by_opponent}/{len(episodes)}) "
        "(structurally 0 in Stage A/B since peaceful_agent never drops bombs -- first stage "
        "where this can be nonzero at all)",
    )

    all_rewards = np.array([r for _, r in recorded_rewards], dtype=np.float64)
    n_nonfinite = int(np.sum(~np.isfinite(all_rewards))) if len(all_rewards) else 0
    common.log_print(log_file, f"\nreward calls total: {len(recorded_rewards)}  non-finite: {n_nonfinite}")
    if len(all_rewards):
        common.log_print(
            log_file,
            f"reward min/mean/max = {all_rewards.min():.4f} / {all_rewards.mean():.4f} / {all_rewards.max():.4f}",
        )

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
        common.log_print(log_file, f"  step reward on trigger: {r:.4f}")
        if not np.isfinite(r):
            common.log_print(log_file, "FAIL: GOT_KILLED_BY_OPPONENT_PENALTY step produced non-finite reward")
            ok = False

    bomb_dropped_rewards = [r for events_list, r in recorded_rewards if e.BOMB_DROPPED in events_list]
    n_bomb_dropped_nonfinite = sum(1 for r in bomb_dropped_rewards if not np.isfinite(r))
    common.log_print(
        log_file,
        f"\nBOMB_DROPPED-transition rewards (covers wasteful-bomb-penalty path, exercising the "
        f"kill_target-motivated bomb_threatens_reachable_opponent() fix): {len(bomb_dropped_rewards)} calls, "
        f"non-finite={n_bomb_dropped_nonfinite}, "
        f"min/mean/max = {min(bomb_dropped_rewards):.4f} / {np.mean(bomb_dropped_rewards):.4f} / "
        f"{max(bomb_dropped_rewards):.4f}" if bomb_dropped_rewards else
        "\nBOMB_DROPPED-transition rewards: none recorded this run",
    )
    if n_bomb_dropped_nonfinite:
        common.log_print(log_file, "FAIL: non-finite reward on a BOMB_DROPPED transition")
        ok = False

    n_coin_contested = sum(1 for v in recorded_coin_contested if v)
    n_semantic_calls = len(recorded_coin_contested)
    coin_contested_rate = n_coin_contested / n_semantic_calls if n_semantic_calls else float("nan")
    common.log_print(
        log_file,
        f"\ncoin_contested: True on {n_coin_contested}/{n_semantic_calls} semantic-extraction calls "
        f"({coin_contested_rate:.4f})",
    )

    kill_values = np.array([v for _, v in recorded_kill_target], dtype=np.float64)
    n_kill_nonfinite = int(np.sum(~np.isfinite(kill_values))) if len(kill_values) else 0
    n_kill_out_of_range = int(np.sum((kill_values < 0.0) | (kill_values > 1.0))) if len(kill_values) else 0
    n_has_kill_target = sum(1 for has_target, _ in recorded_kill_target if has_target)
    common.log_print(
        log_file,
        f"\nexpected_kill_value_at_target: has_kill_target=True on {n_has_kill_target}/{len(recorded_kill_target)} "
        f"semantic-extraction calls; value range [{kill_values.min():.3f}, {kill_values.max():.3f}] "
        f"(non-finite={n_kill_nonfinite}, out-of-[0,1]={n_kill_out_of_range})",
    )
    if n_kill_nonfinite or n_kill_out_of_range:
        common.log_print(log_file, "FAIL: expected_kill_value_at_target out of the expected [0,1] range")
        ok = False

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

    common.log_print(log_file, "\nStage C smoke test: " + ("PASS" if ok else "FLAGGED -- see above"))
    log_file.close()
    common.ring_bell()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
