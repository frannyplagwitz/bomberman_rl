"""Read-only investigation: is expected_kill_value_at_target much smaller in
scale than crates_destructible_at_target, and is the network less sensitive
to it than to the crate signal?

Runs a 100-episode evaluation against one coin_collector_agent for each of
task3_stage_c_v3_seed{0,1,2}.pt, recording the exact 34-dim input every step.

Four analyses per seed:
1. Per-dimension audit: theoretical range vs realized
   min/max/mean/std/nonzero fraction.
2. expected_kill_value_at_target distribution on has_kill_target=1 steps.
3. First-layer weight-column L2 norm x realized std per input, ranked, vs
   randomly initialized networks.
4. Pearson correlation between feature value and P(BOMB) on steps where the
   current tile is the target and BOMB is legal, for both the crate and the
   kill signal (a wider population than the earlier "not bombed" analysis).

Data only.

Usage:
  python -m agent_code.rhine.scripts.diagnose_kill_vs_crate_signal_sensitivity
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.distributions import Categorical

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.model import ActorCriticMLP, masked_logits
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import extract_semantic_state

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100
SEEDS = [0, 1, 2]
N_RANDOM_INITS = 20

DIM_META = [
    # (1-based #, name, normalization description, theoretical range)
    (1, "can_move_up", "binary", "[0,1]"),
    (2, "can_move_down", "binary", "[0,1]"),
    (3, "can_move_left", "binary", "[0,1]"),
    (4, "can_move_right", "binary", "[0,1]"),
    (5, "has_reachable_coin", "binary", "[0,1]"),
    (6, "nearest_reachable_coin_distance", "/32 (clipped to 1.0)", "[0,1]"),
    (7, "coin_path_up", "binary", "[0,1]"),
    (8, "coin_path_down", "binary", "[0,1]"),
    (9, "coin_path_left", "binary", "[0,1]"),
    (10, "coin_path_right", "binary", "[0,1]"),
    (11, "bomb_available", "binary", "[0,1]"),
    (12, "has_bombing_target", "binary", "[0,1]"),
    (13, "nearest_bombing_position_distance", "/32 (clipped to 1.0)", "[0,1]"),
    (14, "bombing_path_up", "binary", "[0,1]"),
    (15, "bombing_path_down", "binary", "[0,1]"),
    (16, "bombing_path_left", "binary", "[0,1]"),
    (17, "bombing_path_right", "binary", "[0,1]"),
    (18, "crates_destructible_at_target", "/12 (MAX_CRATES_PER_BOMB=4*BOMB_POWER)", "[0,1]"),
    (19, "current_tile_in_danger", "binary", "[0,1]"),
    (20, "nearest_threat_timer", "/4 (BOMB_TIMER)", "[0,1]"),
    (21, "has_reachable_opponent", "binary", "[0,1]"),
    (22, "nearest_opponent_distance", "/32 (clipped to 1.0)", "[0,1]"),
    (23, "opponent_direction_up", "binary", "[0,1]"),
    (24, "opponent_direction_down", "binary", "[0,1]"),
    (25, "opponent_direction_left", "binary", "[0,1]"),
    (26, "opponent_direction_right", "binary", "[0,1]"),
    (27, "coin_contested", "binary", "[0,1]"),
    (28, "has_kill_target", "binary", "[0,1]"),
    (29, "nearest_kill_distance", "/32 (clipped to 1.0)", "[0,1]"),
    (30, "kill_direction_up", "binary", "[0,1]"),
    (31, "kill_direction_down", "binary", "[0,1]"),
    (32, "kill_direction_left", "binary", "[0,1]"),
    (33, "kill_direction_right", "binary", "[0,1]"),
    (34, "expected_kill_value_at_target", "none (already in [0,1], max escape-difficulty)", "[0,1]"),
]

IDX_CRATES = 17   # feature index (0-based) for #18
IDX_KILLVAL = 33  # feature index (0-based) for #34
BOMB_IDX = cfg.ACTIONS.index("BOMB")


def _bomb_prob(model, features, mask):
    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        logits = masked_logits(logits, mask_t)
        probs = Categorical(logits=logits).probs.squeeze(0)
    return float(probs[BOMB_IDX].item())


def collect_steps(checkpoint_path, log_file, label):
    """Runs N_ROUNDS episodes, returns:
      feats: (n_steps, 34) array of the exact network-input vector
      bomb_probs: (n_steps,) masked P(BOMB) per step
      has_kill: (n_steps,) bool, has_crate: (n_steps,) bool
      kill_dist0_legal: (n_steps,) bool, "current tile is itself the kill target and BOMB mask-legal"
      crate_dist0_legal: (n_steps,) bool, same for crate/bombing target
    """
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    rows = []
    bomb_probs = []
    has_kill = []
    has_crate = []
    kill_opp = []
    crate_opp = []

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)

    original_act = rhine_callbacks.act

    def instrumented_act(self, game_state):
        chosen = original_act(self, game_state)
        if game_state is not None:
            semantic = extract_semantic_state(game_state)
            mask = mask_from_semantic(semantic)
            from agent_code.rhine.features import features_from_semantic
            features = features_from_semantic(semantic)
            p_bomb = _bomb_prob(self.model, features, mask)

            rows.append(features)
            bomb_probs.append(p_bomb)
            has_kill.append(bool(semantic.has_kill_target))
            has_crate.append(bool(semantic.has_bombing_target))
            kill_opp.append(bool(semantic.has_kill_target
                                  and semantic.nearest_kill_distance == 0
                                  and mask[BOMB_IDX]))
            crate_opp.append(bool(semantic.has_bombing_target
                                   and semantic.nearest_bombing_distance == 0
                                   and mask[BOMB_IDX]))
        return chosen

    rhine_callbacks.act = instrumented_act
    try:
        for round_idx in range(1, N_ROUNDS + 1):
            world.new_round()
            world.user_input = "WAIT"
            while world.running:
                common._disable_think_time_limit(world)
                world.do_step("WAIT")
    finally:
        rhine_callbacks.act = original_act

    common.log_print(log_file, f"  [{label}] collected {len(rows)} steps over {N_ROUNDS} rounds")
    return (
        np.array(rows, dtype=np.float64),
        np.array(bomb_probs, dtype=np.float64),
        np.array(has_kill, dtype=bool),
        np.array(has_crate, dtype=bool),
        np.array(kill_opp, dtype=bool),
        np.array(crate_opp, dtype=bool),
    )


def audit_table(feats, log_file):
    common.log_print(log_file, f"\n  {'#':>3} {'name':<34} {'norm':<32} {'range':<8} "
                                f"{'min':>7} {'max':>7} {'mean':>8} {'std':>8} {'nz%':>7}")
    for i, (num, name, norm, rng) in enumerate(DIM_META):
        col = feats[:, i]
        nz = 100.0 * np.mean(col != 0)
        marker = " <==" if num in (18, 34) else ""
        common.log_print(
            log_file,
            f"  {num:>3} {name:<34} {norm:<32} {rng:<8} "
            f"{col.min():>7.4f} {col.max():>7.4f} {col.mean():>8.4f} {col.std():>8.4f} {nz:>6.1f}%{marker}",
        )


def kill_bucket_distribution(feats, has_kill, log_file):
    vals = feats[has_kill, IDX_KILLVAL]
    n = len(vals)
    common.log_print(log_file, f"\n  has_kill_target=1 steps: n={n}")
    if n == 0:
        return
    buckets = [0.0, 0.25, 0.5, 0.75, 1.0]
    for b in buckets:
        frac = 100.0 * np.mean(np.isclose(vals, b, atol=1e-6))
        common.log_print(log_file, f"    value=={b:.2f}: {frac:.1f}%")
    other = 100.0 * np.mean(~np.isin(np.round(vals, 6), np.round(buckets, 6)))
    common.log_print(log_file, f"    other (not exactly on grid): {other:.1f}%  "
                                f"(mean={vals.mean():.3f} std={vals.std():.3f})")


def layer1_weight_analysis(checkpoint_path, feats, log_file):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    w = checkpoint["model_state_dict"]["trunk.0.weight"].numpy()  # (64, 34)
    l2 = np.linalg.norm(w, axis=0)  # (34,)
    stds = feats.std(axis=0)
    score = l2 * stds

    n_features = w.shape[1]
    n_actions = cfg.MODEL_CONFIG.n_actions
    hidden_sizes = cfg.MODEL_CONFIG.hidden_sizes
    rand_l2 = np.zeros(n_features)
    for seed in range(N_RANDOM_INITS):
        torch.manual_seed(seed)
        m = ActorCriticMLP(n_features=n_features, n_actions=n_actions, hidden_sizes=hidden_sizes)
        rand_l2 += np.linalg.norm(m.trunk[0].weight.detach().numpy(), axis=0)
    rand_l2 /= N_RANDOM_INITS
    rand_score = rand_l2 * stds

    order = np.argsort(-score)
    common.log_print(log_file, f"\n  {'rank':>4} {'#':>3} {'name':<34} {'L2':>8} {'std':>8} "
                                f"{'L2xstd':>9} {'rand_L2':>8} {'rand_L2xstd':>12}")
    for rank, i in enumerate(order, start=1):
        num, name, _, _ = DIM_META[i]
        marker = " <==" if num in (18,) or 28 <= num <= 34 else ""
        common.log_print(
            log_file,
            f"  {rank:>4} {num:>3} {name:<34} {l2[i]:>8.4f} {stds[i]:>8.4f} "
            f"{score[i]:>9.4f} {rand_l2[i]:>8.4f} {rand_score[i]:>12.4f}{marker}",
        )
    return l2, stds, score, rand_l2, rand_score


def correlation_baseline(feats, bomb_probs, kill_opp, crate_opp, log_file):
    def _corr(mask, idx, name):
        n = int(mask.sum())
        if n < 2:
            common.log_print(log_file, f"    {name}: n={n} (insufficient for correlation)")
            return float("nan"), n
        x = feats[mask, idx]
        y = bomb_probs[mask]
        if x.std() < 1e-9 or y.std() < 1e-9:
            common.log_print(log_file, f"    {name}: n={n} (zero variance, corr undefined)")
            return float("nan"), n
        r = float(np.corrcoef(x, y)[0, 1])
        common.log_print(log_file, f"    {name}: n={n} r={r:.3f}")
        return r, n

    common.log_print(log_file, "\n  crate baseline (current tile IS bombing target, BOMB mask-legal):")
    r_crate, n_crate = _corr(crate_opp, IDX_CRATES, "crates_destructible_at_target vs P(BOMB)")
    common.log_print(log_file, "\n  kill signal, same population structure (current tile IS kill target, BOMB mask-legal):")
    r_kill, n_kill = _corr(kill_opp, IDX_KILLVAL, "expected_kill_value_at_target vs P(BOMB)")
    return r_crate, n_crate, r_kill, n_kill


def run_one_seed(seed, log_file):
    ckpt = common.MODELS_DIR / f"task3_stage_c_v3_seed{seed}.pt"
    label = f"seed{seed}"
    common.log_print(log_file, "\n" + "=" * 78)
    common.log_print(log_file, f"SEED {seed}  ({ckpt.name})")
    common.log_print(log_file, "=" * 78)

    feats, bomb_probs, has_kill, has_crate, kill_opp, crate_opp = collect_steps(ckpt, log_file, label)

    common.log_print(log_file, "\n--- Part 1: per-dimension audit ---")
    audit_table(feats, log_file)

    common.log_print(log_file, "\n--- Part 2: expected_kill_value_at_target bucket distribution (has_kill_target=1) ---")
    kill_bucket_distribution(feats, has_kill, log_file)

    common.log_print(log_file, "\n--- Part 3: first-layer weight-column L2 norm x realized std ---")
    l2, stds, score, rand_l2, rand_score = layer1_weight_analysis(ckpt, feats, log_file)

    common.log_print(log_file, "\n--- Part 4: crate baseline vs kill signal, same population/method ---")
    r_crate, n_crate, r_kill, n_kill = correlation_baseline(feats, bomb_probs, kill_opp, crate_opp, log_file)

    common.log_print(log_file, "\n--- SEED SUMMARY ---")
    common.log_print(
        log_file,
        f"  seed={seed} n_steps={len(feats)} "
        f"#18_std={stds[IDX_CRATES]:.4f} #34_std={stds[IDX_KILLVAL]:.4f} "
        f"#18_L2xstd={score[IDX_CRATES]:.4f} #34_L2xstd={score[IDX_KILLVAL]:.4f} "
        f"#18_rank={int(np.sum(score > score[IDX_CRATES]) + 1)} "
        f"#34_rank={int(np.sum(score > score[IDX_KILLVAL]) + 1)} "
        f"r_crate={r_crate:.3f}(n={n_crate}) r_kill={r_kill:.3f}(n={n_kill})",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None,
                         help="Run only this training seed's checkpoint (for parallel invocation). Default: all seeds sequentially.")
    args = parser.parse_args()

    seeds_to_run = [args.seed] if args.seed is not None else SEEDS
    tag = f"seed{args.seed}" if args.seed is not None else "all"
    log_file, log_path = common.open_log_file(f"kill_vs_crate_signal_sensitivity_{tag}")
    common.log_print(log_file, f"Kill vs crate signal sensitivity audit -- log: {log_path}")
    common.log_print(log_file, f"eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} opponents={OPPONENTS} breaker=on")

    for seed in seeds_to_run:
        run_one_seed(seed, log_file)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
