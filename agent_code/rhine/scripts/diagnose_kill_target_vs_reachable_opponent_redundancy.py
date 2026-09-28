"""Read-only investigation: is has_kill_target (#28) redundant with
has_reachable_opponent (#21), or does it carry independent information?

Runs a 100-episode evaluation against one coin_collector_agent for each of
task3_stage_c_v3_seed{0,1,2}.pt, recording #21, #28, #29-34 and the chosen
action every step.

Reports, per seed and pooled:
1. total steps; steps where #28 != #21 (count, %).
2. mismatches split into (a) #28=1 & #21=0, (b) #28=0 & #21=1.
3. whether #29-34 are all 0 whenever #28=0 (empirical construction check).
4. chosen-action distribution within (a) and (b).

Data only.

Usage:
  python -m agent_code.rhine.scripts.diagnose_kill_target_vs_reachable_opponent_redundancy [--seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import extract_semantic_state

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"]
EVAL_SEED = 1000
N_ROUNDS = 100
SEEDS = [0, 1, 2]

IDX_21 = 20  # has_reachable_opponent
IDX_28 = 27  # has_kill_target
IDX_29_34 = list(range(28, 34))  # nearest_kill_distance, kill_dir x4, expected_kill_value_at_target


def collect_steps(checkpoint_path, log_file):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    rows = []
    actions = []

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False, opponents=OPPONENTS)
    original_act = rhine_callbacks.act

    def instrumented_act(self, game_state):
        chosen = original_act(self, game_state)
        if game_state is not None:
            semantic = extract_semantic_state(game_state)
            features = features_from_semantic(semantic)
            rows.append(features)
            actions.append(chosen)
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

    common.log_print(log_file, f"  collected {len(rows)} steps over {N_ROUNDS} rounds")
    return np.array(rows, dtype=np.float64), np.array(actions, dtype=object)


def action_dist_str(actions_subset):
    n = len(actions_subset)
    if n == 0:
        return "n=0"
    counts = {a: int(np.sum(actions_subset == a)) for a in cfg.ACTIONS}
    parts = ", ".join(f"{a}={100*counts[a]/n:.1f}%" for a in cfg.ACTIONS)
    return f"n={n}  {parts}"


def analyze_seed(seed, log_file):
    ckpt = common.MODELS_DIR / f"task3_stage_c_v3_seed{seed}.pt"
    common.log_print(log_file, "\n" + "=" * 78)
    common.log_print(log_file, f"SEED {seed}  ({ckpt.name})")
    common.log_print(log_file, "=" * 78)

    feats, actions = collect_steps(ckpt, log_file)
    n_total = len(feats)

    f21 = feats[:, IDX_21]
    f28 = feats[:, IDX_28]
    mismatch = f21 != f28
    n_mismatch = int(mismatch.sum())

    cat_a = (f28 == 1) & (f21 == 0)  # kill target exists, opponent not "reachable"
    cat_b = (f28 == 0) & (f21 == 1)  # opponent "reachable", no kill target
    n_a, n_b = int(cat_a.sum()), int(cat_b.sum())

    f28_zero = f28 == 0
    n_f28_zero = int(f28_zero.sum())
    sub = feats[f28_zero][:, IDX_29_34]
    nonzero_mask = sub != 0
    n_counterexamples = int(np.any(nonzero_mask, axis=1).sum())
    per_dim_counterexamples = nonzero_mask.sum(axis=0)

    common.log_print(log_file, "\n--- Part 1: total steps / mismatch ---")
    common.log_print(log_file, f"  total_steps={n_total}  mismatch(#28!=#21)={n_mismatch} ({100*n_mismatch/n_total:.2f}%)")

    common.log_print(log_file, "\n--- Part 2: mismatch split ---")
    common.log_print(log_file, f"  (a) #28=1 & #21=0: {n_a} ({100*n_a/n_total:.2f}% of total, "
                                f"{100*n_a/n_mismatch if n_mismatch else float('nan'):.1f}% of mismatches)")
    common.log_print(log_file, f"  (b) #28=0 & #21=1: {n_b} ({100*n_b/n_total:.2f}% of total, "
                                f"{100*n_b/n_mismatch if n_mismatch else float('nan'):.1f}% of mismatches)")

    common.log_print(log_file, "\n--- Part 3: #28=0 steps, are #29-34 all zero? ---")
    common.log_print(log_file, f"  #28=0 steps: n={n_f28_zero}; counterexamples (any of #29-34 nonzero): {n_counterexamples}")
    if n_counterexamples:
        names = ["#29 nearest_kill_distance", "#30 kill_dir_up", "#31 kill_dir_down",
                 "#32 kill_dir_left", "#33 kill_dir_right", "#34 expected_kill_value_at_target"]
        for name, cnt in zip(names, per_dim_counterexamples):
            if cnt:
                common.log_print(log_file, f"    {name}: {int(cnt)} nonzero occurrences")

    common.log_print(log_file, "\n--- Part 4: chosen-action distribution within (a) and (b) ---")
    common.log_print(log_file, f"  (a) #28=1,#21=0: {action_dist_str(actions[cat_a])}")
    common.log_print(log_file, f"  (b) #28=0,#21=1: {action_dist_str(actions[cat_b])}")

    common.log_print(log_file, "\n--- SEED SUMMARY ---")
    common.log_print(
        log_file,
        f"  seed={seed} n_total={n_total} n_mismatch={n_mismatch} pct_mismatch={100*n_mismatch/n_total:.2f} "
        f"n_a={n_a} pct_a={100*n_a/n_total:.2f} n_b={n_b} pct_b={100*n_b/n_total:.2f} "
        f"n_counterexamples={n_counterexamples}",
    )
    return {
        "n_total": n_total, "n_mismatch": n_mismatch, "n_a": n_a, "n_b": n_b,
        "n_counterexamples": n_counterexamples,
        "actions_a": actions[cat_a], "actions_b": actions[cat_b],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    args = parser.parse_args()

    seeds_to_run = [args.seed] if args.seed is not None else SEEDS
    tag = f"seed{args.seed}" if args.seed is not None else "all"
    log_file, log_path = common.open_log_file(f"kill_target_vs_reachable_opponent_redundancy_{tag}")
    common.log_print(log_file, f"has_kill_target vs has_reachable_opponent redundancy audit -- log: {log_path}")
    common.log_print(log_file, f"eval_seed={EVAL_SEED} n_rounds={N_ROUNDS} opponents={OPPONENTS} breaker=on")

    for seed in seeds_to_run:
        analyze_seed(seed, log_file)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
