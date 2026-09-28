"""Stage A: 50-coin training on coin-heaven.

Launch in the background; the script writes its own timestamped log under
agent_code/rhine/logs/.

Usage:
    nohup python -m agent_code.rhine.scripts.run_stage_a --ablation A3 \
        --total-timesteps 300000 > /dev/null 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common

ABLATION_REWARD_OVERRIDES = {
    "A1": {"STEP_COST": 0.0, "PROGRESS_WEIGHT": 0.0},  # Sparse: COIN_COLLECTED only.
    "A2": {"PROGRESS_WEIGHT": 0.0},                     # + step cost.
    "A3": {},                                           # + distance shaping (full default).
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["A1", "A2", "A3"], default="A3")
    parser.add_argument(
        "--total-timesteps", type=int, default=300_000,
        help="Rough training budget in environment steps; actual count depends on episode "
             "lengths. This default is a starting proposal, not a validated value -- "
             "revisit before actually launching Stage A.",
    )
    parser.add_argument("--eval-interval-rounds", type=int, default=50,
                         help="Run a periodic deterministic evaluation burst every N training rounds.")
    parser.add_argument("--eval-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file(f"stage_a_{args.ablation}")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Stage A ({args.ablation}) -- log file: {log_path}")
    common.log_print(log_file, f"metrics CSV: {csv_path}")
    common.log_print(log_file, f"args: {vars(args)}")

    # Per-ablation filename so concurrent runs don't overwrite each other;
    # task1_stage_a.pt is reserved for the chosen ablation.
    checkpoint_path = common.MODELS_DIR / f"task1_stage_a_{args.ablation}.pt"
    reward_override = ABLATION_REWARD_OVERRIDES[args.ablation]

    total_steps = 0
    round_idx = 0
    burst_seed = args.seed
    init_ckpt = None  # First burst starts from a fresh random init.

    while total_steps < args.total_timesteps:
        train_episodes, train_world = common.run_episodes(
            n_rounds=args.eval_interval_rounds, scenario="coin-heaven", seed=burst_seed,
            train=True, init_checkpoint=init_ckpt, save_checkpoint=str(checkpoint_path),
            reward_override=reward_override,
        )
        train_reward_mean = common.training_reward_mean(train_world)
        total_steps += sum(ep.steps for ep in train_episodes)
        round_idx += args.eval_interval_rounds
        burst_seed += 1  # Vary the map seed across bursts.
        init_ckpt = str(checkpoint_path)  # Resume from our own progress from here on.

        eval_episodes, _ = common.run_episodes(
            n_rounds=args.eval_rounds, scenario="coin-heaven", seed=args.eval_seed,
            train=False, init_checkpoint=str(checkpoint_path),
        )
        agg = common.aggregate_metrics(eval_episodes)
        agg["round"] = round_idx
        agg["total_steps"] = total_steps
        agg["train_reward_mean"] = train_reward_mean if train_reward_mean is not None else float("nan")
        # Charts are generated from this CSV, not from in-memory `agg`.
        common.write_metrics_csv_row(csv_path, agg)

        common.log_print(
            log_file,
            f"[eval @ round {round_idx}, step {total_steps}] "
            f"completion_rate={agg['completion_rate']:.2f} "
            f"steps_per_coin={agg['steps_per_coin_mean']:.1f} "
            f"wait_fraction={agg['wait_fraction_mean']:.3f} "
            f"train_reward_mean={agg['train_reward_mean']}",
        )

    common.plot_training_curves(
        csv_path, png_path, x_col="total_steps",
        y_cols=["completion_rate", "steps_per_coin_mean", "train_reward_mean"],
        title=f"Stage A ({args.ablation})",
    )
    common.log_print(log_file, f"Stage A loop finished. Chart: {png_path}")
    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
