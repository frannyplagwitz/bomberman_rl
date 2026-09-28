"""Stage C: conditional fine-tune from the Stage A checkpoint on coin-heaven-9,
only if Stage B's zero-shot results are clearly inadequate. Reward settings
are config.py defaults.

Usage:
    nohup python -m agent_code.rhine.scripts.run_stage_c > /dev/null 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--eval-interval-rounds", type=int, default=50)
    parser.add_argument("--eval-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--eval-seed", type=int, default=4000)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file("stage_c")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Stage C fine-tune -- log file: {log_path}")
    common.log_print(log_file, f"metrics CSV: {csv_path}")
    common.log_print(log_file, f"args: {vars(args)}")

    stage_a_checkpoint = common.MODELS_DIR / "task1_stage_a.pt"
    stage_c_checkpoint = common.MODELS_DIR / "task1_stage_c.pt"
    if not stage_a_checkpoint.is_file():
        common.log_print(log_file, f"FAIL: {stage_a_checkpoint} not found -- Stage C warm-starts from Stage A.")
        log_file.close()
        raise FileNotFoundError(str(stage_a_checkpoint))

    total_steps = 0
    round_idx = 0
    burst_seed = args.seed
    init_ckpt = str(stage_a_checkpoint)  # First burst: warm start from Stage A.

    while total_steps < args.total_timesteps:
        train_episodes, train_world = common.run_episodes(
            n_rounds=args.eval_interval_rounds, scenario="coin-heaven-9", seed=burst_seed,
            train=True, init_checkpoint=init_ckpt, save_checkpoint=str(stage_c_checkpoint),
        )
        train_reward_mean = common.training_reward_mean(train_world)
        total_steps += sum(ep.steps for ep in train_episodes)
        round_idx += args.eval_interval_rounds
        burst_seed += 1
        init_ckpt = str(stage_c_checkpoint)  # Continue from our own progress.

        eval_episodes, _ = common.run_episodes(
            n_rounds=args.eval_rounds, scenario="coin-heaven-9", seed=args.eval_seed,
            train=False, init_checkpoint=str(stage_c_checkpoint),
        )
        agg = common.aggregate_metrics(eval_episodes)
        agg["round"] = round_idx
        agg["total_steps"] = total_steps
        agg["train_reward_mean"] = train_reward_mean if train_reward_mean is not None else float("nan")
        common.write_metrics_csv_row(csv_path, agg)

        common.log_print(
            log_file,
            f"[eval @ round {round_idx}, step {total_steps}] "
            f"completion_rate={agg['completion_rate']:.2f} "
            f"steps_per_coin={agg['steps_per_coin_mean']:.1f} "
            f"train_reward_mean={agg['train_reward_mean']}",
        )

    common.plot_training_curves(
        csv_path, png_path, x_col="total_steps",
        y_cols=["completion_rate", "steps_per_coin_mean", "train_reward_mean"],
        title="Stage C fine-tune",
    )
    common.log_print(log_file, f"Stage C loop finished. Chart: {png_path}")
    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
