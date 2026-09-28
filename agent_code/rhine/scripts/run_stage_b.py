"""Stage B: zero-shot evaluation of the Stage A checkpoint on coin-heaven-9
(deterministic policy, official 400-step limit). No training.

Usage:
    nohup python -m agent_code.rhine.scripts.run_stage_b > /dev/null 2>&1 &
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file("stage_b")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Stage B zero-shot eval -- log file: {log_path}")
    common.log_print(log_file, f"per-episode CSV: {csv_path}")

    checkpoint_path = common.MODELS_DIR / "task1_stage_a.pt"
    if not checkpoint_path.is_file():
        common.log_print(log_file, f"FAIL: {checkpoint_path} not found -- run Stage A first.")
        log_file.close()
        raise FileNotFoundError(str(checkpoint_path))

    episodes, _ = common.run_episodes(
        n_rounds=args.n_rounds, scenario="coin-heaven-9", seed=args.seed,
        train=False, init_checkpoint=str(checkpoint_path),
    )

    # Single zero-shot batch, so the CSV holds per-episode rows, not a curve.
    common.write_episode_csv(csv_path, episodes)

    agg = common.aggregate_metrics(episodes)
    for k, v in agg.items():
        common.log_print(log_file, f"{k}: {v}")

    common.plot_episode_distribution(csv_path, png_path, title="Stage B zero-shot (coin-heaven-9)")
    common.log_print(log_file, f"Chart: {png_path}")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
