"""Task 2 Stage A: loot-crate training, B1/B2 reward ablation.

B1 (baseline): COIN_REWARD + STEP_COST + CRATE_DESTROYED_REWARD + SELF_KILL_PENALTY,
no shaping (default REWARD_CONFIG, ENABLE_BOMBING_PROGRESS_SHAPING=False).
B2 (+ unified progress shaping): B1 + ENABLE_BOMBING_PROGRESS_SHAPING=True.

Launch in the background; the script writes its own timestamped log under
agent_code/rhine/logs/.

Usage:
    nohup python -m agent_code.rhine.scripts.run_stage_a_task2 --ablation B1 \
        --total-timesteps 300000 > /dev/null 2>&1 &
    nohup python -m agent_code.rhine.scripts.run_stage_a_task2 --ablation B2 \
        --total-timesteps 300000 > /dev/null 2>&1 &
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common

ABLATION_REWARD_OVERRIDES = {
    "B1": {"ENABLE_BOMBING_PROGRESS_SHAPING": False},
    "B2": {"ENABLE_BOMBING_PROGRESS_SHAPING": True},
}


def _is_stabilized(recent):
    """Heuristic plateau check over the last evaluation points (each a dict
    with crates_destroyed_mean, wait_fraction_mean, completion_rate): every
    metric's range must be small, relative to its mean for crates and
    absolute for the 0-1 metrics. Report the underlying numbers alongside it.
    """
    if len(recent) < 4:
        return False, {}
    last4 = recent[-4:]
    crates = [r["crates_destroyed_mean"] for r in last4]
    waits = [r["wait_fraction_mean"] for r in last4]
    completions = [r["completion_rate"] for r in last4]

    crates_range = max(crates) - min(crates)
    crates_mean = sum(crates) / 4
    stable_crates = crates_range <= max(5.0, 0.25 * crates_mean)

    wait_range = max(waits) - min(waits)
    stable_wait = wait_range <= 0.15

    completion_range = max(completions) - min(completions)
    stable_completion = completion_range <= 0.15

    detail = {
        "crates_range": crates_range, "crates_mean": crates_mean, "stable_crates": stable_crates,
        "wait_range": wait_range, "stable_wait": stable_wait,
        "completion_range": completion_range, "stable_completion": stable_completion,
    }
    return (stable_crates and stable_wait and stable_completion), detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["B1", "B2"], default="B1")
    parser.add_argument("--total-timesteps", type=int, default=300_000,
                         help="Training budget in environment steps, matching Task 1's Stage A.")
    parser.add_argument("--eval-interval-rounds", type=int, default=50,
                         help="Run a periodic deterministic evaluation burst every N training rounds.")
    parser.add_argument("--eval-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Checkpoint path to resume training from (instead of a fresh "
                              "random init) -- used to restart a run in-place after a driver "
                              "script change without losing prior training progress.")
    parser.add_argument("--resume-steps", type=int, default=0,
                         help="Environment steps already completed before this process started "
                              "(only meaningful together with --resume-from); counted towards "
                              "--total-timesteps so the run stops at the same overall budget.")
    parser.add_argument("--resume-round", type=int, default=0,
                         help="Training round already completed before this process started "
                              "(only meaningful together with --resume-from); numbered "
                              "checkpoint snapshots and CSV rows continue from here.")
    parser.add_argument("--stabilize", action="store_true",
                         help="Instead of stopping at --total-timesteps, keep training until "
                              "crates_destroyed_mean, wait_fraction_mean and completion_rate all "
                              "stop changing much over 4 consecutive eval points (see "
                              "_is_stabilized()), or --total-timesteps is hit as a hard safety cap.")
    args = parser.parse_args()
    if args.resume_from:
        # Must be absolute: the framework chdir()s into agent_code/rhine/
        # before invoking callbacks.
        args.resume_from = str(Path(args.resume_from).resolve())

    log_file, log_path = common.open_log_file(f"task2_stage_a_{args.ablation}")
    csv_path = log_path.with_suffix(".csv")
    png_path = log_path.with_suffix(".png")
    common.log_print(log_file, f"Task 2 Stage A ({args.ablation}) -- log file: {log_path}")
    common.log_print(log_file, f"metrics CSV: {csv_path}")
    common.log_print(log_file, f"args: {vars(args)}")

    # Per-ablation filename so concurrent B1/B2 runs don't overwrite each other.
    checkpoint_path = common.MODELS_DIR / f"task2_stage_a_{args.ablation}.pt"
    snapshot_dir = common.MODELS_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    reward_override = ABLATION_REWARD_OVERRIDES[args.ablation]

    total_steps = args.resume_steps
    round_idx = args.resume_round
    burst_seed = args.seed + (args.resume_round // args.eval_interval_rounds if args.resume_round else 0)
    init_ckpt = args.resume_from  # None: fresh random init.

    if args.resume_from:
        common.log_print(
            log_file,
            f"Resuming from {args.resume_from} at round={round_idx}, total_steps={total_steps}",
        )
    if args.stabilize:
        common.log_print(
            log_file,
            f"Stabilize mode: will keep training past --total-timesteps ({args.total_timesteps}) "
            f"until 4 consecutive eval points show small changes in crates_destroyed_mean/"
            f"wait_fraction_mean/completion_rate (see _is_stabilized()), used as a hard cap only.",
        )

    recent_metrics = []
    stabilized = False

    # Continue while under the cap and, in stabilize mode, not yet stabilized.
    while total_steps < args.total_timesteps and (not args.stabilize or not stabilized):
        train_episodes, train_world = common.run_episodes(
            n_rounds=args.eval_interval_rounds, scenario="loot-crate", seed=burst_seed,
            train=True, init_checkpoint=init_ckpt, save_checkpoint=str(checkpoint_path),
            reward_override=reward_override,
        )
        train_reward_mean = common.training_reward_mean(train_world)
        policy_loss_mean, value_loss_mean = common.training_loss_means(train_world)
        total_steps += sum(ep.steps for ep in train_episodes)
        round_idx += args.eval_interval_rounds
        burst_seed += 1  # Vary the map seed across bursts.
        init_ckpt = str(checkpoint_path)  # Resume from our own progress from here on.

        # Non-overwriting per-round snapshot, so any round's checkpoint stays
        # inspectable after the rolling checkpoint moves on.
        snapshot_path = snapshot_dir / f"task2_stage_a_{args.ablation}_round{round_idx}.pt"
        shutil.copy(checkpoint_path, snapshot_path)

        eval_episodes, _, self_kill_trace_paths = common.run_evaluation_with_self_kill_tracing(
            n_rounds=args.eval_rounds, scenario="loot-crate", seed=args.eval_seed,
            init_checkpoint=str(checkpoint_path),
        )
        agg = common.aggregate_metrics(eval_episodes)
        agg["round"] = round_idx
        agg["total_steps"] = total_steps
        agg["train_reward_mean"] = train_reward_mean if train_reward_mean is not None else float("nan")
        agg["policy_loss_mean"] = policy_loss_mean if policy_loss_mean is not None else float("nan")
        agg["value_loss_mean"] = value_loss_mean if value_loss_mean is not None else float("nan")
        # Charts are generated from this CSV, not from in-memory `agg`.
        common.write_metrics_csv_row(csv_path, agg)

        common.log_print(
            log_file,
            f"[eval @ round {round_idx}, step {total_steps}] "
            f"completion_rate={agg['completion_rate']:.2f} "
            f"self_kill_rate={agg['self_kill_rate']:.2f} "
            f"crates_destroyed_mean={agg['crates_destroyed_mean']:.2f} "
            f"bomb_efficiency={agg['bomb_efficiency']:.2f} "
            f"wait_fraction={agg['wait_fraction_mean']:.3f} "
            f"train_reward_mean={agg['train_reward_mean']} "
            f"policy_loss_mean={agg['policy_loss_mean']} "
            f"value_loss_mean={agg['value_loss_mean']} "
            f"snapshot={snapshot_path.name}",
        )
        if self_kill_trace_paths:
            common.log_print(
                log_file,
                f"  !! self-kill occurred ({len(self_kill_trace_paths)} round(s)) -- "
                f"trace(s): {[str(p) for p in self_kill_trace_paths]}, "
                f"checkpoint snapshot for replay: {snapshot_path}",
            )

        if args.stabilize:
            recent_metrics.append({
                "round": round_idx,
                "crates_destroyed_mean": agg["crates_destroyed_mean"],
                "wait_fraction_mean": agg["wait_fraction_mean"],
                "completion_rate": agg["completion_rate"],
            })
            stabilized, detail = _is_stabilized(recent_metrics)
            if stabilized:
                common.log_print(
                    log_file,
                    f"STABILIZED at round {round_idx} (step {total_steps}): "
                    f"last 4 points -- crates_range={detail['crates_range']:.2f} "
                    f"(mean={detail['crates_mean']:.2f}), wait_range={detail['wait_range']:.3f}, "
                    f"completion_range={detail['completion_range']:.3f}. Stopping.",
                )

    common.plot_training_curves(
        csv_path, png_path, x_col="total_steps",
        y_cols=[
            "completion_rate", "self_kill_rate", "crates_destroyed_mean", "bomb_efficiency",
            "train_reward_mean", "policy_loss_mean", "value_loss_mean",
        ],
        title=f"Task 2 Stage A ({args.ablation})",
    )
    common.log_print(log_file, f"Task 2 Stage A ({args.ablation}) loop finished. Chart: {png_path}")
    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
