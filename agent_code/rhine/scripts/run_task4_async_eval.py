"""Task 4 async evaluator: watches one training seed's snapshot directory and
evaluates each new snapshot out-of-process (see run_stage_a2_task3.py's
--disable-periodic-eval). Writes one CSV row per snapshot with
aggregate_metrics()'s fields plus `round`/`snapshot_path`, so
select_task4_checkpoint.py can reconstruct _select_best_checkpoint()'s input.

A train=False evaluation never creates an optimizer or writes the checkpoint,
and loads each snapshot into its own model, so the training loop's state is
unaffected.

Usage (one process per training seed):
    nohup python -m agent_code.rhine.scripts.run_task4_async_eval --seed 0 \
        > /dev/null 2>&1 &
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run

SCENARIO = "classic"
OPPONENTS = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]
STAGE = "task4"  # Filename stem of run_stage_a2_task3.py's --stage TASK4.


def _snapshot_round(path: Path) -> int:
    # task3_stage_task4_seed{N}_round{R}.pt
    return int(path.stem.rsplit("_round", 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models-dir", type=str, default=str(common.MODELS_DIR),
                         help="Directory containing the training run's checkpoint + snapshots/ "
                              "subdirectory (must match that run's --checkpoint-path-override parent).")
    parser.add_argument("--eval-rounds", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=3000)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--out-csv", type=str, default=None,
                         help="Default: agent_code/rhine/logs/task4_async_eval_seed<N>.csv")
    parser.add_argument("--opponents", nargs="+", default=None)
    args = parser.parse_args()
    opponents = args.opponents if args.opponents else OPPONENTS

    cfg.ENABLE_OSCILLATION_BREAKER = True
    # Task 4 keeps training, evaluation and submission config identical.
    cfg.ENABLE_DEADLOCK_BOMB = False

    # Absolute path: it is read inside the framework's chdir into the agent directory.
    models_dir = Path(args.models_dir).resolve()
    snapshot_dir = models_dir / "snapshots"
    csv_path = Path(args.out_csv) if args.out_csv else common.LOGS_DIR / f"task4_async_eval_seed{args.seed}.csv"
    done_marker = snapshot_dir / f"task3_stage_{STAGE}_seed{args.seed}.done"
    pattern_prefix = f"task3_stage_{STAGE}_seed{args.seed}_round"

    log_file, log_path = common.open_log_file(f"task4_async_eval_seed{args.seed}")
    common.log_print(log_file, f"Task 4 async eval, seed={args.seed} -- log: {log_path}")
    common.log_print(
        log_file,
        f"watching {snapshot_dir} for {pattern_prefix}*.pt, done_marker={done_marker}, "
        f"eval_rounds={args.eval_rounds} eval_seed={args.eval_seed} opponents={opponents} "
        f"breaker=True no_bomb_when_cleared={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED} "
        f"deadlock_bomb=False -> {csv_path}",
    )

    processed_rounds = set()
    if csv_path.exists():
        for row in common._read_csv_rows(csv_path):
            processed_rounds.add(int(row["round"]))
        common.log_print(log_file, f"resuming: {len(processed_rounds)} round(s) already in {csv_path}")

    def _pending():
        return sorted(
            (p for p in snapshot_dir.glob(f"{pattern_prefix}*.pt") if _snapshot_round(p) not in processed_rounds),
            key=_snapshot_round,
        )

    while True:
        candidates = _pending()
        for snapshot_path in candidates:
            round_idx = _snapshot_round(snapshot_path)
            t0 = time.time()
            eval_episodes, _, self_kill_trace_paths = common.run_evaluation_with_self_kill_tracing(
                n_rounds=args.eval_rounds, scenario=SCENARIO, seed=args.eval_seed,
                init_checkpoint=str(snapshot_path), disable_think_time_limit=True,
                opponents=opponents,
            )
            agg = common.aggregate_metrics(eval_episodes)
            agg["round"] = round_idx
            agg["snapshot_path"] = str(snapshot_path)
            n_sustained = sum(
                1 for ep in eval_episodes
                if ep.actions and longest_oscillation_run(ep.actions)[0] >= OSCILLATION_THRESHOLD
            )
            agg["oscillation_fraction"] = n_sustained / len(eval_episodes) if eval_episodes else float("nan")
            common.write_metrics_csv_row(csv_path, agg)
            processed_rounds.add(round_idx)
            common.log_print(
                log_file,
                f"[eval round={round_idx}] score_mean={agg['score_mean']:.2f} "
                f"self_kill_rate={agg['self_kill_rate']:.2f} "
                f"got_killed_by_opponent_rate={agg['got_killed_by_opponent_rate']:.2f} "
                f"oscillation_fraction={agg['oscillation_fraction']:.3f} "
                f"completion_rate={agg['completion_rate']:.2f} "
                f"elapsed={time.time() - t0:.0f}s snapshot={snapshot_path.name}",
            )
            if self_kill_trace_paths:
                common.log_print(log_file, f"  self-kill trace(s): {[str(p) for p in self_kill_trace_paths]}")

        if done_marker.exists() and not _pending():
            common.log_print(log_file, "done marker found and no pending snapshots -- exiting.")
            break
        time.sleep(args.poll_interval_seconds)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
