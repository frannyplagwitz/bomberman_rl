"""Task 2 diagnostic: steps to completion under a raised safety cap instead of
the official 400-step limit. Runs every saved B1/B2 snapshot for 30 rounds and
writes one table per ablation with:

  - completion_rate_2000cap: share of rounds reaching completion
  - steps_to_completion_mean/median/std_2000cap: over completed rounds only

Completion is environment.py's native end condition (all crates destroyed,
all coins collected, no bombs/explosions left). Without opponents or death, a
round ending before the cap must have completed, so its length is the
completion step.

Measures capability only; efficiency metrics would be diluted by
post-completion idling and are not recomputed. Only rounds with a saved
snapshot can be measured.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import settings as s

s.MAX_STEPS = 2000  # Safety cap for this process only.

from agent_code.rhine.scripts import common

EVAL_ROUNDS = 30
EVAL_SEED = 1000
SCENARIO = "loot-crate"

FIELDS = [
    "round", "total_steps", "n_rounds_evaluated",
    "completion_rate_2000cap",
    "steps_to_completion_mean_2000cap", "steps_to_completion_median_2000cap",
    "steps_to_completion_std_2000cap",
]

# round -> total_steps lookup from the per-run training CSVs.
SOURCE_CSVS = {
    "B1": [
        "task2_stage_a_B1_run_20260905_220006.csv",
        "task2_stage_a_B1_run_20260905_232002.csv",
        "task2_stage_a_B1_run_20260906_172940.csv",
    ],
    "B2": [
        "task2_stage_a_B2_run_20260905_220029.csv",
        "task2_stage_a_B2_run_20260905_232002.csv",
        "task2_stage_a_B2_run_20260906_172940.csv",
    ],
}


def load_total_steps_by_round(ablation):
    lookup = {}
    for csv_name in SOURCE_CSVS[ablation]:
        path = common.LOGS_DIR / csv_name
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                lookup[int(row["round"])] = int(row["total_steps"])
    return lookup


def measure_one_snapshot(snapshot_path):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(snapshot_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=EVAL_SEED, train=False)

    completion_steps = []
    n = EVAL_ROUNDS
    for _ in range(EVAL_ROUNDS):
        ep = common.run_one_round(world)
        # Without opponents or death, ending before the cap means completion.
        if ep.steps < s.MAX_STEPS:
            completion_steps.append(ep.steps)
    world.end()

    return {
        "n_rounds_evaluated": n,
        "completion_rate_2000cap": len(completion_steps) / n,
        "steps_to_completion_mean_2000cap": float(np.mean(completion_steps)) if completion_steps else float("nan"),
        "steps_to_completion_median_2000cap": float(np.median(completion_steps)) if completion_steps else float("nan"),
        "steps_to_completion_std_2000cap": float(np.std(completion_steps)) if completion_steps else float("nan"),
    }


def main():
    log_file, log_path = common.open_log_file("task2_completion_steps")
    common.log_print(
        log_file,
        f"Completion-steps supplementary table -- EVALUATION CONDITION: {s.MAX_STEPS}-step "
        f"SAFETY CAP (not the official 400-step limit). COMPLETION CONDITION: all crates "
        f"destroyed AND all coins collected (the game's own native end condition). "
        f"-- log file: {log_path}",
    )

    snapshot_dir = common.MODELS_DIR / "snapshots"

    for ablation in ["B1", "B2"]:
        total_steps_by_round = load_total_steps_by_round(ablation)
        snapshots = sorted(
            snapshot_dir.glob(f"task2_stage_a_{ablation}_round*.pt"),
            key=lambda p: int(p.stem.rsplit("round", 1)[-1]),
        )
        out_path = common.LOGS_DIR / f"task2_stage_a_{ablation}_completion_steps_2000cap.csv"

        common.log_print(log_file, f"\n=== {ablation}: {len(snapshots)} snapshot(s) ===")
        with open(out_path, "w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=FIELDS)
            writer.writeheader()
            for snap in snapshots:
                round_num = int(snap.stem.rsplit("round", 1)[-1])
                fields = measure_one_snapshot(snap)
                fields["round"] = round_num
                fields["total_steps"] = total_steps_by_round.get(round_num, "")
                writer.writerow(fields)
                out_f.flush()
                common.log_print(
                    log_file,
                    f"  round={round_num:>4} total_steps={fields['total_steps']} "
                    f"completion_rate_2000cap={fields['completion_rate_2000cap']:.2f} "
                    f"steps_to_completion_mean_2000cap={fields['steps_to_completion_mean_2000cap']:.1f} "
                    f"median={fields['steps_to_completion_median_2000cap']:.1f} "
                    f"std={fields['steps_to_completion_std_2000cap']:.1f}",
                )
        common.log_print(log_file, f"  written: {out_path}")

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
