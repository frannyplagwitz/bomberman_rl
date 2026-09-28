"""Read-only diagnostic: root-causes the self-kills of the B2 (fixed)
checkpoint seen in its training-time and final evaluations.

Two parts:
(1) For each captured self-kill trace, reports whether the final
    chosen_action lay outside legal_actions (a mask bypass).
(2) Re-evaluates the same checkpoint repeatedly without concurrent load,
    clearing the 'BombeRLeWorld' logger's handlers before each run (the
    framework truncates game.log per world and accumulates stale handlers),
    and checks self_kill_rate reproducibility and whether think-time
    warnings coincide with self-kills.
"""
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common

N_REPEATS = 5
EVAL_ROUNDS = 30
EVAL_SEED = 1000
SCENARIO = "loot-crate"
CHECKPOINT = common.MODELS_DIR / "task2_stage_a2_B2.pt"
GAME_LOG = Path("/home/rhine/projects/bomberman_rl-master/logs/game.log")
SNAPSHOT_DIR = common.LOGS_DIR / "b2_fixed_selfkill_diagnosis"

EXISTING_TRACES = [
    ("round100 (training burst)", "selfkill_trace_20260907_223811_207895.log"),
    ("round350 (training burst)", "selfkill_trace_20260907_225800_092507.log"),
    ("round800 (training burst)", "selfkill_trace_20260907_233943_935669.log"),
    ("round950 (training burst)", "selfkill_trace_20260907_235113_863250.log"),
    ("round1050 (training burst)", "selfkill_trace_20260907_235917_294450.log"),
]


def part1_check_existing_traces(log_file):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, "PART 1: were the 5 known training-burst self-kills a mask bypass?")
    common.log_print(log_file, "=" * 70)
    for label, fname in EXISTING_TRACES:
        path = common.LOGS_DIR / fname
        lines = path.read_text().strip().splitlines()
        last_line = lines[-1]
        # Parse chosen_action and legal_actions off the last recorded step.
        chosen = last_line.split("chosen_action=")[-1].strip()
        legal_str = last_line.split("legal_actions=")[1].split("]")[0] + "]"
        legal = eval(legal_str)  # Trusted: our own log format.
        bypassed = chosen not in legal
        common.log_print(
            log_file,
            f"  {label} ({fname}): last step -> {last_line.split('step=')[1][:120]}\n"
            f"    chosen_action={chosen!r} in legal_actions={legal}? "
            f"{'NO -- MASK BYPASSED' if bypassed else 'YES -- mask was respected, death was NOT a mask bypass'}",
        )


def part2_reproducibility_and_timeout_correlation(log_file):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, "PART 2: isolated (non-concurrent) reproducibility + timeout correlation")
    common.log_print(log_file, "=" * 70)
    common.log_print(
        log_file,
        f"Re-evaluating {CHECKPOINT.name} {N_REPEATS}x, {EVAL_ROUNDS} rounds each, seed={EVAL_SEED}, "
        f"scenario={SCENARIO}, no other heavy process running.",
    )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    bomberleworld_logger = logging.getLogger("BombeRLeWorld")

    self_kill_counts = []
    for i in range(1, N_REPEATS + 1):
        # Clear stale handlers so this run's game.log reflects only this run.
        for h in list(bomberleworld_logger.handlers):
            bomberleworld_logger.removeHandler(h)

        episodes, _, trace_paths = common.run_evaluation_with_self_kill_tracing(
            n_rounds=EVAL_ROUNDS, scenario=SCENARIO, seed=EVAL_SEED, init_checkpoint=str(CHECKPOINT),
        )
        n_self_kills = sum(1 for ep in episodes if ep.self_kill)
        self_kill_counts.append(n_self_kills)

        snapshot_path = SNAPSHOT_DIR / f"game_log_run{i}.log"
        if GAME_LOG.exists():
            shutil.copy(GAME_LOG, snapshot_path)
        timeout_lines = []
        if snapshot_path.exists():
            timeout_lines = [
                line for line in snapshot_path.read_text().splitlines() if "exceeded think time" in line
            ]

        common.log_print(
            log_file,
            f"  run {i}: self_kill_rate={n_self_kills}/{EVAL_ROUNDS}, "
            f"self-kill trace file(s)={[p.name for p in trace_paths]}, "
            f"'exceeded think time' warnings in this run's isolated game.log snapshot: {len(timeout_lines)}",
        )
        for line in timeout_lines:
            common.log_print(log_file, f"    {line}")

        if trace_paths:
            for p in trace_paths:
                last_line = p.read_text().strip().splitlines()[-1]
                chosen = last_line.split("chosen_action=")[-1].strip()
                legal_str = last_line.split("legal_actions=")[1].split("]")[0] + "]"
                legal = eval(legal_str)
                bypassed = chosen not in legal
                common.log_print(
                    log_file,
                    f"    new self-kill trace {p.name}: last step -> ...{last_line.split('step=')[1][:120]}\n"
                    f"      chosen_action={chosen!r} in legal_actions={legal}? "
                    f"{'NO -- MASK BYPASSED' if bypassed else 'YES -- NOT a mask bypass, needs separate investigation'}",
                )

    common.log_print(
        log_file,
        f"\nSummary across {N_REPEATS} isolated (non-concurrent) runs: self_kill counts = {self_kill_counts} "
        f"(out of {EVAL_ROUNDS} rounds each). Total self-kills across all isolated runs: {sum(self_kill_counts)}.",
    )


def main():
    log_file, log_path = common.open_log_file("task2_diagnose_b2_fixed_selfkill")
    common.log_print(log_file, f"B2(fixed) self-kill diagnosis -- log file: {log_path}")

    part1_check_existing_traces(log_file)
    part2_reproducibility_and_timeout_correlation(log_file)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
