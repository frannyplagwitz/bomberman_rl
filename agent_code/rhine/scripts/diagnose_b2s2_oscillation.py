"""Read-only diagnostic: for B2+S2's evaluation rounds with sustained
oscillation, replays train._update_stall_v2 step by step (fresh history per
round) to show why the V2 stall penalty never fired.

Tests the hypothesis that brief step-outs to a third tile keep resetting the
confinement window before it fills.
"""
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.diagnose_stage_a2_failure_modes import _full_trace
from agent_code.rhine.train import _update_stall_v2

B2S2 = common.MODELS_DIR / "task2_stage_a2_B2S2.pt"


def main():
    log_file, log_path = common.open_log_file("task2_diagnose_b2s2_oscillation")
    common.log_print(log_file, f"B2+S2 oscillation / stall-v2 interaction diagnosis -- log file: {log_path}")

    round_traces = _full_trace(B2S2)

    sustained = []
    for round_idx, trace, actions in round_traces:
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            sustained.append((round_idx, trace, actions, run_len, start_idx))

    common.log_print(log_file, f"sustained-oscillation rounds: {len(sustained)}/{len(round_traces)}")

    for round_idx, trace, actions, run_len, start_idx in sustained:
        fake_self = SimpleNamespace(recent_positions_v2=deque(maxlen=4))
        stall_v2_trace = [_update_stall_v2(fake_self, rec["semantic"]) for rec in trace]

        window = list(range(start_idx, start_idx + run_len))
        window_triggers = [stall_v2_trace[i] for i in window]
        n_triggered = sum(window_triggers)

        # Distinct-tile count per step within the oscillation run.
        distinct_tiles_in_window_history = []
        hist = deque(maxlen=4)
        bomb_avail_trace = []
        for rec in trace:
            bomb_avail_trace.append(rec["bomb_available"])
            if rec["bomb_available"]:
                hist.append(rec["pos"])
            distinct_tiles_in_window_history.append(len(set(hist)) if len(hist) == 4 else None)

        common.log_print(
            log_file,
            f"\n=== round {round_idx}: oscillation run_len={run_len}, starts step {trace[start_idx]['step']} ===\n"
            f"  stall_v2 triggered on {n_triggered}/{run_len} steps of the oscillation window",
        )

        # Distinct positions in the window and how often a position outside
        # that set briefly appears (a step-out).
        window_positions = [trace[i]["pos"] for i in window]
        core_tiles = set(window_positions)
        common.log_print(
            log_file,
            f"  distinct tiles visited during the whole oscillation window: {sorted(core_tiles)} "
            f"({len(core_tiles)} tiles)",
        )
        bomb_unavail_in_window = sum(1 for i in window if not bomb_avail_trace[i])
        common.log_print(log_file, f"  bomb_available=False steps within the window: {bomb_unavail_in_window}/{run_len}")

        common.log_print(log_file, "  step-by-step (first 40 steps of the window):")
        for i in window[:40]:
            rec = trace[i]
            dt = distinct_tiles_in_window_history[i]
            common.log_print(
                log_file,
                f"    step={rec['step']:>4} pos={rec['pos']} bomb_avail={rec['bomb_available']} "
                f"chosen={rec['chosen']} window_distinct={dt} stall_v2={stall_v2_trace[i]}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
