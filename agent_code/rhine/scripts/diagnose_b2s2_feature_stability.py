"""Read-only diagnostic: for the B2+S2 rounds with sustained oscillation
(located by diagnose_b2s2_oscillation.py), checks whether the feature vector
is identical across repeated visits to the same tile.

An identical vector means the stateless policy is asked for different actions
from the same input, so a reward-only stall penalty cannot fix it; otherwise
the differing dimensions are reported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.features import features_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.diagnose_stage_a2_failure_modes import _full_trace

B2S2 = common.MODELS_DIR / "task2_stage_a2_B2S2.pt"

FEATURE_NAMES = [
    "can_move_up", "can_move_down", "can_move_left", "can_move_right",
    "has_reachable_coin", "nearest_coin_distance",
    "coin_path_up", "coin_path_down", "coin_path_left", "coin_path_right",
    "bomb_available", "has_bombing_target", "nearest_bombing_distance",
    "bombing_path_up", "bombing_path_down", "bombing_path_left", "bombing_path_right",
    "crates_destructible_at_target",
    "current_tile_in_danger", "nearest_threat_timer",
]
assert len(FEATURE_NAMES) == 20


def main():
    log_file, log_path = common.open_log_file("task2_diagnose_b2s2_feature_stability")
    common.log_print(log_file, f"B2+S2 feature-vector stability diagnosis -- log file: {log_path}")
    common.log_print(
        log_file,
        "Question: within a sustained-oscillation window, when the agent repeatedly returns to the "
        "same tile, is the 20-dim feature vector byte-for-byte identical each time?",
    )

    round_traces = _full_trace(B2S2)

    sustained = []
    for round_idx, trace, actions in round_traces:
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            sustained.append((round_idx, trace, actions, run_len, start_idx))

    common.log_print(log_file, f"sustained-oscillation rounds: {len(sustained)}/{len(round_traces)} -> "
                                f"{[r for r, *_ in sustained]}")

    overall_all_identical = True

    for round_idx, trace, actions, run_len, start_idx in sustained:
        window = list(range(start_idx, start_idx + run_len))
        common.log_print(
            log_file,
            f"\n=== round {round_idx}: oscillation window steps {trace[window[0]]['step']}-{trace[window[-1]]['step']} "
            f"(run_len={run_len}) ===",
        )

        feats_by_pos = {}
        for i in window:
            rec = trace[i]
            fv = features_from_semantic(rec["semantic"])
            feats_by_pos.setdefault(rec["pos"], []).append((rec["step"], rec["chosen"], fv))

        common.log_print(log_file, f"  distinct tiles in window: {sorted(feats_by_pos.keys())}")

        for pos, entries in sorted(feats_by_pos.items()):
            vectors = [fv for _, _, fv in entries]
            steps = [st for st, _, _ in entries]
            actions_here = [a for _, a, _ in entries]
            distinct_vectors = []
            for fv in vectors:
                if not any(np.array_equal(fv, dv) for dv in distinct_vectors):
                    distinct_vectors.append(fv)

            identical = len(distinct_vectors) == 1
            overall_all_identical &= identical
            distinct_actions_here = sorted(set(actions_here))
            common.log_print(
                log_file,
                f"  tile {pos}: visited {len(entries)}x at steps {steps}, actions chosen there={distinct_actions_here} "
                f"-> {'IDENTICAL every visit' if identical else f'{len(distinct_vectors)} DISTINCT feature vectors'}",
            )

            if not identical:
                ref = distinct_vectors[0]
                for dv in distinct_vectors[1:]:
                    diff_idx = np.where(~np.isclose(ref, dv))[0]
                    diffs = [f"{FEATURE_NAMES[j]}(#{j+1}): {ref[j]:.4f}->{dv[j]:.4f}" for j in diff_idx]
                    common.log_print(log_file, f"    differs in {len(diff_idx)} dim(s): {', '.join(diffs)}")

        # Same check restricted to direct back-and-forth recurrences.
        common.log_print(log_file, "  step-by-step detail (pos, chosen, key dims 1-4/11-18/19-20):")
        key_idx = list(range(0, 4)) + list(range(10, 18)) + [18, 19]
        for i in window[:20]:
            rec = trace[i]
            fv = features_from_semantic(rec["semantic"])
            key_vals = " ".join(f"{FEATURE_NAMES[j]}={fv[j]:.3f}" for j in key_idx)
            common.log_print(
                log_file,
                f"    step={rec['step']:>4} pos={rec['pos']} chosen={rec['chosen']:<5} {key_vals}",
            )

    common.log_print(
        log_file,
        f"\n=== SUMMARY: across all {len(sustained)} sustained-oscillation rounds, every same-tile "
        f"revisit within the window produced an IDENTICAL 20-dim feature vector: {overall_all_identical} ===",
    )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
