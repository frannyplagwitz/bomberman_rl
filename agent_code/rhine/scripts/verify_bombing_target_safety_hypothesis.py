"""Read-only diagnostic: tests whether bombing_target_info() choosing targets
by crates_in_blast alone (without an escape check) steers the agent onto
tiles with nearest_bombing_distance == 0 where BOMB is masked out, causing
the oscillation/freeze failure modes.

Regenerates full per-step traces deterministically with the helpers of
diagnose_stage_a2_failure_modes.py and diagnose_oscillation.py. For the
oscillation group (canonical B2) and the freeze group (B2+C), restricted to
each round's failure window (the oscillation run; the longest WAIT streak),
reports at step and round level how often distance 0 coincides with an
illegal BOMB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.diagnose_stage_a2_failure_modes import _full_trace

CANONICAL_B2 = common.MODELS_DIR / "task2_stage_a.pt"
B2C = common.MODELS_DIR / "task2_stage_a2_B2C.pt"


def _longest_wait_streak(actions):
    best_len, best_start = 0, 0
    cur_len, cur_start = 0, 0
    for i, a in enumerate(actions):
        if a == "WAIT":
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_len = 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    return best_len, best_start


def analyze_oscillation(checkpoint_path, log_file, label):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, f"OSCILLATION analysis -- {label}")
    common.log_print(log_file, "=" * 70)

    round_traces = _full_trace(checkpoint_path)

    total_window_steps = 0
    total_dist0_steps = 0
    total_dist0_bomb_illegal_steps = 0
    unsafe_steps = []    # Distance 0, BOMB illegal, bomb available: masked by the escape check.
    cooldown_steps = []  # Distance 0, BOMB illegal, no bomb available: unrelated to escape safety.

    round_level_rows = []  # (round_idx, window_len, dist0_steps, dist0_bomb_illegal_steps)
    examples = []

    for round_idx, trace, actions in round_traces:
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue
        window = trace[start_idx: start_idx + run_len]
        dist0 = [r for r in window if r["nearest_bombing_distance"] == 0]
        dist0_bomb_illegal = [r for r in dist0 if "BOMB" not in r["legal"]]

        total_window_steps += len(window)
        total_dist0_steps += len(dist0)
        total_dist0_bomb_illegal_steps += len(dist0_bomb_illegal)
        round_level_rows.append((round_idx, len(window), len(dist0), len(dist0_bomb_illegal)))

        for r in dist0_bomb_illegal:
            (unsafe_steps if r["bomb_available"] else cooldown_steps).append(r)

        if dist0_bomb_illegal and len(examples) < 4:
            examples.append((round_idx, window, "oscillation"))

    common.log_print(log_file, f"qualifying (sustained-oscillation) rounds: {len(round_level_rows)}/{len(round_traces)}")
    common.log_print(log_file, f"\n--- per-step granularity (pooled across all {len(round_level_rows)} rounds' oscillation windows) ---")
    common.log_print(log_file, f"total steps in oscillation windows: {total_window_steps}")
    common.log_print(
        log_file,
        f"of those, steps at nearest_bombing_distance==0: {total_dist0_steps} "
        f"({100 * total_dist0_steps / total_window_steps:.1f}% of window steps)" if total_window_steps else "n/a",
    )
    if total_dist0_steps:
        common.log_print(
            log_file,
            f"of the distance==0 steps, BOMB NOT in legal_actions: {total_dist0_bomb_illegal_steps} "
            f"({100 * total_dist0_bomb_illegal_steps / total_dist0_steps:.1f}% of distance==0 steps, "
            f"{100 * total_dist0_bomb_illegal_steps / total_window_steps:.1f}% of ALL window steps)",
        )
        common.log_print(
            log_file,
            f"  breakdown of those {total_dist0_bomb_illegal_steps} steps by REASON: "
            f"bomb_available=True but masked out (escape-route-safety reason, directly matches the hypothesis) = {len(unsafe_steps)}; "
            f"bomb_available=False (on cooldown / already used, unrelated to escape-route safety) = {len(cooldown_steps)}",
        )
    else:
        common.log_print(log_file, "of the distance==0 steps: n/a (zero distance==0 steps observed)")

    common.log_print(log_file, "\n--- per-round granularity ---")
    rounds_with_any_dist0 = sum(1 for r in round_level_rows if r[2] > 0)
    rounds_with_any_dist0_bomb_illegal = sum(1 for r in round_level_rows if r[3] > 0)
    common.log_print(
        log_file,
        f"rounds whose oscillation window contains >=1 distance==0 step: "
        f"{rounds_with_any_dist0}/{len(round_level_rows)}",
    )
    common.log_print(
        log_file,
        f"rounds whose oscillation window contains >=1 (distance==0 AND BOMB illegal) step: "
        f"{rounds_with_any_dist0_bomb_illegal}/{len(round_level_rows)}",
    )
    per_round_fracs = [(r[3] / r[2]) if r[2] else 0.0 for r in round_level_rows]
    common.log_print(
        log_file,
        f"per-round (distance==0-AND-BOMB-illegal / distance==0) fraction, averaged only over rounds "
        f"with >=1 distance==0 step: {np.mean([f for f, r in zip(per_round_fracs, round_level_rows) if r[2] > 0]):.3f}"
        if rounds_with_any_dist0 else "n/a (no round had any distance==0 step)",
    )
    common.log_print(log_file, "\nper-round detail: round | window_len | dist0_steps | dist0_bomb_illegal_steps")
    for round_idx, wlen, d0, d0bi in round_level_rows:
        common.log_print(log_file, f"  round={round_idx:>2} window_len={wlen:>3} dist0_steps={d0:>3} dist0_bomb_illegal_steps={d0bi:>3}")

    return examples


def analyze_freeze(checkpoint_path, log_file, label):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, f"FREEZE (WAIT-streak) analysis -- {label}")
    common.log_print(log_file, "=" * 70)

    round_traces = _full_trace(checkpoint_path)

    total_window_steps = 0
    total_dist0_steps = 0
    total_dist0_bomb_illegal_steps = 0
    unsafe_steps = []
    cooldown_steps = []
    round_level_rows = []
    examples = []

    FREEZE_THRESHOLD = 20  # Minimum WAIT streak length in steps.

    for round_idx, trace, actions in round_traces:
        wlen, wstart = _longest_wait_streak(actions)
        if wlen < FREEZE_THRESHOLD:
            continue
        window = trace[wstart: wstart + wlen]
        dist0 = [r for r in window if r["nearest_bombing_distance"] == 0]
        dist0_bomb_illegal = [r for r in dist0 if "BOMB" not in r["legal"]]

        total_window_steps += len(window)
        total_dist0_steps += len(dist0)
        total_dist0_bomb_illegal_steps += len(dist0_bomb_illegal)
        round_level_rows.append((round_idx, len(window), len(dist0), len(dist0_bomb_illegal)))

        for r in dist0_bomb_illegal:
            (unsafe_steps if r["bomb_available"] else cooldown_steps).append(r)

        if dist0_bomb_illegal and len(examples) < 4:
            examples.append((round_idx, window, "freeze"))

    common.log_print(log_file, f"qualifying (WAIT-streak >= {FREEZE_THRESHOLD}) rounds: {len(round_level_rows)}/{len(round_traces)}")
    common.log_print(log_file, f"\n--- per-step granularity (pooled across all {len(round_level_rows)} rounds' WAIT-streak windows) ---")
    common.log_print(log_file, f"total steps in WAIT-streak windows: {total_window_steps}")
    common.log_print(
        log_file,
        f"of those, steps at nearest_bombing_distance==0: {total_dist0_steps} "
        f"({100 * total_dist0_steps / total_window_steps:.1f}% of window steps)" if total_window_steps else "n/a",
    )
    if total_dist0_steps:
        common.log_print(
            log_file,
            f"of the distance==0 steps, BOMB NOT in legal_actions: {total_dist0_bomb_illegal_steps} "
            f"({100 * total_dist0_bomb_illegal_steps / total_dist0_steps:.1f}% of distance==0 steps, "
            f"{100 * total_dist0_bomb_illegal_steps / total_window_steps:.1f}% of ALL window steps)",
        )
        common.log_print(
            log_file,
            f"  breakdown of those {total_dist0_bomb_illegal_steps} steps by REASON: "
            f"bomb_available=True but masked out (escape-route-safety reason, directly matches the hypothesis) = {len(unsafe_steps)}; "
            f"bomb_available=False (on cooldown / already used, unrelated to escape-route safety) = {len(cooldown_steps)}",
        )
    else:
        common.log_print(log_file, "of the distance==0 steps: n/a (zero distance==0 steps observed)")

    common.log_print(log_file, "\n--- per-round granularity ---")
    rounds_with_any_dist0 = sum(1 for r in round_level_rows if r[2] > 0)
    rounds_with_any_dist0_bomb_illegal = sum(1 for r in round_level_rows if r[3] > 0)
    common.log_print(
        log_file,
        f"rounds whose WAIT-streak window contains >=1 distance==0 step: "
        f"{rounds_with_any_dist0}/{len(round_level_rows)}",
    )
    common.log_print(
        log_file,
        f"rounds whose WAIT-streak window contains >=1 (distance==0 AND BOMB illegal) step: "
        f"{rounds_with_any_dist0_bomb_illegal}/{len(round_level_rows)}",
    )
    common.log_print(log_file, "\nper-round detail: round | window_len | dist0_steps | dist0_bomb_illegal_steps")
    for round_idx, wlen, d0, d0bi in round_level_rows:
        common.log_print(log_file, f"  round={round_idx:>2} window_len={wlen:>3} dist0_steps={d0:>3} dist0_bomb_illegal_steps={d0bi:>3}")

    return examples


def main():
    log_file, log_path = common.open_log_file("task2_verify_bombing_target_safety_hypothesis")
    common.log_print(log_file, f"Bombing-target-safety hypothesis verification -- log file: {log_path}")

    osc_examples = analyze_oscillation(CANONICAL_B2, log_file, "canonical B2 (task2_stage_a.pt, round1500)")
    freeze_examples = analyze_freeze(B2C, log_file, "B2+C (task2_stage_a2_B2C.pt, round1000)")

    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, "Example trace snippets (distance==0 & BOMB illegal occurrences, with what happened next)")
    common.log_print(log_file, "=" * 70)
    for round_idx, window, kind in osc_examples + freeze_examples:
        common.log_print(log_file, f"\n-- {kind} example, round {round_idx} --")
        for rec in window[:15]:
            common.log_print(
                log_file,
                f"  step={rec['step']:>4} pos={rec['pos']} nearest_bombing_distance={rec['nearest_bombing_distance']} "
                f"bomb_avail={rec['bomb_available']} legal={rec['legal']} chosen={rec['chosen']}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
