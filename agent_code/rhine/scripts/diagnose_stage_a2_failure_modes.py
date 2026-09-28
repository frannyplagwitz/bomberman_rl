"""Stage A2 read-only diagnostic, two parts:

(1) B2+S: for each sustained-oscillation round, replays the action/position
    sequence through train._update_stall_counter() alongside
    nearest_bombing_distance/bomb_available/mask, to see whether the stall
    penalty ever engages during the oscillation.

(2) B2+C / B2+SC: checks whether the drop in crates destroyed together with a
    high wait fraction is a genuine WAIT collapse (long, static WAIT streaks
    while BOMB/movement stay legal).
"""
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.state_processing import extract_semantic_state
from agent_code.rhine.train import _update_stall_counter

SEED = 1000
SCENARIO = "loot-crate"
N_ROUNDS = 30


def _full_trace(checkpoint_path, n_rounds=N_ROUNDS, seed=SEED):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=seed, train=False)
    agent = world.agents[0]

    round_traces = []
    for round_idx in range(1, n_rounds + 1):
        world.new_round()
        world.user_input = "WAIT"
        trace = []
        while world.running:
            state = world.get_state_for_agent(agent)
            record = None
            if state is not None:
                semantic = extract_semantic_state(state)
                mask = mask_from_semantic(semantic)
                legal = [cfg.ACTIONS[i] for i, m in enumerate(mask) if m]
                record = {
                    "step": state["step"], "pos": semantic.self_pos, "legal": legal,
                    "bomb_available": semantic.bomb_available,
                    "has_bombing_target": semantic.has_bombing_target,
                    "nearest_bombing_distance": semantic.nearest_bombing_distance,
                    "semantic": semantic, "mask": mask,
                }
            world.do_step("WAIT")
            if record is not None:
                trace.append(record)

        actions = list(world.replay["actions"][agent.name])
        for rec, a in zip(trace, actions):
            rec["chosen"] = a
        round_traces.append((round_idx, trace, actions))
    world.end()
    return round_traces


def investigate_b2s_stall_interaction(checkpoint_path, log_file):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, "(1) B2+S oscillation vs stall_counter interaction")
    common.log_print(log_file, "=" * 70)

    round_traces = _full_trace(checkpoint_path)
    sustained = []
    for round_idx, trace, actions in round_traces:
        run_len, start_idx = longest_oscillation_run(actions)
        if run_len >= OSCILLATION_THRESHOLD:
            sustained.append((round_idx, trace, actions, run_len, start_idx))

    common.log_print(log_file, f"sustained-oscillation rounds: {len(sustained)}/{len(round_traces)}")

    n_examples = min(5, len(sustained))
    stall_ever_above_zero_count = 0
    stall_ever_above_threshold_count = 0
    for round_idx, trace, actions, run_len, start_idx in sustained:
        # Replay over the whole round, since the counter's state carries across steps.
        fake_self = SimpleNamespace(stall_counter=0, recent_positions=deque(maxlen=3))
        stall_trace = []
        for rec in trace:
            sc = _update_stall_counter(fake_self, rec["semantic"], rec["mask"])
            stall_trace.append(sc)

        window_stall = stall_trace[start_idx: start_idx + run_len]
        max_stall_in_window = max(window_stall) if window_stall else 0
        if max_stall_in_window > 0:
            stall_ever_above_zero_count += 1
        if max_stall_in_window > cfg.REWARD_CONFIG.STALL_THRESHOLD:
            stall_ever_above_threshold_count += 1

        common.log_print(
            log_file,
            f"  round={round_idx:>2} run_len={run_len:>3} start_step={trace[start_idx]['step']:>4} "
            f"max_stall_counter_during_run={max_stall_in_window:>3} "
            f"(STALL_THRESHOLD={cfg.REWARD_CONFIG.STALL_THRESHOLD})",
        )

    common.log_print(
        log_file,
        f"\nOf {len(sustained)} sustained-oscillation rounds: stall_counter was >0 at some point "
        f"during the run in {stall_ever_above_zero_count}, and exceeded STALL_THRESHOLD "
        f"(i.e. penalty WAS actively applying) in {stall_ever_above_threshold_count}.",
    )

    common.log_print(log_file, f"\n--- {n_examples} example traces (pos/legal/chosen/bombing_distance/stall_counter) ---")
    for round_idx, trace, actions, run_len, start_idx in sustained[:n_examples]:
        fake_self = SimpleNamespace(stall_counter=0, recent_positions=deque(maxlen=3))
        stall_trace = []
        for rec in trace:
            sc = _update_stall_counter(fake_self, rec["semantic"], rec["mask"])
            stall_trace.append(sc)

        lo = max(0, start_idx - 2)
        hi = min(len(trace), start_idx + run_len + 5)
        common.log_print(log_file, f"\n-- round {round_idx} (oscillation run_len={run_len}, starts step {trace[start_idx]['step']}) --")
        for i in range(lo, hi):
            rec = trace[i]
            common.log_print(
                log_file,
                f"  step={rec['step']:>4} pos={rec['pos']} bomb_avail={rec['bomb_available']} "
                f"nearest_bombing_distance={rec['nearest_bombing_distance']} legal={rec['legal']} "
                f"chosen={rec['chosen']} stall_counter={stall_trace[i]}",
            )


def investigate_wait_collapse(label, checkpoint_path, log_file):
    common.log_print(log_file, "\n" + "=" * 70)
    common.log_print(log_file, f"(2) {label} WAIT-streak / activity investigation")
    common.log_print(log_file, "=" * 70)

    round_traces = _full_trace(checkpoint_path)

    longest_wait_streaks = []
    wait_streak_bomb_legal_fracs = []
    wait_streak_static_fracs = []  # Share of the longest WAIT streak spent on one tile.

    for round_idx, trace, actions in round_traces:
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

        longest_wait_streaks.append(best_len)
        if best_len > 0:
            window = trace[best_start: best_start + best_len]
            bomb_legal_frac = np.mean([("BOMB" in r["legal"]) for r in window])
            positions = {r["pos"] for r in window}
            static_frac = 1.0 if len(positions) == 1 else 0.0
            wait_streak_bomb_legal_fracs.append(bomb_legal_frac)
            wait_streak_static_fracs.append(static_frac)

    common.log_print(
        log_file,
        f"longest WAIT streak per round: mean={np.mean(longest_wait_streaks):.1f} "
        f"median={np.median(longest_wait_streaks):.1f} max={max(longest_wait_streaks)} "
        f"(out of {round_traces[0][1][-1]['step'] + 1 if round_traces else '?'} steps/round)",
    )
    if wait_streak_bomb_legal_fracs:
        common.log_print(
            log_file,
            f"during each round's longest WAIT streak: BOMB legal for a mean of "
            f"{np.mean(wait_streak_bomb_legal_fracs) * 100:.1f}% of those steps; "
            f"streak stayed at a single fixed tile in {sum(wait_streak_static_fracs)}/{len(wait_streak_static_fracs)} rounds",
        )

    common.log_print(log_file, "\n--- 3 example round traces around their longest WAIT streak ---")
    shown = 0
    for round_idx, trace, actions in round_traces:
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
        if best_len < 20 or shown >= 3:
            continue
        shown += 1
        lo, hi = best_start, min(len(trace), best_start + 12)
        common.log_print(log_file, f"\n-- round {round_idx}, longest WAIT streak={best_len} steps starting at step {trace[best_start]['step']} --")
        for i in range(lo, hi):
            rec = trace[i]
            common.log_print(
                log_file,
                f"  step={rec['step']:>4} pos={rec['pos']} bomb_avail={rec['bomb_available']} "
                f"has_bombing_target={rec['has_bombing_target']} nearest_bombing_distance={rec['nearest_bombing_distance']} "
                f"legal={rec['legal']} chosen={rec['chosen']}",
            )


def main():
    log_file, log_path = common.open_log_file("task2_stage_a2_failure_mode_diagnosis")
    common.log_print(log_file, f"Stage A2 failure-mode diagnosis -- log file: {log_path}")

    investigate_b2s_stall_interaction(common.MODELS_DIR / "task2_stage_a2_B2S.pt", log_file)
    investigate_wait_collapse("B2+C", common.MODELS_DIR / "task2_stage_a2_B2C.pt", log_file)
    investigate_wait_collapse("B2+SC", common.MODELS_DIR / "task2_stage_a2_B2SC.pt", log_file)

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
