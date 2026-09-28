"""Task 2 diagnostic: sustained back-and-forth oscillation between two
adjacent tiles, as seen on the canonical B2 checkpoint on loot-crate.

immediate_reverse_count only gives a per-round total. This finds each
round's longest run strictly alternating between two opposite actions,
records its step range and the crates/coins left at that point, and captures
full mask/chosen-action traces for a few rounds to tell apart:

  (a) mask-constrained: only the two opposite directions (plus maybe WAIT)
      are legal, e.g. a dead-end corridor;
  (b) learned behavior: other legal actions exist throughout, but the policy
      keeps alternating.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine import config as cfg
from agent_code.rhine.state_processing import extract_semantic_state
from agent_code.rhine.action_mask import mask_from_semantic

OPPOSITE = common.OPPOSITE_DIRECTION
N_ROUNDS = 30
SEED = 1000
SCENARIO = "loot-crate"
CHECKPOINT = common.MODELS_DIR / "task2_stage_a.pt"
OSCILLATION_THRESHOLD = 10  # Consecutive reversal pairs counted as "sustained".


def longest_oscillation_run(actions):
    """Returns (best_len, best_start_idx): length in actions of the longest
    run alternating between two opposite actions, and its first index.
    """
    if len(actions) < 3:
        return 0, None
    best_len, best_start = 0, None
    run_start = 0
    for i in range(2, len(actions)):
        # The run continues while each action repeats the one two steps back
        # and reverses the previous one.
        if actions[i] == actions[i - 2] and OPPOSITE.get(actions[i - 1]) == actions[i]:
            pass
        else:
            run_len = i - 1 - run_start
            if run_len > best_len:
                best_len, best_start = run_len, run_start
            run_start = i - 1
    run_len = len(actions) - run_start
    if run_len > best_len:
        best_len, best_start = run_len, run_start
    return best_len, best_start


def remaining_counts(world):
    crates_left = int((world.arena == 1).sum())
    coins_left = sum(1 for c in world.coins if c.collectable)
    return crates_left, coins_left


def run_and_diagnose():
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(CHECKPOINT))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario=SCENARIO, seed=SEED, train=False)
    agent = world.agents[0]

    round_results = []
    example_traces = []  # (round_idx, trace) for a few oscillating rounds.

    for round_idx in range(1, N_ROUNDS + 1):
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
                crates_left, coins_left = remaining_counts(world)
                record = {"step": state["step"], "pos": semantic.self_pos,
                          "legal": legal, "crates_left": crates_left, "coins_left": coins_left}
            world.do_step("WAIT")
            if record is not None:
                trace.append(record)

        actions = list(world.replay["actions"][agent.name])
        for rec, a in zip(trace, actions):
            rec["chosen"] = a

        reverse_count = sum(
            1 for prev, cur in zip(actions, actions[1:]) if OPPOSITE.get(prev) == cur
        )
        run_len, start_idx = longest_oscillation_run(actions)
        sustained = run_len >= OSCILLATION_THRESHOLD

        entry = {
            "round": round_idx, "steps": len(actions),
            "immediate_reverse_count": reverse_count,
            "longest_oscillation_run": run_len,
            "sustained": sustained,
        }
        if sustained and start_idx is not None:
            start_rec = trace[start_idx]
            entry["oscillation_start_step"] = start_rec["step"]
            entry["crates_left_at_start"] = start_rec["crates_left"]
            entry["coins_left_at_start"] = start_rec["coins_left"]
            entry["oscillation_runs_to_round_end"] = (start_idx + run_len) >= len(trace) - 1
            if len(example_traces) < 3:
                example_traces.append((round_idx, trace[max(0, start_idx - 2):start_idx + min(run_len + 5, len(trace) - start_idx)]))
        round_results.append(entry)

    world.end()
    return round_results, example_traces


def main():
    log_file, log_path = common.open_log_file("task2_oscillation_diagnostic")
    common.log_print(log_file, f"Oscillation diagnostic (canonical checkpoint) -- log file: {log_path}")
    common.log_print(log_file, f"scenario={SCENARIO} seed={SEED} n_rounds={N_ROUNDS} threshold={OSCILLATION_THRESHOLD}")

    round_results, example_traces = run_and_diagnose()

    sustained_rounds = [r for r in round_results if r["sustained"]]
    common.log_print(log_file, f"\nrounds with sustained oscillation (run>={OSCILLATION_THRESHOLD}): "
                                f"{len(sustained_rounds)}/{len(round_results)}")
    for r in round_results:
        common.log_print(
            log_file,
            f"  round={r['round']:>2} steps={r['steps']:>4} reverse_count={r['immediate_reverse_count']:>3} "
            f"longest_run={r['longest_oscillation_run']:>3} sustained={r['sustained']}"
            + (f" start_step={r.get('oscillation_start_step')} "
               f"crates_left={r.get('crates_left_at_start')} coins_left={r.get('coins_left_at_start')} "
               f"runs_to_end={r.get('oscillation_runs_to_round_end')}" if r["sustained"] else ""),
        )

    if sustained_rounds:
        crates_left_vals = [r["crates_left_at_start"] for r in sustained_rounds]
        coins_left_vals = [r["coins_left_at_start"] for r in sustained_rounds]
        runs_to_end = sum(1 for r in sustained_rounds if r["oscillation_runs_to_round_end"])
        common.log_print(
            log_file,
            f"\ncrates_left_at_oscillation_start: mean={np.mean(crates_left_vals):.1f} "
            f"median={np.median(crates_left_vals):.1f} min={min(crates_left_vals)} max={max(crates_left_vals)}",
        )
        common.log_print(
            log_file,
            f"coins_left_at_oscillation_start: mean={np.mean(coins_left_vals):.1f} "
            f"median={np.median(coins_left_vals):.1f} min={min(coins_left_vals)} max={max(coins_left_vals)}",
        )
        common.log_print(log_file, f"oscillation runs until round end: {runs_to_end}/{len(sustained_rounds)}")

    common.log_print(log_file, "\n=== example traces (mask/legal_actions during oscillation) ===")
    for round_idx, trace_window in example_traces:
        common.log_print(log_file, f"\n-- round {round_idx} --")
        for rec in trace_window:
            common.log_print(
                log_file,
                f"  step={rec['step']:>4} pos={rec['pos']} crates_left={rec['crates_left']} "
                f"coins_left={rec['coins_left']} legal={rec['legal']} chosen={rec['chosen']}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
