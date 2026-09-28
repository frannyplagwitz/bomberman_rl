"""Diagnostic: does CPU contention during evaluation lower the measured
oscillation_fraction through think-time forced WAITs?

Read-only (train=False, no updates, no checkpoint writes). The concurrent
load for condition B is a training process the caller starts separately.

Per round, a step counts as a forced-WAIT candidate when
agent.available_think_time <= 0 before it; each round's oscillation label
and forced-WAIT count come from the same rollout.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run


def run_instrumented_evaluation(checkpoint_path: str, n_rounds: int, seed: int, scenario: str = "loot-crate") -> dict:
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_PPO_OVERRIDE", None)

    world = common.build_world(scenario=scenario, seed=seed, train=False)
    agent = world.agents[0]

    per_round = []  # Per round: {"oscillating", "forced_wait_steps", "n_steps"}.
    total_forced_wait_steps = 0
    total_steps = 0

    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        forced_wait_steps = 0
        n_steps = 0
        while world.running:
            think_time_before = agent.available_think_time
            world.do_step("WAIT")
            n_steps += 1
            if think_time_before is not None and think_time_before <= 0:
                forced_wait_steps += 1

        actions = list(world.replay["actions"][agent.name])
        run_len, _ = longest_oscillation_run(actions)
        oscillating = run_len >= OSCILLATION_THRESHOLD

        per_round.append({
            "oscillating": oscillating,
            "forced_wait_steps": forced_wait_steps,
            "n_steps": n_steps,
        })
        total_forced_wait_steps += forced_wait_steps
        total_steps += n_steps

    world.end()

    oscillating_rounds = [r for r in per_round if r["oscillating"]]
    non_oscillating_rounds = [r for r in per_round if not r["oscillating"]]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_rounds": n_rounds,
        "oscillation_fraction": len(oscillating_rounds) / n_rounds if n_rounds else float("nan"),
        "total_forced_wait_steps": total_forced_wait_steps,
        "total_steps": total_steps,
        "forced_wait_rate": total_forced_wait_steps / total_steps if total_steps else float("nan"),
        "mean_forced_wait_steps_oscillating_rounds": _mean([r["forced_wait_steps"] for r in oscillating_rounds]),
        "mean_forced_wait_steps_non_oscillating_rounds": _mean([r["forced_wait_steps"] for r in non_oscillating_rounds]),
        "per_round": per_round,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file(f"task2_forced_wait_vs_oscillation_{args.label}")
    t0 = time.time()
    result = run_instrumented_evaluation(args.checkpoint, args.n_rounds, args.seed)
    elapsed = time.time() - t0

    common.log_print(
        log_file,
        f"[{args.label}] checkpoint={args.checkpoint} n_rounds={args.n_rounds} seed={args.seed} elapsed={elapsed:.1f}s\n"
        f"  oscillation_fraction={result['oscillation_fraction']:.3f}\n"
        f"  total_forced_wait_steps={result['total_forced_wait_steps']} total_steps={result['total_steps']} "
        f"forced_wait_rate={result['forced_wait_rate']:.4f}\n"
        f"  mean_forced_wait_steps_oscillating_rounds={result['mean_forced_wait_steps_oscillating_rounds']:.3f}\n"
        f"  mean_forced_wait_steps_non_oscillating_rounds={result['mean_forced_wait_steps_non_oscillating_rounds']:.3f}",
    )
    common.ring_bell()


if __name__ == "__main__":
    main()
