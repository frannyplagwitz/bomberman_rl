"""Diagnostic: is Task 2 Stage A's zero completion_rate caused by the MAX_STEPS
cap or by the policy?

Raises settings.MAX_STEPS for this process only (runtime monkeypatch) and runs
each ablation's final checkpoint for 30 deterministic loot-crate rounds,
recording the steps-to-completion distribution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import settings as s

s.MAX_STEPS = 2000  # Must be set before any BombeRLeWorld is built.

from agent_code.rhine.scripts import common


def main():
    log_file, log_path = common.open_log_file("task2_step_limit_diagnostic")
    common.log_print(log_file, f"Step-limit diagnostic (MAX_STEPS={s.MAX_STEPS}) -- log file: {log_path}")

    for ablation in ["B1", "B2"]:
        checkpoint_path = common.MODELS_DIR / f"task2_stage_a_{ablation}.pt"
        episodes, world, _ = common.run_evaluation_with_self_kill_tracing(
            n_rounds=30, scenario="loot-crate", seed=1000, init_checkpoint=str(checkpoint_path),
        )
        steps = [ep.steps for ep in episodes]
        completions = [ep for ep in episodes if ep.completed]
        hit_cap = sum(1 for st in steps if st >= s.MAX_STEPS)

        common.log_print(log_file, f"\n=== {ablation} (checkpoint: {checkpoint_path.name}) ===")
        common.log_print(log_file, f"steps per round: {sorted(steps)}")
        common.log_print(
            log_file,
            f"completion_rate={len(completions)/len(episodes):.3f} "
            f"({len(completions)}/{len(episodes)}), "
            f"hit_{s.MAX_STEPS}_cap={hit_cap}/{len(episodes)}",
        )
        if completions:
            comp_steps = [ep.steps for ep in completions]
            common.log_print(
                log_file,
                f"steps_to_completion: mean={np.mean(comp_steps):.1f} "
                f"median={np.median(comp_steps):.1f} max={np.max(comp_steps)} min={np.min(comp_steps)}",
            )
        common.log_print(
            log_file,
            f"steps distribution: mean={np.mean(steps):.1f} median={np.median(steps):.1f} "
            f"p90={np.percentile(steps, 90):.1f} max={np.max(steps)}",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
