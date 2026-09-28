"""Diagnostic: PPO critic explained variance, reconstructed post-hoc from Task 2
Stage A's per-round checkpoint snapshots (models/snapshots/).

explained_variance = 1 - Var(returns - values) / Var(returns)

Each snapshot is played deterministically (train=False, weights frozen) step by
step; each step's reward is recomputed with rewards.compute_reward under the
matching ablation's reward config, and the value estimate act() computed is
read back. Returns are realized discounted rewards-to-go, with truncation
treated as terminal as in train.py.

Limitation: snapshots exist only for the later part of that training run.
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine.scripts import common
from agent_code.rhine import config as cfg
from agent_code.rhine.rewards import compute_reward

# Explicit historical baseline (shaping off, V1 stall penalty on, V2 off).
_PLAIN_BASELINE = dataclasses.replace(
    cfg.REWARD_CONFIG,
    ENABLE_BOMBING_PROGRESS_SHAPING=False, ENABLE_STALL_PENALTY=True, ENABLE_STALL_PENALTY_V2=False,
)
ABLATION_REWARD_CONFIG = {
    "B1": _PLAIN_BASELINE,
    "B2": dataclasses.replace(_PLAIN_BASELINE, ENABLE_BOMBING_PROGRESS_SHAPING=True),
}
GAMMA = cfg.PPO_CONFIG.gamma


def rollout_for_ev(checkpoint_path: str, ablation: str, n_rounds: int, seed: int):
    reward_config = ABLATION_REWARD_CONFIG[ablation]

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", checkpoint_path)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = common.build_world(scenario="loot-crate", seed=seed, train=False)
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self

    all_values = []
    all_returns = []

    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        episode_values = []
        episode_rewards = []
        while world.running:
            old_state = world.get_state_for_agent(agent)
            world.do_step("WAIT")
            new_state = world.get_state_for_agent(agent)
            action = world.replay["actions"][agent.name][-1]
            events = list(agent.events)  # This step's events, not yet reset.
            reward = compute_reward(old_state, action, new_state, events, reward_config)
            episode_values.append(fake_self.last_value)
            episode_rewards.append(reward)

        returns = np.zeros(len(episode_rewards))
        running = 0.0
        for t in reversed(range(len(episode_rewards))):
            running = episode_rewards[t] + GAMMA * running
            returns[t] = running
        all_values.extend(episode_values)
        all_returns.extend(returns.tolist())

    world.end()
    return np.array(all_values), np.array(all_returns)


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return float("nan")
    return 1.0 - np.var(returns - values) / var_returns


def main():
    log_file, log_path = common.open_log_file("task2_explained_variance")
    common.log_print(log_file, f"Explained variance diagnostic -- log file: {log_path}")

    for ablation in ["B1", "B2"]:
        common.log_print(log_file, f"\n=== {ablation} ===")
        snap_dir = common.MODELS_DIR / "snapshots"
        snapshots = sorted(
            snap_dir.glob(f"task2_stage_a_{ablation}_round*.pt"),
            key=lambda p: int(p.stem.rsplit("round", 1)[-1]),
        )
        if not snapshots:
            common.log_print(log_file, "  no snapshots found")
            continue
        for snap in snapshots:
            round_num = int(snap.stem.rsplit("round", 1)[-1])
            values, returns = rollout_for_ev(str(snap), ablation, n_rounds=15, seed=1000)
            ev = explained_variance(values, returns)
            common.log_print(
                log_file,
                f"  round={round_num:>4} n_steps={len(values):>5} "
                f"value_mean={values.mean():+.3f} return_mean={returns.mean():+.3f} "
                f"var_return={returns.var():.3f} var_value={values.var():.3f} "
                f"explained_variance={ev:+.4f}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
