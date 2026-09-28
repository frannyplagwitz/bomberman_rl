"""Validation experiment: does a larger critic improve explained variance, or
is the low EV of the canonical B2 checkpoint caused by something else
(features, reward scale, training length)?

Two architectures, both from a fresh random init with the same short budget
and B2's reward config:

  baseline       -- unchanged ActorCriticMLP (critic head directly on the
                    shared trunk).
  wider_critic   -- same shared trunk and actor, plus a critic-only hidden
                    layer before the value head.

Checkpoints go to a scratch dir under logs/. Explained variance is computed as
in compute_explained_variance.py.
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn

from agent_code.rhine.scripts import common
from agent_code.rhine import config as cfg
from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine.model import ActorCriticMLP, masked_logits  # noqa: F401
from agent_code.rhine.rewards import compute_reward

REWARD_CONFIG = dataclasses.replace(
    cfg.REWARD_CONFIG,
    ENABLE_BOMBING_PROGRESS_SHAPING=True, ENABLE_STALL_PENALTY=True, ENABLE_STALL_PENALTY_V2=False,
)
REWARD_OVERRIDE = {"ENABLE_BOMBING_PROGRESS_SHAPING": True, "ENABLE_STALL_PENALTY": True, "ENABLE_STALL_PENALTY_V2": False}
GAMMA = cfg.PPO_CONFIG.gamma

SCRATCH_DIR = common.LOGS_DIR / "critic_capacity_experiment"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

BURST_STEPS = 20_000
N_BURSTS = 3
EVAL_ROUNDS = 15
EVAL_SEED = 1000
TRAIN_SEED = 0


class WiderCriticMLP(nn.Module):
    """Same actor path as ActorCriticMLP; the critic gets its own extra hidden
    layer instead of reading directly off the shared trunk.
    """

    def __init__(self, n_features: int, n_actions: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(in_dim, n_actions)
        self.critic_extra = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh())
        self.critic_head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor):
        z = self.trunk(features)
        logits = self.actor_head(z)
        value = self.critic_head(self.critic_extra(z)).squeeze(-1)
        return logits, value


VARIANTS = {
    "baseline": ActorCriticMLP,
    "wider_critic": WiderCriticMLP,
}


def run_episodes_capped_by_steps(model_cls, checkpoint_path, init_ckpt, seed, step_budget):
    """Runs rounds one at a time until step_budget is reached, since
    loot-crate episode lengths vary widely.
    """
    rhine_callbacks.ActorCriticMLP = model_cls
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", init_ckpt)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", '{"ENABLE_BOMBING_PROGRESS_SHAPING": true, "ENABLE_STALL_PENALTY": true, "ENABLE_STALL_PENALTY_V2": false}')

    world = common.build_world(scenario="loot-crate", seed=seed, train=True)
    total = 0
    while total < step_budget:
        ep = common.run_one_round(world)
        total += ep.steps
    world.end()
    return total


def rollout_for_ev(model_cls, checkpoint_path, seed, n_rounds):
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", str(checkpoint_path))
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)
    rhine_callbacks.ActorCriticMLP = model_cls

    world = common.build_world(scenario="loot-crate", seed=seed, train=False)
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self

    all_values, all_returns = [], []
    for _ in range(n_rounds):
        world.new_round()
        world.user_input = "WAIT"
        ep_values, ep_rewards = [], []
        while world.running:
            old_state = world.get_state_for_agent(agent)
            world.do_step("WAIT")
            new_state = world.get_state_for_agent(agent)
            action = world.replay["actions"][agent.name][-1]
            events = list(agent.events)
            reward = compute_reward(old_state, action, new_state, events, REWARD_CONFIG)
            ep_values.append(fake_self.last_value)
            ep_rewards.append(reward)
        returns = np.zeros(len(ep_rewards))
        running = 0.0
        for t in reversed(range(len(ep_rewards))):
            running = ep_rewards[t] + GAMMA * running
            returns[t] = running
        all_values.extend(ep_values)
        all_returns.extend(returns.tolist())
    world.end()
    return np.array(all_values), np.array(all_returns)


def explained_variance(values, returns):
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return float("nan")
    return 1.0 - np.var(returns - values) / var_returns


def main():
    log_file, log_path = common.open_log_file("critic_capacity_experiment")
    common.log_print(log_file, f"Critic capacity experiment -- log file: {log_path}")
    common.log_print(
        log_file,
        f"BURST_STEPS={BURST_STEPS} N_BURSTS={N_BURSTS} EVAL_ROUNDS={EVAL_ROUNDS} "
        f"(reward: B2, progress shaping ON; scenario: loot-crate)",
    )

    for variant_name, model_cls in VARIANTS.items():
        common.log_print(log_file, f"\n=== {variant_name} ===")
        checkpoint_path = SCRATCH_DIR / f"{variant_name}.pt"
        init_ckpt = None
        total_steps = 0
        for burst in range(N_BURSTS):
            steps_done = run_episodes_capped_by_steps(
                model_cls, checkpoint_path, init_ckpt, TRAIN_SEED + burst, BURST_STEPS
            )
            total_steps += steps_done
            init_ckpt = str(checkpoint_path)

            values, returns = rollout_for_ev(model_cls, checkpoint_path, EVAL_SEED, EVAL_ROUNDS)
            ev = explained_variance(values, returns)
            common.log_print(
                log_file,
                f"  after {total_steps:>6} steps: n_eval_steps={len(values):>5} "
                f"value_mean={values.mean():+.3f} return_mean={returns.mean():+.3f} "
                f"var_return={returns.var():.3f} explained_variance={ev:+.4f}",
            )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
