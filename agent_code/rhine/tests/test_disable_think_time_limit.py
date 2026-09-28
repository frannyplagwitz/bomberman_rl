"""Tests for common._disable_think_time_limit() and its wiring into
_run_one_round_traced()/run_evaluation_with_self_kill_tracing(). Uses a real
BombeRLeWorld (train=False), since the mechanism is a real framework
interaction.
"""
import pytest

from agent_code.rhine.scripts import common


def _clear_env():
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)


def test_disable_think_time_limit_sets_available_think_time_to_inf():
    _clear_env()
    world = common.build_world(scenario="loot-crate", seed=0, train=False)
    world.new_round()
    common._disable_think_time_limit(world)
    assert world.agents[0].available_think_time == float("inf")
    world.end()


def test_disable_think_time_limit_does_not_touch_agent_train_flag():
    """The override must not flip agent.train, which would make the framework
    dispatch training callbacks to an agent never set up for training.
    """
    _clear_env()
    world = common.build_world(scenario="loot-crate", seed=0, train=False)
    world.new_round()
    common._disable_think_time_limit(world)
    assert world.agents[0].train is False
    world.end()


def test_one_time_override_after_new_round_does_not_survive_a_normal_step():
    """A single override after new_round() does not survive a normal step,
    because poll_and_run_agents() resets the budget after each step; hence the
    override must be re-applied before every do_step().
    """
    _clear_env()
    world = common.build_world(scenario="loot-crate", seed=0, train=False)
    world.new_round()
    agent = world.agents[0]
    common._disable_think_time_limit(world)
    assert agent.available_think_time == float("inf")

    world.do_step("WAIT")

    assert agent.available_think_time == pytest.approx(agent.base_timeout)
    assert agent.available_think_time != float("inf")
    world.end()


def test_disable_think_time_limit_prevents_forced_wait_from_prior_deficit():
    """An agent in think-time deficit has its next action forced to WAIT and
    the deficit extended, unless the override is applied first, in which case
    the step takes the normal branch and the budget is reset.
    """
    _clear_env()
    world = common.build_world(scenario="loot-crate", seed=0, train=False)
    world.new_round()
    agent = world.agents[0]
    base_timeout = agent.base_timeout

    agent.available_think_time = -1.0
    world.do_step("WAIT")
    assert agent.available_think_time == pytest.approx(-1.0 + base_timeout)  # Skip branch.

    agent.available_think_time = -1.0
    common._disable_think_time_limit(world)
    world.do_step("WAIT")
    assert agent.available_think_time == pytest.approx(base_timeout)  # Normal branch.
    world.end()


def test_run_evaluation_with_self_kill_tracing_disable_think_time_limit_runs_clean():
    """With the flag on, a short real evaluation runs cleanly and every
    recorded available_think_time_before stays positive.
    """
    _clear_env()
    episodes, world, trace_paths = common.run_evaluation_with_self_kill_tracing(
        n_rounds=2, scenario="loot-crate", seed=1000, init_checkpoint=None,
        disable_think_time_limit=True,
    )
    assert len(episodes) == 2
    world.end()


def test_run_evaluation_with_self_kill_tracing_default_off_is_unaffected():
    """Omitting the flag leaves existing callers unaffected."""
    _clear_env()
    episodes, world, trace_paths = common.run_evaluation_with_self_kill_tracing(
        n_rounds=2, scenario="loot-crate", seed=1000, init_checkpoint=None,
    )
    assert len(episodes) == 2
    world.end()
