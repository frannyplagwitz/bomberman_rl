from collections import deque

from agent_code.rhine.scripts import common


def _make_episode(self_kill=True):
    return common.EpisodeMetrics(
        round_index=1, steps=5, coins_collected=0, completed=False,
        invalid_action_count=0, wait_count=1, immediate_reverse_count=0,
        crates_destroyed=0, bombs_dropped=0, self_kill=self_kill,
    )


def test_write_self_kill_trace_appends_timing_section_with_correct_values(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "LOGS_DIR", tmp_path)

    trace = [
        {"step": 1, "self_pos": (5, 5), "bomb_available": True, "current_tile_in_danger": False,
         "nearest_threat_timer": 0, "legal_actions": ["UP", "WAIT"], "chosen_action": "UP"},
        {"step": 2, "self_pos": (5, 4), "bomb_available": False, "current_tile_in_danger": True,
         "nearest_threat_timer": 3, "legal_actions": ["DOWN"], "chosen_action": "DOWN"},
        {"step": 3, "self_pos": (5, 5), "bomb_available": False, "current_tile_in_danger": True,
         "nearest_threat_timer": 0, "legal_actions": ["LEFT"], "chosen_action": "WAIT"},
    ]
    # Step 1 is normal, step 2 overruns by itself, step 3 is fast but starts in
    # deficit from an earlier overrun.
    recent_step_timings = deque([(1, 2.1, 0.480), (2, 612.7, 0.500), (3, 3.6, -0.220)], maxlen=5)

    path = common.write_self_kill_trace(trace, _make_episode(), {"seed": 1000}, recent_step_timings)
    content = path.read_text()

    # Original per-step lines are unchanged in format.
    assert "step=  1 pos=(5, 5)" in content
    assert "legal_actions=['UP', 'WAIT'] chosen_action=UP" in content
    assert "step=  3 pos=(5, 5)" in content
    assert "legal_actions=['LEFT'] chosen_action=WAIT" in content

    # Timing section present with matching legal/chosen values.
    assert "Last steps' do_step() wall-clock duration and available_think_time" in content
    assert (
        "step=  1 duration=2.1ms available_think_time_before=0.480s "
        "legal_actions=['UP', 'WAIT'] chosen_action=UP\n"
    ) in content
    # Step 2 overran by itself without a prior deficit.
    assert (
        "step=  2 duration=612.7ms available_think_time_before=0.500s "
        "legal_actions=['DOWN'] chosen_action=DOWN <-- this step itself exceeded 500ms\n"
    ) in content
    # Step 3 was fast but started in deficit, so it was forced to WAIT.
    assert (
        "step=  3 duration=3.6ms available_think_time_before=-0.220s "
        "legal_actions=['LEFT'] chosen_action=WAIT "
        "<-- FORCED WAIT: think-time budget already in deficit before this step\n"
    ) in content
    assert content.count("<-- FORCED WAIT: think-time budget already in deficit before this step") == 1
    assert content.count("<-- this step itself exceeded 500ms") == 1


def test_write_self_kill_trace_without_timings_matches_old_format(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "LOGS_DIR", tmp_path)

    trace = [
        {"step": 1, "self_pos": (5, 5), "bomb_available": True, "current_tile_in_danger": False,
         "nearest_threat_timer": 0, "legal_actions": ["UP"], "chosen_action": "UP"},
    ]

    path_without_arg = common.write_self_kill_trace(trace, _make_episode(), {"seed": 1000})
    path_with_none = common.write_self_kill_trace(trace, _make_episode(), {"seed": 1000}, None)

    for path in (path_without_arg, path_with_none):
        content = path.read_text()
        assert "Last steps' do_step() wall-clock duration" not in content
        assert "step=  1 pos=(5, 5)" in content


def test_run_one_round_traced_produces_no_file_io_regardless_of_outcome(monkeypatch, tmp_path):
    """_run_one_round_traced() keeps its timing buffer in memory and never
    writes files (writing is the caller's explicit choice on self-kill). Also
    checks the (step, duration_ms, available_think_time_before) tuple shape.
    """
    monkeypatch.setattr(common, "LOGS_DIR", tmp_path)
    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)
    world = common.build_world(scenario="loot-crate", seed=1000, train=False)

    metrics, trace, recent_step_timings = common._run_one_round_traced(world)
    world.end()

    assert list(tmp_path.iterdir()) == []
    assert len(recent_step_timings) <= 5
    assert all(
        isinstance(duration, float) and isinstance(think_time, float)
        for _step, duration, think_time in recent_step_timings
    )
    assert len(trace) >= 1
