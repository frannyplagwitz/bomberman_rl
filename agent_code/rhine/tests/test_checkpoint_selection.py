"""Tests for run_stage_a2_task3.py's _select_best_checkpoint(), a pure function
over per-eval-point dicts: got_killed_by_opponent_rate as a filter-only
criterion alongside self_kill_rate/oscillation_fraction, and per-candidate
filter-reason reporting.
"""
from agent_code.rhine.scripts.run_stage_a2_task3 import _select_best_checkpoint


def _point(round_, score_mean, self_kill_rate=0.01, got_killed_by_opponent_rate=0.01,
           oscillation_fraction=0.05, completion_rate=0.5):
    return {
        "round": round_, "snapshot_path": f"snap_{round_}.pt", "score_mean": score_mean,
        "self_kill_rate": self_kill_rate, "got_killed_by_opponent_rate": got_killed_by_opponent_rate,
        "oscillation_fraction": oscillation_fraction, "completion_rate": completion_rate,
    }


def test_picks_highest_score_mean_when_all_candidates_qualify():
    history = [_point(r, score) for r, score in
               [(100, 5.0), (200, 6.0), (300, 7.0), (400, 4.0), (500, 3.0), (600, 2.0)]]
    selection = _select_best_checkpoint(history)
    assert selection["chosen"]["round"] == 300
    assert len(selection["shortlist"]) == 5  # Top-5 by score_mean.
    assert len(selection["qualified"]) == 5  # All shortlisted candidates pass.


def test_got_killed_by_opponent_rate_filters_out_top_scorer():
    # The top scorer is a got_killed_by_opponent_rate outlier and must be
    # filtered out despite winning on score_mean.
    history = [_point(r, score, got_killed_by_opponent_rate=0.02) for r, score in
               [(100, 3.0), (200, 2.5), (300, 2.0), (400, 1.5), (500, 1.0)]]
    history.append(_point(600, 9.0, got_killed_by_opponent_rate=0.6))
    selection = _select_best_checkpoint(history)
    assert selection["chosen"]["round"] == 100  # Best score_mean among the qualified rest.
    reasons_600 = selection["shortlist_reasons"][600]
    assert reasons_600 and "got_killed_by_opponent_rate" in reasons_600[0]
    assert selection["shortlist_reasons"][100] == []


def test_self_kill_rate_filter_still_active():
    history = [_point(r, score, self_kill_rate=0.01) for r, score in
               [(100, 3.0), (200, 2.5), (300, 2.0), (400, 1.5), (500, 1.0)]]
    history.append(_point(600, 9.0, self_kill_rate=0.5))
    selection = _select_best_checkpoint(history)
    assert selection["chosen"]["round"] == 100
    assert "self_kill_rate" in selection["shortlist_reasons"][600][0]


def test_falls_back_to_full_shortlist_when_all_candidates_filtered():
    # Every shortlisted candidate is a got_killed_by_opponent_rate outlier, so
    # selection must fall back to the best of the shortlist.
    history = [_point(100 + i, 1.0 + 0.1 * i, got_killed_by_opponent_rate=0.02) for i in range(9)]
    history += [_point(900 + i, 5.0 + i, got_killed_by_opponent_rate=0.5) for i in range(5)]
    selection = _select_best_checkpoint(history)
    assert selection["qualified"] == []
    assert selection["chosen"]["round"] == 904  # Highest score_mean in the shortlist.
    assert all(selection["shortlist_reasons"][h["round"]] for h in selection["shortlist"])


def test_completion_rate_never_filters():
    history = [_point(r, score, completion_rate=cr) for r, score, cr in
               [(100, 5.0, 0.9), (200, 4.0, 0.1), (300, 3.0, 0.5)]]
    selection = _select_best_checkpoint(history)
    assert selection["chosen"]["round"] == 100
    assert len(selection["qualified"]) == 3
