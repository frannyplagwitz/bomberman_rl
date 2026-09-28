"""Task 4 offline investigation, item f: round-by-round paired comparison of
task4_seed{0,1,2}_breaker_only_A.pkl.gz vs task4_seed{0,1,2}_deadlock_A.pkl.gz.
Read-only.

Both runs share eval_seed and the opponent-RNG pin but are not reseeded per
round, so any trajectory change shifts all later rounds. Reports whether
rounds before the first deadlock-bomb trigger are identical and how the runs
diverge afterwards. Separate launches are not guaranteed to be
reproducible, and the recordings cannot identify the source.
"""
import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed, group):
    label = "breaker_only" if group == "breaker_only" else "deadlock"
    path = DATA_DIR / f"task4_seed{seed}_{label}_A.pkl.gz"
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError(f"{path} is a partial recording")
    return payload


def _round_triggered(rec) -> bool:
    return any(x.get("deadlock_bomb_triggered") for x in rec["steps"])


def _outcome(rec) -> str:
    m = rec["metrics"]
    if m["self_kill"]:
        return "self_kill"
    if m["got_killed_by_opponent"]:
        return "got_killed"
    return "survived"


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation_deadlock_pairs_f.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== (f) deadlock-bomb paired analysis (breaker_only vs deadlock, by round) ===")

    n_untriggered_identical = 0
    n_untriggered_different = 0
    different_cases = []
    triggered_rounds = []  # (seed, round, bo_rec, dl_rec)
    pre_first_trigger = {"identical": 0, "different": 0}
    post_first_trigger_untriggered = {"identical": 0, "different": 0}

    for s in SEEDS:
        bo = load(s, "breaker_only")
        dl = load(s, "deadlock")
        bo_by_round = {r["round"]: r for r in bo["records"]}
        dl_by_round = {r["round"]: r for r in dl["records"]}
        common_rounds = sorted(set(bo_by_round) & set(dl_by_round))
        assert len(common_rounds) == 100, f"seed{s}: expected 100 shared round numbers, got {len(common_rounds)}"
        first_trigger_round = next((r for r in common_rounds if _round_triggered(dl_by_round[r])), None)
        for rnd in common_rounds:
            bo_r, dl_r = bo_by_round[rnd], dl_by_round[rnd]
            triggered = _round_triggered(dl_r)
            if not triggered:
                bo_actions = bo_r["metrics"]["actions"]
                dl_actions = dl_r["metrics"]["actions"]
                same_actions = bo_actions == dl_actions
                same_outcome = _outcome(bo_r) == _outcome(dl_r)
                bucket = "identical" if (same_actions and same_outcome) else "different"
                if first_trigger_round is not None and rnd < first_trigger_round:
                    pre_first_trigger[bucket] += 1
                else:
                    post_first_trigger_untriggered[bucket] += 1
                if same_actions and same_outcome:
                    n_untriggered_identical += 1
                else:
                    n_untriggered_different += 1
                    different_cases.append({
                        "seed": s, "round": rnd, "same_actions": same_actions, "same_outcome": same_outcome,
                        "bo_outcome": _outcome(bo_r), "dl_outcome": _outcome(dl_r),
                        "bo_len": len(bo_actions), "dl_len": len(dl_actions),
                        "before_first_trigger": first_trigger_round is not None and rnd < first_trigger_round,
                    })
            else:
                triggered_rounds.append((s, rnd, bo_r, dl_r))

    out("\n--- f.1: verification on never-triggered rounds ---")
    n_total_untriggered = n_untriggered_identical + n_untriggered_different
    out(f"never-triggered rounds (deadlock_bomb_triggered False for the whole round, n={n_total_untriggered}): "
        f"identical action-sequence+outcome={n_untriggered_identical} "
        f"different={n_untriggered_different}")
    if n_untriggered_different == 0:
        out("CONFIRMED: pairing assumption holds -- every never-triggered round has byte-identical "
            "action sequence and outcome between breaker_only and deadlock. The two runs are a valid "
            "counterfactual pair (same RNG stream, same rhine actions until the override fires).")
    else:
        out(f"PAIRING ASSUMPTION DOES NOT HOLD for {n_untriggered_different}/{n_total_untriggered} "
            "never-triggered round(s).")
        pre_total = pre_first_trigger["identical"] + pre_first_trigger["different"]
        post_total = post_first_trigger_untriggered["identical"] + post_first_trigger_untriggered["different"]
        out(f"  broken down by position relative to each seed's own first trigger round:")
        out(f"    BEFORE first trigger (n={pre_total}): identical={pre_first_trigger['identical']} "
            f"different={pre_first_trigger['different']}")
        out(f"    AT/AFTER first trigger, this specific round untriggered (n={post_total}): "
            f"identical={post_first_trigger_untriggered['identical']} "
            f"different={post_first_trigger_untriggered['different']}")
        if pre_first_trigger["different"] == 0:
            out("  -> Consistent with the cascading-RNG-drift hypothesis: every pre-first-trigger round "
                "matches exactly, and divergence only appears once a trigger has fired somewhere earlier "
                "in that seed's run (rule_based_agent's shared `random`-module stream accumulates draws "
                "across the whole 100-round run with no per-round reseed, so a round whose length/"
                "trajectory changed shifts everything that follows it).")
        else:
            out(f"  -> NOT fully consistent with the cascading-RNG-drift hypothesis: "
                f"{pre_first_trigger['different']} round(s) already diverge BEFORE any trigger has fired "
                "in that seed's run. Some source of run-to-run nondeterminism between the two separate "
                "process launches exists independent of the deadlock-bomb mechanism itself; this "
                "recording has no lower-level RNG-draw-count data to identify it, and re-running games "
                "to trace it is out of scope for this read-only investigation -- reported as an open "
                "finding, not root-caused.")
        out("\n  full list of divergent never-triggered rounds:")
        for c in different_cases:
            out(f"    seed{c['seed']} round={c['round']}: same_actions={c['same_actions']} "
                f"same_outcome={c['same_outcome']} bo_outcome={c['bo_outcome']} dl_outcome={c['dl_outcome']} "
                f"bo_len={c['bo_len']} dl_len={c['dl_len']} before_first_trigger={c['before_first_trigger']}")

    out(f"\nsample size: {n_total_untriggered} never-triggered rounds out of 300 total (3 seeds x 100); "
        f"{len(triggered_rounds)} rounds had >=1 deadlock-bomb trigger.")

    out("\n--- f.2: paired outcomes for triggered rounds ---")
    transition = Counter()
    score_diffs = []
    for s, rnd, bo_r, dl_r in triggered_rounds:
        bo_out, dl_out = _outcome(bo_r), _outcome(dl_r)
        bo_score = bo_r["metrics"]["coins_collected"] * 1 + bo_r["metrics"]["opponent_kills"] * 5
        dl_score = dl_r["metrics"]["coins_collected"] * 1 + dl_r["metrics"]["opponent_kills"] * 5
        score_diff = dl_score - bo_score
        score_diffs.append(score_diff)
        transition[(bo_out, dl_out)] += 1
        out(f"  seed{s} round={rnd}: breaker_only={bo_out} (score={bo_score}) -> "
            f"deadlock={dl_out} (score={dl_score}) score_diff={score_diff:+d}")

    out(f"\ntransition table (breaker_only_outcome -> deadlock_outcome), n={len(triggered_rounds)}:")
    for (bo_out, dl_out), n in sorted(transition.items()):
        out(f"  {bo_out} -> {dl_out}: {n}")
    n_gk_appeared = transition.get(("survived", "got_killed"), 0) + transition.get(("self_kill", "got_killed"), 0)
    n_gk_resolved = transition.get(("got_killed", "survived"), 0) + transition.get(("got_killed", "self_kill"), 0)
    out(f"\ngot_killed transitions: no-got_killed -> got_killed = {n_gk_appeared}; "
        f"got_killed -> no-got_killed = {n_gk_resolved}")
    out(f"mean score_diff over triggered rounds (deadlock - breaker_only): "
        f"{sum(score_diffs) / len(score_diffs) if score_diffs else float('nan'):+.2f} (n={len(score_diffs)})")

    out("\n--- f.3: deadlock-killed-but-breaker_only-survived cases ---")
    newly_killed = [
        (s, rnd, bo_r, dl_r) for s, rnd, bo_r, dl_r in triggered_rounds
        if _outcome(bo_r) != "got_killed" and _outcome(dl_r) == "got_killed"
    ]
    if not newly_killed:
        out("n=0 -- no round where the deadlock-bomb group was got_killed while breaker_only was not. "
            "Nothing to describe (data insufficient because the case doesn't occur, not because it's "
            "unmeasured).")
    else:
        out(f"n={len(newly_killed)}:")
        for s, rnd, bo_r, dl_r in newly_killed:
            dl_steps = dl_r["steps"]
            trig_idx = next(i for i, x in enumerate(dl_steps) if x.get("deadlock_bomb_triggered"))
            death = dl_r["death"]
            death_step = death["step"] if death else dl_steps[-1]["state"]["step"]
            post_trigger = dl_steps[trig_idx:]
            out(f"  seed{s} round={rnd}: trigger_step={dl_steps[trig_idx]['state']['step']} "
                f"death_step={death_step} steps_trigger_to_death={death_step - dl_steps[trig_idx]['state']['step']}")
            out(f"    actions from trigger to end: {[x['action'] for x in post_trigger]}")
            out(f"    breaker_only outcome for the same round: {_outcome(bo_r)} "
                f"(actions: {bo_r['metrics']['actions'][trig_idx:trig_idx + len(post_trigger)]})")

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
