"""Task 4 offline investigation: initiative rate when a covered opponent is
trapped (cannot escape a hypothetical bomb in time). Read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Reuses investigate_task4_bomb_ev.py's _min_escape_steps_for_opponent() and
its trapped criterion (escape steps above FUSE_BUDGET, or no path), applied
at every step.

Definitions:
- opportunity: BOMB legal in the recorded final_mask and the own blast covers
  a trapped alive opponent (hypothetical bomb at self_pos).
- event: a maximal run of consecutive opportunity steps within a round;
  "acted" if any step in it chose BOMB.
- hit window: KILL_CHECK_WINDOW steps after the first BOMB step (acted) or
  after the last step (non-acted, a naive comparison only).
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_task4_bomb_ev import _min_escape_steps_for_opponent
from agent_code.rhine.state_processing import blast_coords, extract_semantic_state

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
FUSE_BUDGET = cfg.BOMB_TIMER + 1
KILL_CHECK_WINDOW = cfg.BOMB_TIMER + 1
BOMB_IDX = cfg.ACTIONS.index("BOMB")


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed):
    with gzip.open(DATA_DIR / f"task4_seed{seed}_breaker_only_A.pkl.gz", "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError("partial recording")
    return payload


def _opportunity_at_step(x):
    if not x["final_mask"][BOMB_IDX]:
        return False
    sem = extract_semantic_state(restore_state(x["state"]))
    if not sem.opponents:
        return False
    blast = set(blast_coords(sem.field_arr, sem.self_pos, cfg.BOMB_POWER))
    covered = [o for o in sem.opponents if o in blast]
    if not covered:
        return False
    hyp_danger = {t: set(offsets) for t, offsets in sem.danger_offsets.items()}
    for bt in blast:
        hyp_danger.setdefault(bt, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    for o in covered:
        steps_needed = _min_escape_steps_for_opponent(o, sem.field_arr, sem.blocked, hyp_danger, sem.self_pos)
        if steps_needed is None or steps_needed > FUSE_BUDGET:
            return True
    return False


def _gather_events(recs):
    opp_flags = [_opportunity_at_step(x) for x in recs]
    events = []
    i = 0
    while i < len(opp_flags):
        if opp_flags[i]:
            j = i
            while j + 1 < len(opp_flags) and opp_flags[j + 1]:
                j += 1
            events.append((i, j))
            i = j + 1
        else:
            i += 1
    return opp_flags, events


def _hit_within_window(recs, start_idx):
    kills_seq = [x["kills"] for x in recs]
    base_kills = kills_seq[start_idx]
    end = min(start_idx + KILL_CHECK_WINDOW, len(kills_seq) - 1)
    return kills_seq[end] > base_kills if end < len(kills_seq) else False


def item4a_b():
    out("=== Item 4a/4b: opportunity/event step-level and event-level initiative rate, per seed + combined ===")
    all_step_flags = []
    all_events = []
    all_bomb_probs_not_acted = []
    combined = {"n_step_opps": 0, "n_events": 0, "n_step_acted": 0, "n_events_acted": 0,
                "acted_hits": 0, "acted_total": 0, "not_acted_hits": 0, "not_acted_total": 0}

    for s in SEEDS:
        payload = load(s)
        n_rounds_step_opps = []
        n_rounds_events = []
        seed_events_acted = 0
        seed_events_total = 0
        seed_step_opps = 0
        seed_step_acted = 0
        seed_probs_not_acted = []
        acted_hits, acted_total = 0, 0
        not_acted_hits, not_acted_total = 0, 0
        for r in payload["records"]:
            recs = r["steps"]
            opp_flags, events = _gather_events(recs)
            n_rounds_step_opps.append(sum(opp_flags))
            n_rounds_events.append(len(events))
            for i, x in enumerate(recs):
                if opp_flags[i]:
                    seed_step_opps += 1
                    if x["action"] == "BOMB":
                        seed_step_acted += 1
                    else:
                        seed_probs_not_acted.append(float(x["probs_final"][BOMB_IDX]))
            for start, end in events:
                seed_events_total += 1
                bomb_steps_in_event = [k for k in range(start, end + 1) if recs[k]["action"] == "BOMB"]
                if bomb_steps_in_event:
                    seed_events_acted += 1
                    first_bomb = bomb_steps_in_event[0]
                    acted_total += 1
                    if _hit_within_window(recs, first_bomb):
                        acted_hits += 1
                else:
                    not_acted_total += 1
                    if _hit_within_window(recs, end):
                        not_acted_hits += 1

        out(f"\n  seed{s}: mean opportunity steps/round={np.mean(n_rounds_step_opps):.3f} "
            f"mean events/round={np.mean(n_rounds_events):.3f}")
        out(f"    step-level initiative rate: {seed_step_acted}/{seed_step_opps} = "
            f"{100 * seed_step_acted / seed_step_opps if seed_step_opps else float('nan'):.3f}%")
        out(f"    event-level initiative rate: {seed_events_acted}/{seed_events_total} = "
            f"{100 * seed_events_acted / seed_events_total if seed_events_total else float('nan'):.3f}%")
        if seed_probs_not_acted:
            arr = np.asarray(seed_probs_not_acted)
            out(f"    BOMB probs_final on non-acted opportunity steps: n={len(arr)} mean={arr.mean():.4f} "
                f"median={np.median(arr):.4f} p90={np.percentile(arr, 90):.4f} max={arr.max():.4f}")
        out(f"    acted-event hit rate (kill within {KILL_CHECK_WINDOW} ticks): {acted_hits}/{acted_total} = "
            f"{100 * acted_hits / acted_total if acted_total else float('nan'):.3f}%")
        out(f"    non-acted-event hit rate (any kill within {KILL_CHECK_WINDOW} ticks of event end): "
            f"{not_acted_hits}/{not_acted_total} = {100 * not_acted_hits / not_acted_total if not_acted_total else float('nan'):.3f}%")

        combined["n_step_opps"] += seed_step_opps
        combined["n_events"] += seed_events_total
        combined["n_step_acted"] += seed_step_acted
        combined["n_events_acted"] += seed_events_acted
        combined["acted_hits"] += acted_hits
        combined["acted_total"] += acted_total
        combined["not_acted_hits"] += not_acted_hits
        combined["not_acted_total"] += not_acted_total
        all_bomb_probs_not_acted.extend(seed_probs_not_acted)

    out("\n  --- combined (3 seeds) ---")
    out(f"  step-level initiative rate: {combined['n_step_acted']}/{combined['n_step_opps']} = "
        f"{100 * combined['n_step_acted'] / combined['n_step_opps'] if combined['n_step_opps'] else float('nan'):.3f}%")
    out(f"  event-level initiative rate: {combined['n_events_acted']}/{combined['n_events']} = "
        f"{100 * combined['n_events_acted'] / combined['n_events'] if combined['n_events'] else float('nan'):.3f}%")
    if all_bomb_probs_not_acted:
        arr = np.asarray(all_bomb_probs_not_acted)
        out(f"  BOMB probs_final on non-acted opportunity steps: n={len(arr)} mean={arr.mean():.4f} "
            f"median={np.median(arr):.4f} p90={np.percentile(arr, 90):.4f} max={arr.max():.4f}")
    out(f"  acted-event hit rate: {combined['acted_hits']}/{combined['acted_total']} = "
        f"{100 * combined['acted_hits'] / combined['acted_total'] if combined['acted_total'] else float('nan'):.3f}%")
    out(f"  non-acted-event hit rate: {combined['not_acted_hits']}/{combined['not_acted_total']} = "
        f"{100 * combined['not_acted_hits'] / combined['not_acted_total'] if combined['not_acted_total'] else float('nan'):.3f}%")
    if combined["acted_total"] < 30:
        out(f"  NOTE: acted-event n={combined['acted_total']} <30, small sample, for reference only.")


def item4c():
    out("\n=== Item 4c: reconciliation against bomb_ev.py's 183 'trapped' own-bomb placements ===")
    from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
    from agent_code.rhine.state_processing import bomb_threatens_reachable_opponent
    n_trapped_bombs = 0
    n_inside_event = 0
    n_outside_event = 0
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            recs = r["steps"]
            opp_flags, events = _gather_events(recs)
            event_step_set = set()
            for start, end in events:
                event_step_set.update(range(start, end + 1))
            bomb_indices = [i for i, x in enumerate(recs) if x["action"] == "BOMB"]
            for i in bomb_indices:
                x = recs[i]
                gs = restore_state(x["state"])
                sem = extract_semantic_state(gs)
                pos = sem.self_pos
                if not bomb_threatens_reachable_opponent(gs, pos, cfg.BOMB_POWER):
                    continue
                blast = set(blast_coords(sem.field_arr, pos, cfg.BOMB_POWER))
                covered = [o for o in sem.opponents if o in blast]
                hyp_danger = {t: set(offsets) for t, offsets in sem.danger_offsets.items()}
                for bt in blast:
                    hyp_danger.setdefault(bt, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
                steps_list = [
                    _min_escape_steps_for_opponent(o, sem.field_arr, sem.blocked, hyp_danger, pos) for o in covered
                ]
                finite = [v for v in steps_list if v is not None]
                min_steps = min(finite) if finite else None
                if min_steps is not None and min_steps <= FUSE_BUDGET:
                    continue  # Not trapped.
                n_trapped_bombs += 1
                if i in event_step_set:
                    n_inside_event += 1
                else:
                    n_outside_event += 1
    out(f"  trapped own-bomb placements found here: {n_trapped_bombs} (bomb_ev.py reported 183 including (15,1))")
    out(f"  of these, falling inside an item-4 opportunity event: {n_inside_event}; outside: {n_outside_event}")
    if n_outside_event:
        out("  A placement can be 'trapped' (bomb_ev.py's own post-hoc criterion, evaluated on the covered "
            "opponent AFTER the bomb was placed) yet fall outside an item-4 'opportunity event' if the "
            "opportunity criterion's own gating conditions (BOMB legal in final_mask, blast covering an "
            "opponent) were not simultaneously true at that exact recorded step for some other reason -- "
            "e.g. the breaker/tie-break masked BOMB illegal at that instant for unrelated safety reasons, "
            "or floating-point/ordering differences in which opponent set intersects the blast. Reported as "
            "a raw count difference, not further root-caused (read-only, per this round's constraints).")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation4_trapped_opportunity.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item4a_b()
    item4c()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
