"""Task 4 offline investigation: realized value of rhine's own bomb
placements, opponent-directed vs other. Read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Only bombs actually placed are analyzed, split by whether the blast covered
an opponent's current position (bomb_threatens_reachable_opponent(), a pure
snapshot test). Kill attribution uses a KILL_CHECK_WINDOW after placement;
self-kill attribution uses analyze_stage_d_final._classify_selfkill().

"Steps to safety" for a covered opponent is the shortest
_candidate_escape_paths() length rooted at that opponent under this bomb's
hypothetical danger (same setup as _opponent_escape_difficulty()). Needing
at least the fuse budget, or having no path, counts as not escaping in time.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import settings as s
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import (
    _candidate_escape_paths, bomb_threatens_reachable_opponent, crates_in_blast, extract_semantic_state,
)

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
EXCLUDED_15_1 = {(0, r) for r in (21, 26, 27, 32, 36, 60, 74, 75, 87)}
KILL_CHECK_WINDOW = cfg.BOMB_TIMER + 1
FUSE_BUDGET = cfg.BOMB_TIMER + 1


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


def _min_escape_steps_for_opponent(opp, field_arr, blocked, hypothetical_danger, self_pos):
    opp_blocked = (blocked - {opp}) | {self_pos}
    paths = _candidate_escape_paths(opp, field_arr, opp_blocked, hypothetical_danger, start_offset=0)
    if not paths:
        return None
    return min(path[-1][1] for path in paths.values())


def _gather_bombs():
    """One dict per own BOMB placement across all rounds."""
    records = []
    for s_ in SEEDS:
        payload = load(s_)
        for r in payload["records"]:
            recs = r["steps"]
            is_15_1 = (s_, r["round"]) in EXCLUDED_15_1
            lethal_idx = None
            if r["metrics"]["self_kill"]:
                _, lethal_idx, _ = _classify_selfkill(r)
            kills_seq = [x["kills"] for x in recs] + [r["metrics"]["opponent_kills"]]
            bomb_indices = [i for i, x in enumerate(recs) if x["action"] == "BOMB"]
            for pos_i, i in enumerate(bomb_indices):
                x = recs[i]
                gs = restore_state(x["state"])
                sem = extract_semantic_state(gs)
                pos = sem.self_pos
                opponent_directed = bomb_threatens_reachable_opponent(gs, pos, cfg.BOMB_POWER)
                crates = crates_in_blast(sem.field_arr, pos, cfg.BOMB_POWER)
                is_lethal_self = i == lethal_idx
                # Kill attribution: kills rise within KILL_CHECK_WINDOW with no
                # other own bomb placed in between (ambiguous otherwise).
                next_bomb_i = bomb_indices[pos_i + 1] if pos_i + 1 < len(bomb_indices) else len(recs)
                window_end = min(i + KILL_CHECK_WINDOW, len(kills_seq) - 1, next_bomb_i)
                caused_kill = kills_seq[window_end] > x["kills"] if window_end < len(kills_seq) else False
                ambiguous = (next_bomb_i < i + KILL_CHECK_WINDOW) and caused_kill
                min_steps_if_directed = None
                if opponent_directed:
                    hyp_danger = {t: set(o) for t, o in sem.danger_offsets.items()}
                    from agent_code.rhine.state_processing import blast_coords
                    for bt in blast_coords(sem.field_arr, pos, cfg.BOMB_POWER):
                        hyp_danger.setdefault(bt, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
                    blast = set(blast_coords(sem.field_arr, pos, cfg.BOMB_POWER))
                    covered = [o for o in sem.opponents if o in blast]
                    steps_list = [
                        _min_escape_steps_for_opponent(o, sem.field_arr, sem.blocked, hyp_danger, pos)
                        for o in covered
                    ]
                    finite = [v for v in steps_list if v is not None]
                    min_steps_if_directed = min(finite) if finite else None  # None: nobody escapes.
                records.append({
                    "seed": s_, "round": r["round"], "is_15_1": is_15_1, "opponent_directed": opponent_directed,
                    "crates": crates, "is_lethal_self": is_lethal_self, "caused_kill": caused_kill,
                    "ambiguous": ambiguous, "min_escape_steps": min_steps_if_directed,
                })
    return records


def _report_table(records, label):
    out(f"\n--- {label} (n={len(records)}) ---")
    for cls_name, cls_pred in (("opponent-directed", lambda b: b["opponent_directed"]),
                                ("other", lambda b: not b["opponent_directed"])):
        sub = [b for b in records if cls_pred(b)]
        n = len(sub)
        n_kill = sum(1 for b in sub if b["caused_kill"])
        n_ambiguous = sum(1 for b in sub if b["ambiguous"])
        n_selfkill = sum(1 for b in sub if b["is_lethal_self"])
        mean_crates = np.mean([b["crates"] for b in sub]) if sub else float("nan")
        out(f"  {cls_name}: n={n} kills_caused={n_kill} (ambiguous={n_ambiguous}) "
            f"kill_rate={100 * n_kill / n if n else float('nan'):.3f}% "
            f"real_score_from_kills_per_bomb={s.REWARD_KILL * n_kill / n if n else float('nan'):.4f} "
            f"self_kills_caused={n_selfkill} self_kill_rate={100 * n_selfkill / n if n else float('nan'):.3f}% "
            f"mean_crates_destroyed={mean_crates:.2f}")
        if n < 30:
            out(f"    NOTE: n={n} <30, small sample, for reference only.")


def item5c(records):
    out("\n=== Item 5c: opponent-directed bombs, sub-classified by covered-opponent escape steps vs fuse ===")
    out(f"fuse_budget (BOMB_TIMER+1) = {FUSE_BUDGET}; 'no escape path found' counted in the >fuse bucket.")
    for excl_label, recs in (("INCLUDING (15,1)", records), ("EXCLUDING (15,1)", [b for b in records if not b["is_15_1"]])):
        directed = [b for b in recs if b["opponent_directed"]]
        le_fuse = [b for b in directed if b["min_escape_steps"] is not None and b["min_escape_steps"] <= FUSE_BUDGET]
        gt_fuse = [b for b in directed if b not in le_fuse]
        out(f"\n  {excl_label}: opponent-directed n={len(directed)}")
        for name, sub in (("escape_steps<=fuse (opponent likely escapes)", le_fuse),
                           ("escape_steps>fuse OR no path (opponent likely trapped)", gt_fuse)):
            n = len(sub)
            n_kill = sum(1 for b in sub if b["caused_kill"])
            out(f"    {name}: n={n} kills={n_kill} kill_rate={100 * n_kill / n if n else float('nan'):.3f}%"
                + ("  NOTE: n<30, small sample, for reference only." if n < 30 else ""))


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation3_bomb_ev.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== Item 5: real expected score from bomb placements ===")
    out(f"settings.py REWARD_KILL={s.REWARD_KILL} (used for 'real_score_from_kills_per_bomb')")
    records = _gather_bombs()
    out(f"\ntotal own bomb placements: {len(records)}")

    _report_table(records, "5b: INCLUDING (15,1)")
    _report_table([b for b in records if not b["is_15_1"]], "5b: EXCLUDING (15,1)")
    item5c(records)

    out("\n=== Item 5e: correspondence to the Stage D kill-behavior definitions ===")
    out("Stage D kill behavior: opportunity = bomb_available + opponent in blast_coords(self_pos) + BOMB "
        "mask-legal, checked at every step regardless of the action taken; initiative = BOMB chosen given "
        "an opportunity; hit = kill within BOMB_TIMER+1 ticks after bombing.")
    out("This item's classification only covers bombs THAT WERE ACTUALLY PLACED (the 'initiative=True' "
        "subset of that opportunity population), re-split by whether the blast covers an opponent at "
        "placement (same snapshot test as 'opportunity', but evaluated post-hoc on the chosen action "
        "rather than pre-hoc on every candidate step) -- 'opponent-directed' here corresponds to placements "
        "made during a real opportunity per that definition; 'other' corresponds to bombs placed with no "
        "opponent in the blast at that instant (pure crate-clearing or exploratory placements).")

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
