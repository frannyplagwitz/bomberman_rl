"""Task 4 offline investigation, read-only on
task4_seed{0,1,2}_breaker_only_A.pkl.gz.

Item 1: detailed tail of the single c_new_single_opponent_siege case and the
single unclassified case (labels from investigate_task4_core.py's section_c()).

Item 2: the repeated death coordinate, plus a death-coordinate repeat census
across all breaker_only rounds (self-kill and got_killed).
"""
import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill, _siege_or_c1
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _instrument_step
from agent_code.rhine.scripts.eval_stage_d_final import restore_state

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None


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


# Fixed spawn corners of the classic scenario.
SPAWN_CORNERS = {(1, 1), (1, 15), (15, 1), (15, 15)}


def item1():
    out("=== Item 1: c-class siege case and unclassified case, full tail description ===")
    out("Reuses: analyze_stage_d_final._classify_selfkill()/_siege_or_c1() labeling unchanged.")
    found = []
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            if not r["metrics"]["self_kill"]:
                continue
            label, bomb_idx, flags = _classify_selfkill(r)
            if label == "unclassified" and bomb_idx is not None:
                findings = _siege_or_c1(r, bomb_idx)
                siege = [f for f in findings if f[2] == 1 and f[3] >= 2 and f[4]]
                conflict_later = [f for f in findings if f[2] >= 1]
                if siege:
                    label = "c_new_single_opponent_siege"
                elif conflict_later:
                    label = "c1_later_leg_conflict"
            if label in ("c_new_single_opponent_siege", "unclassified"):
                found.append((s, r, label, bomb_idx))

    out(f"\nn={len(found)} case(s) matching (expect 1 siege + 1 unclassified per the earlier run)")
    for s, r, label, bomb_idx in found:
        recs = r["steps"]
        death = r["death"]
        tail = recs[-10:]
        out(f"\n--- seed{s} round={r['round']} label={label} bomb_idx={bomb_idx} death_step={death['step'] if death else None} death_pos={tuple(death['pos']) if death else None} ---")
        for x in tail:
            st = x["state"]
            pos = tuple(int(v) for v in st["self"][3])
            others = [tuple(int(v) for v in o[3]) for o in st["others"]]
            bombs = [(tuple(b[0]), b[1]) for b in st["bombs"]]
            out(f"    step={st['step']:>4} pos={pos} action={x['action']} others={others} bombs={bombs} "
                f"bomb_available={bool(st['self'][2])} in_danger={bool(x['features'][18])}")
        # Extra instrumented line at the death step for mask/escape detail.
        inst = _instrument_step(restore_state(tail[-1]["state"]))
        out(f"    [instrumented last recorded step] legal_actions={inst['legal_actions']} "
            f"n_conflicting={inst['n_conflicting']} current_tile_in_danger={inst['current_tile_in_danger']}")
        out("  classification: " + (
            "matches known c-class (single-opponent siege: bomb placed, agent's own blast/opponent "
            "converge to progressively remove escape directions until only WAIT remains) -- see "
            "the Stage D self-kill census's c-class definition."
            if label == "c_new_single_opponent_siege" else
            "does not match c2 (no contested-tile INVALID_ACTION), c1 (no later-leg conflict "
            "detected), or c-class siege pattern (_siege_or_c1's own_not_robust check found no "
            "matching step) -- genuinely unclassified by the reused Stage D taxonomy; would need "
            "case-by-case manual review beyond this script's automated checks to name a new pattern, "
            "n=1, not attempting to invent a new category label from a single case."
        ))


def item2():
    out("\n\n=== Item 2: (15,1) repeated-death-coordinate check + full census ===")
    out("New definition (no Stage D precedent for a coordinate-repeat census script): death coordinate "
        "= death['pos'] as recorded (self-kill or got_killed alike); census counts exact (x,y) repeats "
        "across all 300 breaker_only rounds.")
    all_deaths = []  # (seed, round, pos, cause, step, actions_tail)
    for s in SEEDS:
        payload = load(s)
        for r in payload["records"]:
            m = r["metrics"]
            if not (m["self_kill"] or m["got_killed_by_opponent"]):
                continue
            death = r["death"]
            pos = tuple(int(v) for v in death["pos"]) if death else None
            cause = "self_kill" if m["self_kill"] else "got_killed"
            all_deaths.append({
                "seed": s, "round": r["round"], "pos": pos, "cause": cause,
                "step": death["step"] if death else None,
                "actions_tail": m["actions"][-8:],
            })

    seed0_15_1 = [d for d in all_deaths if d["seed"] == 0 and d["pos"] == (15, 1)]
    out(f"\nseed0 (15,1) cases: n={len(seed0_15_1)}")
    out(f"is (15,1) a spawn corner? {(15, 1) in SPAWN_CORNERS} (classic scenario's 4 fixed corners: {sorted(SPAWN_CORNERS)})")
    for d in seed0_15_1:
        early = d["step"] is not None and d["step"] <= 30
        out(f"  round={d['round']:>3} cause={d['cause']} death_step={d['step']} early_death(<=30 steps)={early} "
            f"last_actions={d['actions_tail']}")

    out("\n--- full census across all 300 breaker_only rounds' deaths ---")
    n_total_deaths = len(all_deaths)
    pos_counts = Counter(d["pos"] for d in all_deaths if d["pos"] is not None)
    out(f"total deaths with a recorded position: {sum(pos_counts.values())}/{n_total_deaths}")
    repeats = {p: n for p, n in pos_counts.items() if n >= 3}
    out(f"coordinates appearing in >=3 death events (n={len(repeats)} distinct coordinates):")
    for p, n in sorted(repeats.items(), key=lambda kv: -kv[1]):
        cases = [d for d in all_deaths if d["pos"] == p]
        seeds_involved = sorted({d["seed"] for d in cases})
        causes = Counter(d["cause"] for d in cases)
        out(f"  {p}: n={n} seeds={seeds_involved} causes={dict(causes)} is_spawn_corner={p in SPAWN_CORNERS}")
    if not repeats:
        out("  none (n=0) -- data insufficient to characterize a repeat-coordinate pattern beyond the "
            "single (15,1)/seed0 case already found.")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation2_case_review.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item1()
    item2()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
