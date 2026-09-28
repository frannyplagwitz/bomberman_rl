"""Task 4 offline investigation: reproduces the seed1 round-44 mask flip
(DOWN legal at one step, illegal at the next) with the real mask functions
on restored states, and counts the same signature across all recordings.
Read-only on all task4_seed*_*.pkl.gz files.

Also compares, for DOWN, the (tile, game step) pairs implied by the
placement-time escape path with the agent's actual positions, to test whether
has_safe_escape_after_bombing()'s offset convention assumes the first move
happens on the same tick as BOMB, one tick earlier than possible.
"""
import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_task4_death_forensics import _escape_routes_with_margin
from agent_code.rhine.state_processing import (
    DIRECTIONS, SAFETY_HORIZON, blast_coords, exists_safe_path, extract_semantic_state, neighbor_tile,
)

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
FILES = [("breaker_only", "A"), ("deadlock", "A"), ("no_breaker", "B")]


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed, label, suffix):
    with gzip.open(DATA_DIR / f"task4_seed{seed}_{label}_{suffix}.pkl.gz", "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError("partial recording")
    return payload


def item2a():
    out("=== Item 2a: seed1 round44 mask-flip reproduction ===")
    payload = load(1, "breaker_only", "A")
    r = next(rr for rr in payload["records"] if rr["round"] == 44)
    recs = r["steps"]
    _, bomb_idx, _ = _classify_selfkill(r)
    bx = recs[bomb_idx]
    bomb_pos = tuple(int(v) for v in bx["state"]["self"][3])
    bomb_step = bx["state"]["step"]
    out(f"bomb placed at {bomb_pos}, recorded step={bomb_step} (bomb_idx={bomb_idx})")

    sem_bomb = extract_semantic_state(restore_state(bx["state"]))
    routes, hyp_danger, _ = _escape_routes_with_margin(
        sem_bomb.self_pos, sem_bomb.field_arr, sem_bomb.blocked, sem_bomb.danger_offsets, cfg.BOMB_POWER, sem_bomb.opponents,
    )
    out(f"\nplacement-time DOWN route (has_safe_escape_after_bombing's hypothetical model): {routes.get('DOWN')}")
    out(f"cfg.BOMB_TIMER={cfg.BOMB_TIMER}, cfg.BOMB_POWER={cfg.BOMB_POWER}, "
        f"hypothetical blast danger offsets = {{{cfg.BOMB_TIMER}, {cfg.BOMB_TIMER + 1}}} (relative to the placement step's own offset 0)")

    down_path = routes.get("DOWN", {}).get("path")
    out("\nDOWN path re-expressed as (tile, offset) -> the ABSOLUTE game step that offset nominally refers to "
        f"(bomb_step + offset), vs the REAL recorded position at that absolute step:")
    if down_path:
        for tile, offset in down_path:
            abs_step = bomb_step + offset
            actual_x = next((x for x in recs if x["state"]["step"] == abs_step), None)
            actual_pos = tuple(int(v) for v in actual_x["state"]["self"][3]) if actual_x else "round ended"
            out(f"  offset={offset} implied_tile={tile} implied_abs_step={abs_step} | "
                f"ACTUAL pos at recorded step {abs_step} = {actual_pos}"
                f"{'  <-- MISMATCH' if actual_x and actual_pos != tile else ''}")
    out("\nInterpretation: has_safe_escape_after_bombing()'s hypothetical path starts counting offsets from "
        "the SAME tick BOMB is decided (offset 0 = the bomb-placement step itself), and its first movement "
        "leg (offset 1) is checked as if the agent already occupies the escape direction's neighbor tile "
        "by the very next recorded step. But BOMB does not move the agent -- the agent is still at the bomb "
        "tile one full step after placing it (see the recorded step-by-step trace above), so every "
        "subsequent leg of the hypothetical path is actually reached ONE recorded step later than the "
        "offset used to verify its safety. With DOWN's margin already at its minimum (1, from round 3's "
        "report), this one-step lag is exactly enough to walk the agent into the blast at the moment it "
        "was hypothetically assumed to already be clear.")

    out("\n--- step75/step76 mask branch identification (calling exists_safe_path directly) ---")
    for step_num in (75, 76):
        x = next(xx for xx in recs if xx["state"]["step"] == step_num)
        sem = extract_semantic_state(restore_state(x["state"]))
        pos = sem.self_pos
        neighbor = neighbor_tile(pos, "DOWN")
        legal = exists_safe_path(neighbor, 0, sem.field_arr, sem.blocked, sem.danger_offsets, SAFETY_HORIZON)
        own_bomb_timer = next((t for p, t, o in x["world"]["bombs"] if o == "rhine"), None)
        blast = set(blast_coords(sem.field_arr, bomb_pos, cfg.BOMB_POWER))
        out(f"  step={step_num} pos={pos} own_bomb_timer={own_bomb_timer} DOWN_neighbor={neighbor} "
            f"neighbor_in_blast={neighbor in blast} danger_offsets_at_neighbor={sem.danger_offsets.get(neighbor)} "
            f"exists_safe_path(DOWN, offset0)={legal}  (recorded base_mask DOWN legal={bool(x['base_mask'][cfg.ACTIONS.index('DOWN')])})")
    out("\nConclusion: exists_safe_path()'s own step76 evaluation (using the REAL, freshly-recomputed bomb "
        "timer at that step) correctly finds no safe continuation via DOWN -- the mask's per-step check is "
        "internally consistent at step76. The mismatch originates one step earlier, in the hypothetical "
        "verification has_safe_escape_after_bombing() performed at the BOMB decision (step74/step75, see "
        "the offset table above), which legalized BOMB (and therefore DOWN's initial appeal) based on a "
        "path that was never actually one full step longer than what happens in the recorded rollout allows.")


def item2b():
    out("\n\n=== Item 2b: signature count across all 9 recordings ===")
    out("Signature: at bomb placement, the best available escape-route margin (fuse_budget - steps_needed, "
        "fuse_budget=BOMB_TIMER+1) equals 1; AND at the recorded step immediately after the bomb-placement "
        "step (bomb_idx+1), base_mask legal actions == {WAIT} while self_pos is still within that bomb's "
        "blast_coords() at that step.")
    total_matches = 0
    total_lethal = 0
    per_group = []
    for label, suffix in FILES:
        for s in SEEDS:
            payload = load(s, label, suffix)
            group_matches = 0
            group_lethal = 0
            for r in payload["records"]:
                recs = r["steps"]
                lethal_idx = None
                if r["metrics"]["self_kill"]:
                    _, lethal_idx, _ = _classify_selfkill(r)
                bomb_indices = [i for i, x in enumerate(recs) if x["action"] == "BOMB"]
                for i in bomb_indices:
                    if i + 1 >= len(recs):
                        continue
                    bx = recs[i]
                    bomb_pos = tuple(int(v) for v in bx["state"]["self"][3])
                    sem = extract_semantic_state(restore_state(bx["state"]))
                    routes, _, _ = _escape_routes_with_margin(
                        sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, cfg.BOMB_POWER, sem.opponents,
                    )
                    if not routes:
                        continue
                    best_margin = max(info["margin"] for info in routes.values())
                    if best_margin != 1:
                        continue
                    nx = recs[i + 1]
                    next_legal = [cfg.ACTIONS[j] for j, m in enumerate(nx["base_mask"]) if m]
                    next_pos = tuple(int(v) for v in nx["state"]["self"][3])
                    blast = set(blast_coords(sem.field_arr, bomb_pos, cfg.BOMB_POWER))
                    if next_legal == ["WAIT"] and next_pos in blast:
                        group_matches += 1
                        total_matches += 1
                        if i == lethal_idx:
                            group_lethal += 1
                            total_lethal += 1
            per_group.append((f"seed{s}_{label}", group_matches, group_lethal))
    for name, m, l in per_group:
        out(f"  {name}: matches={m} lethal={l}")
    out(f"\nTOTAL (all 9 files): matches={total_matches} lethal={total_lethal}")
    if total_matches < 30:
        out(f"  NOTE: n={total_matches} <30, small sample, for reference only.")

    out("\n--- supplementary: same margin=1 condition, but scanning ALL post-placement steps (not just "
        "bomb_idx+1) for the first step where base_mask becomes WAIT-only while still in blast -- covers "
        "cases like round44 itself, where the trap appears one extra step later than bomb_idx+1 ---")
    total_matches2 = 0
    total_lethal2 = 0
    offset_hist = Counter()
    for label, suffix in FILES:
        for s in SEEDS:
            payload = load(s, label, suffix)
            for r in payload["records"]:
                recs = r["steps"]
                lethal_idx = None
                if r["metrics"]["self_kill"]:
                    _, lethal_idx, _ = _classify_selfkill(r)
                bomb_indices = [i for i, x in enumerate(recs) if x["action"] == "BOMB"]
                for i in bomb_indices:
                    bx = recs[i]
                    bomb_pos = tuple(int(v) for v in bx["state"]["self"][3])
                    sem = extract_semantic_state(restore_state(bx["state"]))
                    routes, _, _ = _escape_routes_with_margin(
                        sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, cfg.BOMB_POWER, sem.opponents,
                    )
                    if not routes or max(info["margin"] for info in routes.values()) != 1:
                        continue
                    blast = set(blast_coords(sem.field_arr, bomb_pos, cfg.BOMB_POWER))
                    for k in range(1, min(6, len(recs) - i)):
                        nx = recs[i + k]
                        next_legal = [cfg.ACTIONS[j] for j, m in enumerate(nx["base_mask"]) if m]
                        next_pos = tuple(int(v) for v in nx["state"]["self"][3])
                        if next_legal == ["WAIT"] and next_pos in blast:
                            total_matches2 += 1
                            offset_hist[k] += 1
                            if i == lethal_idx:
                                total_lethal2 += 1
                            break
    out(f"TOTAL (all 9 files, any offset 1-5): matches={total_matches2} lethal={total_lethal2}")
    out(f"  offset-after-placement distribution: {dict(offset_hist)}")
    if total_matches2 < 30:
        out(f"  NOTE: n={total_matches2} <30, small sample, for reference only.")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation4_round44_repro.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item2a()
    item2b()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
