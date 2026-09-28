"""Task 4 offline investigation: step-by-step forensic reconstruction of the
repeated seed0 self-kill cases at one fixed tile and the seed1 round-44
unclassified case. Read-only on the breaker_only/deadlock/no_breaker
recordings.

Classification reuses analyze_stage_d_final._classify_selfkill()/_siege_or_c1();
escape routes reuse state_processing's machinery.

Definitions used in this report:
- "move succeeded" at step i: the position at step i+1 moved in the chosen
  direction; otherwise the blocking cause is read from the recorded snapshot
  (same categories as classify_invalid_action()).
- "escape-route margin" for a candidate path of length L:
  (BOMB_TIMER + 1) - L; larger means more spare ticks.
- "actual route margin": (BOMB_TIMER + 1) - k, where k is the first offset at
  which the recorded path reaches a permanently safe tile; "never reached
  safety" if that does not happen before death.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill, _siege_or_c1
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import (
    DIRECTIONS, _candidate_escape_paths, _DIRECTION_OFFSETS, blast_coords, compute_danger_offsets, neighbor_tile,
)

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
SPAWN_CORNERS = {(1, 1), (1, 15), (15, 1), (15, 15)}


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


def _own_bomb(world_bombs):
    for pos, timer, owner in world_bombs:
        if owner == "rhine":
            return (tuple(pos), timer)
    return None


def _move_failure_cause(step_x, next_x, action):
    """Forensic blocking cause for a directional action that did not move the
    agent (same categories as classify_invalid_action())."""
    pos = tuple(int(v) for v in step_x["state"]["self"][3])
    dx, dy = _DIRECTION_OFFSETS[action]
    target = (pos[0] + dx, pos[1] + dy)
    new_pos = tuple(int(v) for v in next_x["state"]["self"][3])
    if new_pos != pos:
        return None
    field = restore_state(step_x["state"])["field"]
    if field[target[0], target[1]] == -1:
        return "wall"
    if field[target[0], target[1]] == 1:
        return "crate"
    for bpos, timer, owner in step_x["world"]["bombs"]:
        if tuple(bpos) == target:
            return f"bomb(owner={owner},timer={timer})"
    for o in step_x["state"]["others"]:
        if tuple(int(v) for v in o[3]) == target:
            return "opponent"
    for owner, coords, timer in step_x["world"]["expl"]:
        if target in {tuple(c) for c in coords}:
            return f"explosion(owner={owner})"
    return "contested_tile(target was free -- same-tick collision)"


def _print_last_n_steps(recs, death, n=10):
    tail = recs[-n:]
    for i, x in enumerate(tail):
        st = x["state"]
        pos = tuple(int(v) for v in st["self"][3])
        legal = [cfg.ACTIONS[i2] for i2, m in enumerate(x["final_mask"]) if m]
        own_bomb = _own_bomb(x["world"]["bombs"])
        others = [tuple(int(v) for v in o[3]) for o in st["others"]]
        cause = ""
        if i < len(tail) - 1 and x["action"] in DIRECTIONS:
            c = _move_failure_cause(x, tail[i + 1], x["action"])
            if c:
                cause = f"  MOVE_FAILED cause={c}"
        elif i == len(tail) - 1 and x["action"] in DIRECTIONS and death is not None:
            # Last recorded step: compare against the death position.
            new_pos = tuple(death["pos"])
            if new_pos == pos:
                dx, dy = _DIRECTION_OFFSETS[x["action"]]
                target = (pos[0] + dx, pos[1] + dy)
                field = restore_state(st)["field"]
                if field[target[0], target[1]] not in (-1, 1) and target not in {
                    tuple(b[0]) for b in x["world"]["bombs"]
                } and target not in {tuple(int(v) for v in o[3]) for o in st["others"]}:
                    cause = "  MOVE_FAILED cause=contested_tile(target was free -- same-tick collision, agent died here)"
                else:
                    field_val = field[target[0], target[1]]
                    cause = f"  MOVE_FAILED cause={'wall' if field_val == -1 else 'crate' if field_val == 1 else 'occupied'}"
        out(f"    step={st['step']:>4} pos={pos} action={x['action']:>5} legal={legal} "
            f"own_bomb={own_bomb} others={others}{cause}")


def _escape_routes_with_margin(self_pos, field_arr, blocked, danger_offsets, power, opponents):
    hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
    for blast_tile in blast_coords(field_arr, self_pos, power):
        hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
    blocked_with_new_bomb = blocked | {self_pos}
    paths = _candidate_escape_paths(self_pos, field_arr, blocked_with_new_bomb, hypothetical_danger, start_offset=0)
    fuse_budget = cfg.BOMB_TIMER + 1
    routes = {}
    for d, path in paths.items():
        L = path[-1][1]  # Steps needed to reach the verified-safe tile.
        routes[d] = {"path": path, "steps_needed": L, "margin": fuse_budget - L}
    return routes, hypothetical_danger, blocked_with_new_bomb


def _actual_route_margin(recs, bomb_idx, hypothetical_danger):
    fuse_budget = cfg.BOMB_TIMER + 1
    for k in range(0, len(recs) - bomb_idx):
        idx = bomb_idx + k
        if idx >= len(recs):
            break
        pos = tuple(int(v) for v in recs[idx]["state"]["self"][3])
        remaining_danger = {o for o in hypothetical_danger.get(pos, ()) if o > k}
        if not remaining_danger:
            return k, fuse_budget - k
    return None, None


def item1():
    out("=== Item 1: (15,1) nine self-kill cases, full forensic detail ===")
    payload = load(0, "breaker_only", "A")
    target_rounds = [21, 26, 27, 32, 36, 60, 74, 75, 87]
    cases = {r["round"]: r for r in payload["records"] if r["round"] in target_rounds}
    out(f"n={len(cases)} (expect 9)")

    for rnd in target_rounds:
        r = cases[rnd]
        recs, death = r["steps"], r["death"]
        out(f"\n{'=' * 10} round={rnd} death_step={death['step']} death_pos={tuple(death['pos'])} {'=' * 10}")
        out("  --- 1a: last 10 steps ---")
        _print_last_n_steps(recs, death, n=10)

        label, bomb_idx, flags = _classify_selfkill(r)
        if label == "unclassified" and bomb_idx is not None:
            findings = _siege_or_c1(r, bomb_idx)
            siege = [f for f in findings if f[2] == 1 and f[3] >= 2 and f[4]]
            conflict_later = [f for f in findings if f[2] >= 1]
            if siege:
                label = "c_new_single_opponent_siege"
            elif conflict_later:
                label = "c1_later_leg_conflict"

        out(f"\n  --- 1b: bomb placement step (bomb_idx={bomb_idx}) ---")
        bx = recs[bomb_idx]
        bomb_pos = tuple(int(v) for v in bx["state"]["self"][3])
        gs = restore_state(bx["state"])
        from agent_code.rhine.state_processing import extract_semantic_state
        sem = extract_semantic_state(gs)
        routes, hyp_danger, _ = _escape_routes_with_margin(
            sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, cfg.BOMB_POWER, sem.opponents,
        )
        out(f"    bomb placed at {bomb_pos}, step={bx['state']['step']}")
        out(f"    escape routes available: {len(routes)}")
        for d, info in routes.items():
            out(f"      dir={d}: steps_needed={info['steps_needed']} margin(fuse-steps)={info['margin']}")
        actual_k, actual_margin = _actual_route_margin(recs, bomb_idx, hyp_danger)
        if actual_k is None:
            out(f"    ACTUAL route taken: never reached a permanently-safe tile before death "
                f"(died {len(recs) - 1 - bomb_idx} steps after placement)")
        else:
            out(f"    ACTUAL route taken: reached safety after {actual_k} steps, margin={actual_margin}")

        out(f"\n  --- 1c: final-mask check at death step ---")
        last = recs[-1]
        final_mask_legal = [cfg.ACTIONS[i] for i, m in enumerate(last["final_mask"]) if m]
        chosen = last["action"]
        out(f"    step={last['state']['step']} chosen_action={chosen} final_mask_legal={final_mask_legal} "
            f"non_lethal_alternatives_available={len(final_mask_legal) > 1 or (len(final_mask_legal) == 1 and final_mask_legal[0] != chosen)}")

        out(f"\n  --- 1d: Stage D classification ---")
        out(f"    label={label}")
        out(f"    triggering flags: {flags}")

    out("\n--- 1e: cross-group (15,1) deaths for seed0, + spawn-corner census across all 9 files ---")
    for label, suffix in (("breaker_only", "A"), ("deadlock", "A"), ("no_breaker", "B")):
        p = load(0, label, suffix)
        matches = [r for r in p["records"] if r["death"] and tuple(r["death"]["pos"]) == (15, 1)]
        causes = [("self_kill" if r["metrics"]["self_kill"] else "got_killed") for r in matches]
        out(f"  seed0 {label}: (15,1) deaths = {len(matches)}, causes={causes}")

    out("\n  spawn-corner death census, all 9 files, exact-coordinate match:")
    for label, suffix in (("breaker_only", "A"), ("deadlock", "A"), ("no_breaker", "B")):
        for s in SEEDS:
            p = load(s, label, suffix)
            corner_counts = {c: 0 for c in SPAWN_CORNERS}
            for r in p["records"]:
                if r["death"] and tuple(r["death"]["pos"]) in SPAWN_CORNERS:
                    corner_counts[tuple(r["death"]["pos"])] += 1
            nonzero = {c: n for c, n in corner_counts.items() if n > 0}
            if nonzero:
                out(f"    seed{s} {label}: {nonzero}")

    out("\n--- 1f: seed0 breaker_only, first-5-step action distribution when spawning at (15,1) ---")
    from collections import Counter
    spawn_rounds = 0
    action_by_position = [Counter() for _ in range(5)]
    for r in payload["records"]:
        recs = r["steps"]
        if not recs:
            continue
        first_pos = tuple(int(v) for v in recs[0]["state"]["self"][3])
        if first_pos != (15, 1):
            continue
        spawn_rounds += 1
        for i in range(min(5, len(recs))):
            action_by_position[i][recs[i]["action"]] += 1
    out(f"  rounds spawning at (15,1): {spawn_rounds}/100")
    if spawn_rounds == 0:
        out("  n=0 -- rhine never spawns at (15,1) in this recording (seat/corner assignment fixed elsewhere); "
            "data insufficient to report a first-5-step distribution for this specific spawn corner.")
    else:
        for i, counter in enumerate(action_by_position):
            out(f"    step_offset={i}: {dict(counter)}")


def item2():
    out("\n\n=== Item 2: seed1 round44 unclassified case, full forensic detail ===")
    payload = load(1, "breaker_only", "A")
    r = next(rr for rr in payload["records"] if rr["round"] == 44)
    recs, death = r["steps"], r["death"]
    step_by_num = {x["state"]["step"]: x for x in recs}

    out("\n--- 2a: step 74-78 detail ---")
    for step_num in range(74, 79):
        if step_num not in step_by_num:
            out(f"  step={step_num}: not recorded (round may have ended earlier in the tail window)")
            continue
        x = step_by_num[step_num]
        st = x["state"]
        pos = tuple(int(v) for v in st["self"][3])
        base_legal = [cfg.ACTIONS[i] for i, m in enumerate(x["base_mask"]) if m]
        final_legal = [cfg.ACTIONS[i] for i, m in enumerate(x["final_mask"]) if m]
        probs = {cfg.ACTIONS[i]: round(float(p), 3) for i, p in enumerate(x["probs_final"])}
        own_bomb = _own_bomb(x["world"]["bombs"])
        out(f"  step={step_num} pos={pos} action={x['action']} base_mask_legal={base_legal} "
            f"final_mask_legal={final_legal} probs_final={probs} own_bomb={own_bomb}")

    out("\n--- 2b: step 75 (bomb placement step, i.e. the step where BOMB is chosen) ---")
    _, bomb_idx, _ = _classify_selfkill(r)
    bx = recs[bomb_idx]
    bomb_pos = tuple(int(v) for v in bx["state"]["self"][3])
    field = restore_state(bx["state"])["field"]
    out(f"  bomb placed at {bomb_pos} on recorded step {bx['state']['step']} (bomb_idx={bomb_idx})")
    out("  4 neighbors of the bomb tile:")
    for d in DIRECTIONS:
        n = neighbor_tile(bomb_pos, d)
        val = field[n[0], n[1]]
        occ = "wall" if val == -1 else "crate" if val == 1 else "free"
        for bpos, timer, owner in bx["world"]["bombs"]:
            if tuple(bpos) == n:
                occ = f"bomb(owner={owner},timer={timer})"
        for o in bx["state"]["others"]:
            if tuple(int(v) for v in o[3]) == n:
                occ = "opponent"
        out(f"    {d} -> {n}: {occ}")
    from agent_code.rhine.state_processing import extract_semantic_state
    sem = extract_semantic_state(restore_state(bx["state"]))
    routes, hyp_danger, _ = _escape_routes_with_margin(
        sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, cfg.BOMB_POWER, sem.opponents,
    )
    out(f"  escape routes at placement: {len(routes)}")
    for d, info in routes.items():
        out(f"    dir={d}: steps_needed={info['steps_needed']} margin={info['margin']}")

    out("\n--- 2c: (5,6) pocket check ---")
    pos_5_6 = (5, 6)
    field_val = field[5, 6]
    out(f"  (5,6) field value: {field_val} ({'wall' if field_val == -1 else 'crate' if field_val == 1 else 'free'})")
    exits = []
    for d in DIRECTIONS:
        n = neighbor_tile(pos_5_6, d)
        v = field[n[0], n[1]]
        exits.append((d, n, "wall" if v == -1 else "crate" if v == 1 else "free"))
    out(f"  (5,6) neighbors (its own exits): {exits}")
    step75 = step_by_num.get(75)
    if step75:
        final_legal_75 = [cfg.ACTIONS[i] for i, m in enumerate(step75["final_mask"]) if m]
        out(f"  step 75 final_mask legal actions: {final_legal_75} (chosen: {step75['action']})")
        out(f"  non-DOWN alternatives available at step 75: {[a for a in final_legal_75 if a != 'DOWN']}")

    out("\n--- 2d: c1 criteria check ---")
    findings = _siege_or_c1(r, bomb_idx)
    conflict_later = [f for f in findings if f[2] >= 1]
    out(f"  _siege_or_c1 findings (step, legal_actions, n_conflicting, n_candidates, all_not_robust, action): {findings}")
    out(f"  c1 requires n_conflicting>=1 at some post-bomb step with a legal escape candidate: "
        f"{'MATCHES' if conflict_later else 'DOES NOT MATCH'} (n_conflicting is 0 at every post-bomb step "
        f"shown above -- no opponent's real BFS distance ever comes within range of a candidate escape "
        f"tile's offset, so the mask's N+1-path redundancy logic never even engages; the death instead "
        f"looks like the chosen tile (5,6) simply not being far enough outside the bomb's own blast "
        f"radius, an escape-distance shortfall rather than an opponent-conflict pattern)" if not conflict_later else "")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation3_death_forensics.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")
    item1()
    item2()
    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
