"""Task 4 offline investigation: path-level counterfactual replay of two
candidate reroute rules on recorded data. Read-only on all
task4_seed*_*.pkl.gz files; breaker_only is the main analysis (3a-3d),
deadlock/no_breaker are a cross-check (3e).

Only data recorded for each step is used; outcomes after a different action
are not simulated.

Definitions:
- post-bomb danger window for an own bomb at bomb_idx: recorded steps
  bomb_idx+1 .. bomb_idx+SAFETY_HORIZON, stopping at the first step without
  current danger or at round end.
- legal escape first-step: a movement direction legal in the recorded
  final_mask with a path from _candidate_escape_paths() under that step's
  real danger_offsets.
- conflict-free / has-conflict: whether _path_conflicts() is empty for the
  whole candidate path.
- R1: if both conflict-free and has-conflict first-steps exist, drop the
  has-conflict ones.
- R2: drop first-steps whose margin (FUSE_BUDGET - steps_needed) is below
  the best margin at that step.
- R1+R2: R1, then R2 on the survivors.
- Counterfactual choice: argmax of the recorded probs_final over the
  original final_mask minus the dropped directions (no renormalization).
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts.analyze_stage_d_final import _classify_selfkill
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.scripts.investigate_task4_c2_precursor import _gather_metrics_for_bomb
from agent_code.rhine.state_processing import (
    SAFETY_HORIZON, _candidate_escape_paths, _opponent_distance_maps, _path_conflicts,
    bfs_distances, extract_semantic_state,
)

SEEDS = (0, 1, 2)
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None
EXCLUDED_15_1 = {(0, r) for r in (21, 26, 27, 32, 36, 60, 74, 75, 87)}
FUSE_BUDGET = cfg.BOMB_TIMER + 1


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


def _real_candidate_paths(sem):
    return _candidate_escape_paths(sem.self_pos, sem.field_arr, sem.blocked, sem.danger_offsets, start_offset=0)


def _classify_step(sem, x):
    """{direction: {"path", "margin", "conflict"}} for the legal escape
    first-steps at this step; {} if none qualify."""
    candidate_paths = _real_candidate_paths(sem)
    final_legal_dirs = {cfg.ACTIONS[i] for i, m in enumerate(x["final_mask"]) if m}
    legal_dirs = [d for d in candidate_paths if d in final_legal_dirs]
    if not legal_dirs:
        return {}
    opp_dist_maps = _opponent_distance_maps(sem.field_arr, sem.blocked, sem.opponents)
    info = {}
    for d in legal_dirs:
        path = candidate_paths[d]
        conflicts = _path_conflicts(path, opp_dist_maps)
        margin = FUSE_BUDGET - path[-1][1]
        info[d] = {"path": path, "margin": margin, "conflict": bool(conflicts)}
    return info


def _apply_r1(info):
    conflict_free = [d for d, v in info.items() if not v["conflict"]]
    has_conflict = [d for d, v in info.items() if v["conflict"]]
    if conflict_free and has_conflict:
        return set(conflict_free), True
    return set(info.keys()), False


def _apply_r2(dirs, info):
    if len(dirs) <= 1:
        return set(dirs), False
    best = max(info[d]["margin"] for d in dirs)
    survivors = {d for d in dirs if info[d]["margin"] >= best}
    return survivors, len(survivors) < len(dirs)


def _rule_survivors(rule, info):
    if rule == "R1":
        survivors, intervened = _apply_r1(info)
    elif rule == "R2":
        survivors, intervened = _apply_r2(set(info.keys()), info)
    else:  # R1+R2
        after_r1, r1_did = _apply_r1(info)
        survivors, r2_did = _apply_r2(after_r1, info)
        intervened = r1_did or r2_did
    return survivors, intervened


def _counterfactual_action(x, removed_dirs):
    if not removed_dirs:
        return x["action"], False
    new_mask = np.array(x["final_mask"], dtype=bool).copy()
    for d in removed_dirs:
        new_mask[cfg.ACTIONS.index(d)] = False
    if not new_mask.any():
        return x["action"], False
    probs = np.asarray(x["probs_final"], dtype=np.float64)
    masked_probs = np.where(new_mask, probs, -np.inf)
    new_action = cfg.ACTIONS[int(np.argmax(masked_probs))]
    return new_action, new_action != x["action"]


def _danger_window(recs, bomb_idx):
    steps = []
    i = bomb_idx + 1
    while i < len(recs) and i <= bomb_idx + SAFETY_HORIZON:
        sem = extract_semantic_state(restore_state(recs[i]["state"]))
        if not sem.current_tile_in_danger:
            break
        steps.append((i, sem))
        i += 1
    return steps


def _gather_all_bombs(label, suffix):
    """One dict per own BOMB placement with its danger-window step evaluations."""
    bombs = []
    for s in SEEDS:
        payload = load(s, label, suffix)
        for r in payload["records"]:
            recs = r["steps"]
            lethal_idx = None
            if r["metrics"]["self_kill"]:
                _, lethal_idx, _ = _classify_selfkill(r)
            is_15_1 = (s, r["round"]) in EXCLUDED_15_1
            bomb_indices = [i for i, x in enumerate(recs) if x["action"] == "BOMB"]
            for i in bomb_indices:
                window = _danger_window(recs, i)
                step_evals = []
                for idx, sem in window:
                    info = _classify_step(sem, recs[idx])
                    if not info:
                        continue
                    step_evals.append((idx, info, recs[idx]))
                nearest_opp_dist = _gather_metrics_for_bomb(recs[i]["state"])["f29_raw"]
                bombs.append({
                    "seed": s, "round": r["round"], "bomb_idx": i, "is_lethal": i == lethal_idx,
                    "is_15_1": is_15_1, "step_evals": step_evals, "nearest_opp_dist_at_bomb": nearest_opp_dist,
                })
    return bombs


def item3a(bombs, exclude_15_1):
    label = "EXCLUDING (15,1)" if exclude_15_1 else "INCLUDING (15,1)"
    subset = [b for b in bombs if not (exclude_15_1 and b["is_15_1"])]
    lethal_bombs = [b for b in subset if b["is_lethal"]]
    out(f"\n--- Item 3a: fatal bombs, {label} (n={len(lethal_bombs)}) ---")
    if len(lethal_bombs) < 30:
        out(f"  NOTE: n={len(lethal_bombs)} <30, small sample, for reference only.")
    for rule in ("R1", "R2", "R1+R2"):
        n_removes_actual = 0
        n_decision_changed_bombs = 0
        for b in lethal_bombs:
            removed_actual_any = False
            changed_any = False
            for idx, info, x in b["step_evals"]:
                survivors, intervened = _rule_survivors(rule, info)
                removed = set(info.keys()) - survivors
                if x["action"] in removed:
                    removed_actual_any = True
                _, changed = _counterfactual_action(x, removed)
                if changed:
                    changed_any = True
            if removed_actual_any:
                n_removes_actual += 1
            if changed_any:
                n_decision_changed_bombs += 1
        out(f"  rule={rule}: fatal bombs where the rule removes the actually-chosen action at >=1 step: "
            f"{n_removes_actual}/{len(lethal_bombs)}; fatal bombs with >=1 step where the counterfactual "
            f"decision changes: {n_decision_changed_bombs}/{len(lethal_bombs)}")


def item3b(bombs):
    out("\n--- Item 3b: cost across all 8072 placements' post-bomb danger steps ---")
    total_steps = sum(len(b["step_evals"]) for b in bombs)
    out(f"  total qualifying danger steps across {len(bombs)} placements: {total_steps} "
        f"(mean per placement: {total_steps / len(bombs):.3f})")
    for rule in ("R1", "R2", "R1+R2"):
        n_removed_actions_total = 0
        n_steps_changed = 0
        n_no_intervene = 0
        bombs_with_change = 0
        bombs_with_change_lethal = 0
        for b in bombs:
            bomb_changed = False
            for idx, info, x in b["step_evals"]:
                survivors, intervened = _rule_survivors(rule, info)
                removed = set(info.keys()) - survivors
                n_removed_actions_total += len(removed)
                if not intervened:
                    n_no_intervene += 1
                _, changed = _counterfactual_action(x, removed)
                if changed:
                    n_steps_changed += 1
                    bomb_changed = True
            if bomb_changed:
                bombs_with_change += 1
                if b["is_lethal"]:
                    bombs_with_change_lethal += 1
        out(f"  rule={rule}: mean legal actions removed per placement={n_removed_actions_total / len(bombs):.3f} "
            f"(over {total_steps} steps); steps with decision change={n_steps_changed}; "
            f"placements with >=1 changed step={bombs_with_change} (of which lethal={bombs_with_change_lethal}, "
            f"non-lethal={bombs_with_change - bombs_with_change_lethal}); "
            f"no-alternative non-intervention rate={100 * n_no_intervene / total_steps:.2f}% ({n_no_intervene}/{total_steps})")


def item3c(bombs):
    out("\n--- Item 3c: intervention precision ---")
    n_total = len(bombs)
    n_lethal_total = sum(1 for b in bombs if b["is_lethal"])
    baseline_rate = 100 * n_lethal_total / n_total
    out(f"  baseline lethality rate over all placements: {n_lethal_total}/{n_total} = {baseline_rate:.3f}%")
    for rule in ("R1", "R2", "R1+R2"):
        changed_bombs = []
        for b in bombs:
            changed = False
            for idx, info, x in b["step_evals"]:
                survivors, _ = _rule_survivors(rule, info)
                removed = set(info.keys()) - survivors
                _, ch = _counterfactual_action(x, removed)
                if ch:
                    changed = True
                    break
            if changed:
                changed_bombs.append(b)
        n_changed = len(changed_bombs)
        n_changed_lethal = sum(1 for b in changed_bombs if b["is_lethal"])
        rate = 100 * n_changed_lethal / n_changed if n_changed else float("nan")
        out(f"  rule={rule}: decision-changed placements={n_changed}, of which lethal={n_changed_lethal} "
            f"({rate:.3f}% vs baseline {baseline_rate:.3f}%)" + ("  NOTE: n<30, small sample." if n_changed < 30 else ""))


def _path_intersects_opponent_shortest_path(path, opp, field_arr, blocked):
    opp_dist = bfs_distances(field_arr, opp, blocked=blocked)
    path_tiles = {tile for tile, _ in path}
    d0 = opp_dist.get(path[0][0], float("inf"))
    on_shortest_path = {t for t in path_tiles if opp_dist.get(t, float("inf")) <= d0}
    return bool(on_shortest_path), on_shortest_path


def item3d(bombs, exclude_15_1):
    label = "EXCLUDING (15,1)" if exclude_15_1 else "INCLUDING (15,1)"
    subset = [b for b in bombs if not (exclude_15_1 and b["is_15_1"])]
    lethal5 = [b for b in subset if b["is_lethal"] and b["nearest_opp_dist_at_bomb"] == 5]
    out(f"\n--- Item 3d: fatal bombs with nearest-opponent distance==5 at placement, {label} (n={len(lethal5)}) ---")
    if len(lethal5) < 30:
        out(f"  NOTE: n={len(lethal5)} <30, small sample, for reference only.")
    for b in lethal5:
        out(f"\n  seed{b['seed']} round={b['round']} bomb_idx={b['bomb_idx']}")
        for idx, info, x in b["step_evals"]:
            sem = extract_semantic_state(restore_state(x["state"]))
            chosen = x["action"]
            chosen_info = info.get(chosen)
            heads_toward = None
            if chosen_info and sem.opponents:
                nearest_opp = min(sem.opponents, key=lambda o: bfs_distances(sem.field_arr, o, blocked=sem.blocked).get(sem.self_pos, 999))
                on_path, _ = _path_intersects_opponent_shortest_path(chosen_info["path"], nearest_opp, sem.field_arr, sem.blocked)
                heads_toward = on_path
            conflict_free_alt = any(not v["conflict"] for d, v in info.items() if d != chosen)
            r1_surv, _ = _rule_survivors("R1", info)
            r2_surv, _ = _rule_survivors("R2", info)
            r1_changed = chosen not in r1_surv
            r2_changed = chosen not in r2_surv
            out(f"    step={x['state']['step']} chosen={chosen} route_heads_toward_nearest_opp={heads_toward} "
                f"conflict_free_alternative_exists={conflict_free_alt} "
                f"R1_would_remove_chosen={r1_changed} R2_would_remove_chosen={r2_changed}")


def item3e():
    out("\n--- Item 3e: same as item 3a, on deadlock and no_breaker groups' fatal bombs (not merged into main data) ---")
    for label, suffix in (("deadlock", "A"), ("no_breaker", "B")):
        bombs = _gather_all_bombs(label, suffix)
        lethal = [b for b in bombs if b["is_lethal"]]
        out(f"\n  group={label} fatal bombs n={len(lethal)}")
        if len(lethal) < 30:
            out(f"    NOTE: n={len(lethal)} <30, small sample, for reference only.")
        for rule in ("R1", "R2", "R1+R2"):
            n_removes_actual = 0
            n_decision_changed = 0
            for b in lethal:
                removed_any = False
                changed_any = False
                for idx, info, x in b["step_evals"]:
                    survivors, _ = _rule_survivors(rule, info)
                    removed = set(info.keys()) - survivors
                    if x["action"] in removed:
                        removed_any = True
                    _, ch = _counterfactual_action(x, removed)
                    if ch:
                        changed_any = True
                if removed_any:
                    n_removes_actual += 1
                if changed_any:
                    n_decision_changed += 1
            out(f"    rule={rule}: removes actually-chosen action in >=1 step: {n_removes_actual}/{len(lethal)}; "
                f"decision changes in >=1 step: {n_decision_changed}/{len(lethal)}")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA_DIR / "investigation4_reroute_counterfactual.txt"))
    args = parser.parse_args()
    _out = open(args.out, "w")

    out("=== Item 3: path-level counterfactual reroute replay (main data: breaker_only, 3x100 rounds) ===")
    bombs = _gather_all_bombs("breaker_only", "A")
    out(f"total own bomb placements: {len(bombs)}")

    item3a(bombs, exclude_15_1=False)
    item3a(bombs, exclude_15_1=True)
    item3b(bombs)
    item3c(bombs)
    item3d(bombs, exclude_15_1=False)
    item3d(bombs, exclude_15_1=True)
    item3e()

    _out.close()
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
