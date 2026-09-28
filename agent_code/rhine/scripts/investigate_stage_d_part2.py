"""Stage D part-2 read-only investigation, offline on eval_stage_d_final.py's
recordings (agent_code/rhine/logs/stage_d_final/). Sections are described in
each function's docstring. Reuses analyze_stage_d_final.py's
load()/restore_state() and the production state_processing functions
(patched only for the approximation in (a)).
Usage:
  python -m agent_code.rhine.scripts.investigate_stage_d_part2 <a|b|c|d|all>
"""
import argparse
import contextlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine import state_processing as sp
from agent_code.rhine.action_mask import mask_from_semantic
from agent_code.rhine.scripts import analyze_stage_d_final as A
from agent_code.rhine.scripts.diagnose_kill_target_discrimination import _bomb_wasteful_mismatch_check
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _instrument_step
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import blast_coords, crates_in_blast, extract_semantic_state

SEEDS = (0, 1, 2)
BOMB_IDX = cfg.ACTIONS.index("BOMB")

A.DATA_DIR = A.common.LOGS_DIR / "stage_d_final"


def out(msg=""):
    print(msg)


# =========================================================== (a) margin+1
@contextlib.contextmanager
def _margin_plus_one():
    """What-if approximation: replays the escape check with one extra tick of
    slack in the opponent conflict test (d <= offset + 1). Patches
    state_processing._path_conflicts module-wide and restores it on exit.
    """
    original = sp._path_conflicts

    def patched(path, opponent_dist_maps):
        conflicts = {}
        for tile, offset in path:
            hit = {opp for opp, dmap in opponent_dist_maps.items() if dmap.get(tile, float("inf")) <= offset + 1}
            if hit:
                conflicts[tile] = hit
        return conflicts

    sp._path_conflicts = patched
    try:
        yield
    finally:
        sp._path_conflicts = original


def _bomb_legal_now(semantic) -> bool:
    return sp.has_safe_escape_after_bombing(
        semantic.self_pos, semantic.field_arr, semantic.blocked, semantic.danger_offsets,
        cfg.BOMB_POWER, semantic.opponents,
    )


def section_a():
    out("=== (a) c2 self-kill: placement-moment conflict detail + margin+1 replay (approximate) ===")
    out("Only re-evaluates has_safe_escape_after_bombing() with the opponent-distance margin relaxed by "
        "+1 tick; does not simulate what the policy would have done instead once BOMB is blocked, so "
        "'avoided' below means only 'this exact self-kill mechanism could not have started this way' -- "
        "labelled approximate throughout.\n")

    c2_cases = []
    for s in SEEDS:
        payload = A.load(s, "A")
        for r in payload["records"]:
            if not r["metrics"]["self_kill"]:
                continue
            label, bomb_idx, flags = A._classify_selfkill(r)
            if label == "c2_contested_tile":
                c2_cases.append((s, r, bomb_idx, flags))
    out(f"c2 cases found: {len(c2_cases)} (expected 17 from the prior census)\n")

    avoided = 0
    for s, r, bomb_idx, flags in c2_cases:
        state = restore_state(r["steps"][bomb_idx]["state"])
        semantic = extract_semantic_state(state)
        pos = semantic.self_pos
        opp_maps = sp._opponent_distance_maps(semantic.field_arr, semantic.blocked, semantic.opponents)
        candidate_paths = sp._candidate_escape_paths(pos, semantic.field_arr, semantic.blocked, semantic.danger_offsets, 0)
        detail = []
        for d, path in candidate_paths.items():
            conflicts = sp._path_conflicts(path, opp_maps)
            detail.append((d, len(path), {opp: dist for opp, dist in [
                (o, opp_maps[o].get(path[-1][0], None)) for o in semantic.opponents
            ]}, conflicts))
        with _margin_plus_one():
            legal_margin1 = _bomb_legal_now(semantic)
        was_avoided = not legal_margin1
        avoided += was_avoided
        out(f"seed{s} round={r['round']} bomb_step={state['step']} pos={pos} "
            f"opponents={semantic.opponents} n_candidate_paths={len(candidate_paths)} "
            f"margin+1_would_block={was_avoided}")
        for d, plen, dists, conflicts in detail:
            out(f"    {d}: path_len={plen} opp_dist_to_path_end={dists} conflicts(margin0)={conflicts}")
    out(f"\nOf {len(c2_cases)} c2 self-kill cases, margin+1 would have blocked the placement in "
        f"{avoided}/{len(c2_cases)} (approximate -- see caveat above).\n")

    out("--- pooled cost side: replaying margin+1 over every BOMB action actually taken in group A ---")
    total_bombs, newly_blocked, newly_blocked_was_opportunity = 0, 0, 0
    for s in SEEDS:
        payload = A.load(s, "A")
        seed_bombs = seed_blocked = seed_opp = 0
        for r in payload["records"]:
            for x in r["steps"]:
                if x["action"] != "BOMB":
                    continue
                seed_bombs += 1
                state = restore_state(x["state"])
                semantic = extract_semantic_state(state)
                with _margin_plus_one():
                    legal_margin1 = _bomb_legal_now(semantic)
                if legal_margin1:
                    continue
                seed_blocked += 1
                blast = set(blast_coords(semantic.field_arr, semantic.self_pos, cfg.BOMB_POWER))
                if any(o in blast for o in semantic.opponents):
                    seed_opp += 1
        out(f"seed{s}: BOMB actions taken={seed_bombs} newly blocked under margin+1={seed_blocked} "
            f"({100 * seed_blocked / seed_bombs:.2f}%), of those with an opponent in the blast (kill "
            f"opportunity)={seed_opp} ({100 * seed_opp / seed_blocked if seed_blocked else float('nan'):.1f}%)")
        total_bombs += seed_bombs
        newly_blocked += seed_blocked
        newly_blocked_was_opportunity += seed_opp
    out(f"\nPOOLED: {newly_blocked}/{total_bombs} BOMB placements ({100 * newly_blocked / total_bombs:.2f}%) "
        f"would newly be masked out under margin+1 ({newly_blocked / 3:.1f}/seed over 100 rounds); of those, "
        f"{newly_blocked_was_opportunity}/{newly_blocked} ({100 * newly_blocked_was_opportunity / newly_blocked if newly_blocked else float('nan'):.1f}%) "
        f"were kill opportunities that would be given up.")


# =========================================================== (b) threatened-then-BOMB
def section_b():
    out("=== (b) steps under threat where BOMB was chosen ===")
    out("'under threat' = current_tile_in_danger OR an opponent bomb whose blast covers self_pos right now.\n")
    total_threatened_bomb = 0
    total_with_safe_alt = 0
    death_within = {n: 0 for n in (1, 3, 5, 10)}
    per_case_flags = {"got_killed": [], "self_kill": []}
    for s in SEEDS:
        payload = A.load(s, "A")
        seed_n = seed_alt = 0
        for r in payload["records"]:
            recs = r["steps"]
            death_step = r["death"]["step"] if r["death"] else None
            for i, x in enumerate(recs):
                state = restore_state(x["state"])
                pos = tuple(int(v) for v in x["state"]["self"][3])
                opp_bombs_cover = any(
                    pos in set(blast_coords(state["field"], bp, cfg.BOMB_POWER))
                    for bp, timer, owner in x["world"]["bombs"] if owner != "rhine"
                )
                under_threat = bool(x["features"][18]) or opp_bombs_cover
                if not (under_threat and x["action"] == "BOMB"):
                    continue
                seed_n += 1
                legal = [j for j, m in enumerate(x["final_mask"]) if m and j != BOMB_IDX]
                has_safe_alt = len(legal) > 0
                seed_alt += has_safe_alt
                if death_step is not None and death_step >= x["state"]["step"]:
                    gap = death_step - x["state"]["step"]
                    for n in death_within:
                        if gap <= n:
                            death_within[n] += 1
                    if r["metrics"]["got_killed_by_opponent"]:
                        per_case_flags["got_killed"].append((s, r["round"]))
                    if r["metrics"]["self_kill"]:
                        per_case_flags["self_kill"].append((s, r["round"]))
        out(f"seed{s}: threatened-then-BOMB steps={seed_n}, with a safe non-BOMB alternative available="
            f"{seed_alt} ({100 * seed_alt / seed_n if seed_n else float('nan'):.1f}%)")
        total_threatened_bomb += seed_n
        total_with_safe_alt += seed_alt
    out(f"\nPOOLED: {total_threatened_bomb} threatened-then-BOMB steps "
        f"({total_threatened_bomb / 300:.2f}/round); safe non-BOMB alternative existed in "
        f"{total_with_safe_alt}/{total_threatened_bomb} ({100 * total_with_safe_alt / total_threatened_bomb:.1f}%)")
    for n in (1, 3, 5, 10):
        out(f"  died within {n} step(s) after such a step: {death_within[n]}/{total_threatened_bomb} "
            f"({100 * death_within[n] / total_threatened_bomb:.2f}%)")
    out(f"\ngot_killed cases (5 total) that contain >=1 such step: {len(set(per_case_flags['got_killed']))}/5")
    out(f"self-kill cases (19 total) that contain >=1 such step: {len(set(per_case_flags['self_kill']))}/19")


# =========================================================== (c) step-0 oscillation
def section_c():
    out("=== (c) group B step-0 oscillation cases: spawn-tile mask/probabilities ===")
    for s in SEEDS:
        payload = A.load(s, "B")
        cases = A._osc_cases(payload)
        starters = [c for c in cases if c["start"] == 0]
        for c in starters:
            r = next(rr for rr in payload["records"] if rr["round"] == c["round"])
            x = r["steps"][0]
            legal = [cfg.ACTIONS[i] for i, m in enumerate(x["base_mask"]) if m]
            out(f"\nseed{s} round={c['round']}: spawn pos={tuple(int(v) for v in x['state']['self'][3])} "
                f"opponents_alive={len(x['state']['others'])} run_len={c['run_len']} tiles={c['distinct']}")
            out(f"  mask legal_actions={legal}")
            out(f"  P(action) [base mask]: " + ", ".join(f"{cfg.ACTIONS[i]}={x['probs_base'][i]:.4f}" for i in range(6)))
            out(f"  has_bombing_target={bool(x['features'][11])} bomb_available={bool(x['features'][10])} "
                f"nearest_bombing_position_distance={x['features'][12]:.3f} has_reachable_coin={bool(x['features'][4])}")
            wasteful = _bomb_wasteful_mismatch_check(extract_semantic_state(restore_state(x["state"])), restore_state(x["state"]))
            out(f"  if BOMB were chosen here: crates_in_blast={wasteful['crates']} "
                f"threatens_reachable_opponent={wasteful['threatens_reachable_opponent']} "
                f"currently_wasteful={wasteful['currently_wasteful']} (P(BOMB) shown above is the actual policy output)")


# =========================================================== (d) wasteful-bomb-penalty
def section_d():
    out("=== (d) wasteful-bomb-penalty: code check + crosstab + kill-only accuracy + reward estimate ===")
    out("(d.1) code check: rewards.py::compute_reward() applies WASTEFUL_BOMB_PENALTY iff, at the BOMB_DROPPED "
        "step's old_game_state, crates_in_blast(pos)==0 AND NOT bomb_threatens_reachable_opponent(pos) -- i.e. "
        "penalty fires ONLY on the 'covers neither' cell of the classification below. Verified against "
        f"agent_code/rhine/rewards.py lines ~122-128 and state_processing.py's bomb_threatens_reachable_opponent() "
        f"(pure snapshot: any opponent's CURRENT position in blast_coords(pos, power), no reachability "
        f"pre-filter, no future-movement prediction) and crates_in_blast() (blast_coords() tiles where "
        f"field==1) -- matches the documented behaviour exactly, no discrepancy found.\n")

    crosstab = Counter()
    kill_only_by_bucket = {b: {"n": 0, "hit": 0} for b in ("=0", "(0,0.25)", "[0.25,0.5)", "[0.5,0.75)", ">=0.75")}
    per_seed_costs = []
    for s in SEEDS:
        payload = A.load(s, "A")
        n_rounds = len(payload["records"])
        seed_o1 = seed_o2_05 = seed_o2_1 = seed_o3_05 = seed_o3_1 = seed_o3_2 = 0.0
        for r in payload["records"]:
            recs = r["steps"]
            kills_seq = [x["kills"] for x in recs] + [r["metrics"]["opponent_kills"]]
            steps_seq = [x["state"]["step"] for x in recs] + [recs[-1]["state"]["step"] + 1]
            for i, x in enumerate(recs):
                if x["action"] != "BOMB":
                    continue
                state = restore_state(x["state"])
                semantic = extract_semantic_state(state)
                pos = semantic.self_pos
                crates = crates_in_blast(state["field"], pos, cfg.BOMB_POWER)
                blast = set(blast_coords(state["field"], pos, cfg.BOMB_POWER))
                covers_opp = any(o in blast for o in semantic.opponents)
                window = [j for j in range(i + 1, len(steps_seq)) if steps_seq[j] - x["state"]["step"] <= A.KILL_CHECK_WINDOW]
                got_kill = any(kills_seq[j] > x["kills"] for j in window)
                cratehit = crates > 0
                key_cover = "crates" if crates > 0 else ("opp_only" if covers_opp else "neither")
                key_result = "kill" if got_kill else ("crate" if cratehit else "neither")
                crosstab[(key_cover, key_result)] += 1

                if key_cover == "opp_only":
                    hyp_danger = {t: set(o) for t, o in semantic.danger_offsets.items()}
                    for bt in blast:
                        hyp_danger.setdefault(bt, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
                    diffs = [
                        float(sp._opponent_escape_difficulty(
                            o, semantic.field_arr, semantic.blocked, hyp_danger, pos,
                            [oo for oo in semantic.opponents if oo != o],
                        ))
                        for o in semantic.opponents if o in blast
                    ]
                    d = max(diffs) if diffs else 0.0
                    bucket = "=0" if d == 0 else ("(0,0.25)" if d < 0.25 else "[0.25,0.5)" if d < 0.5 else "[0.5,0.75)" if d < 0.75 else ">=0.75")
                    kill_only_by_bucket[bucket]["n"] += 1
                    kill_only_by_bucket[bucket]["hit"] += got_kill

                    if d == 0:
                        seed_o2_05 += 0.05
                        seed_o2_1 += 0.1
                if key_cover == "crates" and key_result == "neither":
                    seed_o3_05 += 0.05
                    seed_o3_1 += 0.1
                    seed_o3_2 += 0.2
                if crates == 0 and not covers_opp:
                    seed_o1 += 0.2  # Magnitude of the current WASTEFUL_BOMB_PENALTY.
        per_seed_costs.append((seed_o1, seed_o2_05, seed_o2_1, seed_o3_05, seed_o3_1, seed_o3_2, n_rounds))
        out(f"seed{s}: (O1 current) total_wasteful_penalty_this_run={seed_o1:.1f} ({seed_o1 / n_rounds:.3f}/round)")

    out("\n(d.2) crosstab: placement-snapshot classification x settlement outcome (counts pooled 3 seeds, per-round mean in parens)")
    out("|  | settled: kill | settled: crate destroyed | settled: neither |")
    out("|---|---|---|---|")
    for cover in ("crates", "opp_only", "neither"):
        row = [crosstab[(cover, res)] for res in ("kill", "crate", "neither")]
        out(f"| covers {cover} | " + " | ".join(f"{v} ({v/300:.2f}/round)" for v in row) + " |")

    out("\n(d.3) kill-only bombs (blast covers an opponent's current position, no crate) by escape-difficulty score at placement:")
    out("| difficulty bucket | n (per round) | hit rate |")
    for b, d in kill_only_by_bucket.items():
        out(f"| {b} | {d['n']} ({d['n']/300:.2f}/round) | {d['hit']}/{d['n']} = "
            f"{100*d['hit']/d['n'] if d['n'] else float('nan'):.1f}% |")

    out("\n(d.4) estimated cumulative per-round penalty under each option (mean over 3 seeds, 100 rounds each):")
    o1 = np.mean([c[0] / c[6] for c in per_seed_costs])
    o2_05 = np.mean([(c[0] + c[1]) / c[6] for c in per_seed_costs])
    o2_1 = np.mean([(c[0] + c[2]) / c[6] for c in per_seed_costs])
    o3_05 = np.mean([(c[0] + c[3]) / c[6] for c in per_seed_costs])
    o3_1 = np.mean([(c[0] + c[4]) / c[6] for c in per_seed_costs])
    o3_2 = np.mean([(c[0] + c[5]) / c[6] for c in per_seed_costs])
    out(f"  O1 (current): {o1:.3f}/round")
    out(f"  O2, c'=0.05 (only difficulty==0 kill-only bombs get an extra penalty): {o2_05:.3f}/round")
    out(f"  O2, c'=0.10: {o2_1:.3f}/round")
    out(f"  O3, c=0.05 (crate-covering bombs that settle with no crate destroyed and no kill): {o3_05:.3f}/round")
    out(f"  O3, c=0.10: {o3_1:.3f}/round")
    out(f"  O3, c=0.20: {o3_2:.3f}/round")
    out("\n(d.5) 'penalize any bomb that settles with nothing destroyed' variant: not evaluated, per instruction.")



# =========================================================== (i) late-game deadlock tile pair
F_HAS_KILL, F_KILL_DIST, F_KILL_DIR, F_KILL_VALUE = 21, 22, (23, 24, 25, 26), 27

DIR_OFFSETS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
DIR_ORDER = ("UP", "DOWN", "LEFT", "RIGHT")


def _late_game_deadlocks(payload):
    cases = []
    for r in payload["records"]:
        actions = r["metrics"]["actions"]
        run_len, start = A.longest_oscillation_run(actions)
        if run_len < A.OSCILLATION_THRESHOLD:
            continue
        recs = r["steps"]
        window = recs[start:start + run_len + 1]
        feats = np.stack([w["features"] for w in window])
        period2_same = np.mean([np.array_equal(feats[k], feats[k + 2]) for k in range(len(feats) - 2)]) if len(feats) > 2 else 0.0
        crates_left = int(np.sum(restore_state(window[0]["state"])["field"] == 1))
        positions = [tuple(int(v) for v in w["state"]["self"][3]) for w in window]
        distinct = sorted(set(positions))
        if crates_left == 0 and period2_same >= 0.99 and len(distinct) == 2:
            cases.append((r, window, distinct))
    return cases


def section_i():
    out("=== (i) B group late-game deadlock (50 expected): per-tile kill-target/BOMB-legality detail ===")
    out("Category rule: for each of the 2 deadlock tiles, check if it is itself the kill target tile "
        "(has_kill_target and nearest_kill_distance==0). If so: BOMB-illegal -> cat1; BOMB-legal with low "
        "P(BOMB) -> cat2. If neither tile is itself the target, check if either tile's kill_direction points "
        "at the other tile -> cat3 ('pulled back by a kill direction toward a target elsewhere'). Else -> cat4.\n")
    categories = Counter()
    total_cases = 0
    for s in SEEDS:
        payload = A.load(s, "B")
        cases = _late_game_deadlocks(payload)
        for r, window, distinct in cases:
            total_cases += 1
            per_tile = {}
            for tile in distinct:
                rec = next(w for w in window if tuple(int(v) for v in w["state"]["self"][3]) == tile)
                f = rec["features"]
                per_tile[tile] = {
                    "has_kill_target": bool(f[F_HAS_KILL]), "is_target_tile": bool(f[F_HAS_KILL]) and f[F_KILL_DIST] == 0.0,
                    "bomb_available": bool(f[10]), "bomb_legal": bool(rec["base_mask"][BOMB_IDX]),
                    "p_bomb": float(rec["probs_base"][BOMB_IDX]), "f28": float(f[F_KILL_VALUE]),
                    "kill_dirs": {d: bool(f[F_KILL_DIR[i]]) for i, d in enumerate(DIR_ORDER)},
                }
            target_tiles = [t for t in distinct if per_tile[t]["is_target_tile"]]
            label = None
            detail = ""
            if target_tiles:
                t = target_tiles[0]
                info = per_tile[t]
                if not info["bomb_legal"]:
                    label = "cat1_target_tile_bomb_illegal"
                else:
                    label = "cat2_target_tile_bomb_legal_low_p"
                detail = f"target_tile={t} bomb_available={info['bomb_available']} bomb_legal={info['bomb_legal']} P(BOMB)={info['p_bomb']:.4f} #28={info['f28']:.3f}"
            else:
                a, b = distinct
                pulled = False
                for src, dst in ((a, b), (b, a)):
                    for d, on in per_tile[src]["kill_dirs"].items():
                        if on and DIR_OFFSETS[d][0] + src[0] == dst[0] and DIR_OFFSETS[d][1] + src[1] == dst[1]:
                            pulled = True
                if pulled:
                    label = "cat3_nontarget_pulled_by_kill_direction"
                else:
                    label = "cat4_other"
                detail = f"tileA={a} kill_dirs={per_tile[a]['kill_dirs']} has_kill={per_tile[a]['has_kill_target']} | " \
                          f"tileB={b} kill_dirs={per_tile[b]['kill_dirs']} has_kill={per_tile[b]['has_kill_target']}"
            categories[label] += 1
            out(f"seed{s} round={r['round']} tiles={distinct} -> {label}\n    {detail}")
    out(f"\ncategory totals (of {total_cases} late-game deadlock cases): {dict(categories)}")


# =========================================================== (k) mask BOMB when current-tile-in-danger + safe alt exists
def section_k():
    out("=== (k) approximate replay: mask BOMB whenever current_tile_in_danger and a safe non-BOMB action exists ===")
    out("'safe non-BOMB action' reuses the mask's own existing per-action legality (movement legality is already "
        "gated on exists_safe_path()/WAIT on the same check -- see action_mask.py); this rule does not add any new "
        "safety computation, it only additionally removes BOMB when danger is present and >=1 such action is "
        "already legal. Approximate: does not simulate what the policy would pick instead.\n")
    total_flipped = 0
    total_flipped_was_opportunity = 0
    total_steps = 0
    per_case = {"got_killed": set(), "self_kill": set()}
    for s in SEEDS:
        payload = A.load(s, "A")
        seed_flipped = seed_opp = 0
        for r in payload["records"]:
            recs = r["steps"]
            total_steps += len(recs)
            round_flipped = False
            for x in recs:
                mask = x["base_mask"]
                danger = bool(x["features"][18])
                if not (danger and mask[BOMB_IDX]):
                    continue
                other_legal = any(mask[i] for i in range(6) if i != BOMB_IDX)
                if not other_legal:
                    continue
                seed_flipped += 1
                round_flipped = True
                state = restore_state(x["state"])
                blast = set(blast_coords(state["field"], tuple(int(v) for v in x["state"]["self"][3]), cfg.BOMB_POWER))
                opp_positions = {tuple(int(v) for v in o[3]) for o in x["state"]["others"]}
                if opp_positions & blast:
                    seed_opp += 1
            if round_flipped:
                if r["metrics"]["got_killed_by_opponent"]:
                    per_case["got_killed"].add((s, r["round"]))
                if r["metrics"]["self_kill"]:
                    per_case["self_kill"].add((s, r["round"]))
        out(f"seed{s}: steps with BOMB flipped to illegal under this rule={seed_flipped} "
            f"({seed_flipped/100:.2f}/round); of those, opponent in blast (kill opportunity)={seed_opp} "
            f"({100*seed_opp/seed_flipped if seed_flipped else float('nan'):.1f}%)")
        total_flipped += seed_flipped
        total_flipped_was_opportunity += seed_opp
    out(f"\nPOOLED: {total_flipped} steps flipped ({total_flipped/300:.2f}/round); "
        f"{total_flipped_was_opportunity}/{total_flipped} ({100*total_flipped_was_opportunity/total_flipped if total_flipped else float('nan'):.1f}%) "
        f"were kill opportunities given up")
    out(f"got_killed cases (5 total) containing >=1 such flipped step: {len(per_case['got_killed'])}/5 -> {sorted(per_case['got_killed'])}")
    out(f"self-kill cases (19 total) containing >=1 such flipped step: {len(per_case['self_kill'])}/19 -> {sorted(per_case['self_kill'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["a", "b", "c", "d", "i", "k", "all"])
    args = parser.parse_args()
    sections = {"a": section_a, "b": section_b, "c": section_c, "d": section_d, "i": section_i, "k": section_k}
    for name in (sections if args.section == "all" else [args.section]):
        sections[name]()


if __name__ == "__main__":
    main()
