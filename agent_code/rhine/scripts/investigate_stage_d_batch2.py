"""Stage D batch-2 read-only investigation, offline on recorded per-step data
(agent_code/rhine/logs/stage_d_final/ for Stage D group A,
agent_code/rhine/logs/stage_d_zeroshot_rulebased/ for the rule_based_agent
zero-shot eval). #29/#30 are computed here for analysis only.

Usage:
  python -m agent_code.rhine.scripts.investigate_stage_d_batch2 <1|2|3|4|5|all> [--source coin_collector|rulebased]
"""
import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_kill_target_discrimination import KILL_CHECK_WINDOW
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import bfs_distances, blast_coords, crates_in_blast, find_safe_path, SAFETY_HORIZON

SEEDS = (0, 1, 2)
BOMB_IDX = cfg.ACTIONS.index("BOMB")
F_KILL_VALUE = 27

SOURCES = {
    "coin_collector": (common.LOGS_DIR / "stage_d_final", "task3_stage_d_seed{s}_A.pkl.gz"),
    "rulebased": (common.LOGS_DIR / "stage_d_zeroshot_rulebased", "task3_stage_d_seed{s}_vs_rulebased_A.pkl.gz"),
}


def load(source, seed):
    d, pattern = SOURCES[source]
    with gzip.open(d / pattern.format(s=seed), "rb") as f:
        payload = pickle.load(f)
    assert payload["meta"]["complete"], f"{source} seed{seed} incomplete"
    return payload


def out(msg=""):
    print(msg)


def _other_legal_exists(mask):
    return any(mask[i] for i in range(6) if i != BOMB_IDX)


# ============================================================= (1) danger + actually chose BOMB
def section_1(source):
    out(f"=== (2.1) steps where BOMB was actually CHOSEN while current_tile_in_danger=True [{source}] ===")
    out("Population: action=='BOMB' and features[18] (current_tile_in_danger) is True -- i.e. only "
        "steps the agent actually picked, not every step BOMB happened to be mask-legal (that broader, "
        "legality-only population is what section (k) counted: 271 steps for the coin_collector A group, "
        "vs this section's narrower 'actually chose BOMB' population -- explains the 271 vs 157 gap; the "
        "157 in section (b) already used this same narrower definition, so it should reproduce here).\n")
    total = 0
    total_alt = 0
    death_within = {n: 0 for n in (1, 3, 5, 10)}
    cases = {"self_kill": set(), "got_killed": set()}
    for s in SEEDS:
        payload = load(source, s)
        seed_n = seed_alt = 0
        for r in payload["records"]:
            recs = r["steps"]
            death_step = r["death"]["step"] if r["death"] else None
            for i, x in enumerate(recs):
                if x["action"] != "BOMB" or not bool(x["features"][18]):
                    continue
                seed_n += 1
                has_alt = _other_legal_exists(x["base_mask"])
                seed_alt += has_alt
                if death_step is not None and death_step >= x["state"]["step"]:
                    gap = death_step - x["state"]["step"]
                    for n in death_within:
                        if gap <= n:
                            death_within[n] += 1
                    if r["metrics"]["self_kill"]:
                        cases["self_kill"].add((s, r["round"]))
                    if r["metrics"]["got_killed_by_opponent"]:
                        cases["got_killed"].add((s, r["round"]))
        out(f"seed{s}: BOMB-chosen-while-in-danger steps={seed_n} ({seed_n/100:.2f}/round); "
            f"safe non-BOMB alt existed={seed_alt} ({100*seed_alt/seed_n if seed_n else float('nan'):.1f}%)")
        total += seed_n
        total_alt += seed_alt
    out(f"\nPOOLED: {total} steps ({total/300:.2f}/round); safe alt existed in {total_alt}/{total} "
        f"({100*total_alt/total if total else float('nan'):.1f}%)")
    for n in (1, 3, 5, 10):
        out(f"  died within {n} step(s): {death_within[n]}/{total} ({100*death_within[n]/total if total else float('nan'):.2f}%)")
    out(f"self-kill cases containing >=1 such step: {len(cases['self_kill'])} -> {sorted(cases['self_kill'])}")
    out(f"got_killed cases containing >=1 such step: {len(cases['got_killed'])} -> {sorted(cases['got_killed'])}")


# ============================================================= (2) bomb bucketing (rule_based)
def section_2(source):
    out(f"=== (2.2) bomb bucketing by #28 at placement, kill-only bombs [{source}] ===")
    buckets = {b: {"n": 0, "hit": 0} for b in ("=0", "(0,0.25)", "[0.25,0.5)", "[0.5,0.75)", ">=0.75")}
    n_crate_covering = 0
    n_total_bombs = 0
    n_rounds = 0
    for s in SEEDS:
        payload = load(source, s)
        n_rounds += len(payload["records"])
        for r in payload["records"]:
            recs = r["steps"]
            kills_seq = [x["kills"] for x in recs] + [r["metrics"]["opponent_kills"]]
            steps_seq = [x["state"]["step"] for x in recs] + [recs[-1]["state"]["step"] + 1]
            for i, x in enumerate(recs):
                if x["action"] != "BOMB":
                    continue
                n_total_bombs += 1
                state = restore_state(x["state"])
                pos = tuple(int(v) for v in x["state"]["self"][3])
                crates = crates_in_blast(state["field"], pos, cfg.BOMB_POWER)
                if crates > 0:
                    n_crate_covering += 1
                    continue
                blast = set(blast_coords(state["field"], pos, cfg.BOMB_POWER))
                opp_positions = {tuple(int(v) for v in o[3]) for o in x["state"]["others"]}
                if not (opp_positions & blast):
                    continue
                window = [j for j in range(i + 1, len(steps_seq)) if steps_seq[j] - x["state"]["step"] <= KILL_CHECK_WINDOW]
                got_kill = any(kills_seq[j] > x["kills"] for j in window)
                d = float(x["features"][F_KILL_VALUE])
                b = "=0" if d == 0 else ("(0,0.25)" if d < 0.25 else "[0.25,0.5)" if d < 0.5 else "[0.5,0.75)" if d < 0.75 else ">=0.75")
                buckets[b]["n"] += 1
                buckets[b]["hit"] += got_kill
    out(f"total BOMB actions: {n_total_bombs} ({n_total_bombs/n_rounds:.2f}/round); "
        f"crate-covering bombs: {n_crate_covering} ({n_crate_covering/n_rounds:.2f}/round)\n")
    out("| difficulty bucket | n (per round) | hit rate |")
    for b, d in buckets.items():
        out(f"| {b} | {d['n']} ({d['n']/n_rounds:.2f}/round) | {d['hit']}/{d['n']} = "
            f"{100*d['hit']/d['n'] if d['n'] else float('nan'):.1f}% |")


# ============================================================= (3) siege deaths + offline #29/#30
def _opponent_bfs_distance(field_arr, blocked, self_pos, opp_pos):
    """Distance to an opponent: BFS distance from self_pos to the nearest
    reachable free neighbor of `opp_pos`, plus one. None if none is reachable.
    """
    best = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor = (opp_pos[0] + dx, opp_pos[1] + dy)
        if field_arr[neighbor] == -1:
            continue
        dist_map = bfs_distances(field_arr, self_pos, blocked=blocked - {opp_pos})
        if neighbor in dist_map:
            d = dist_map[neighbor] + 1
            if best is None or d < best:
                best = d
    return best


def _feat_29_30(field_arr, blocked, self_pos, opponents):
    dists = [d for d in (_opponent_bfs_distance(field_arr, blocked, self_pos, o) for o in opponents) if d is not None]
    f29 = min(dists) / 16 if dists else 1.0
    f29 = min(f29, 1.0)
    f30 = sum(1 for d in dists if d <= 3) / 3
    return f29, f30


def _safe_reachable_count(field_arr, blocked, self_pos, danger_offsets):
    dist_map = bfs_distances(field_arr, self_pos, blocked=blocked)
    n = 0
    for tile, dist in dist_map.items():
        if dist > SAFETY_HORIZON:
            continue
        if find_safe_path(tile, dist, field_arr, blocked, danger_offsets, SAFETY_HORIZON) is not None:
            n += 1
    return n


def _extract_others(state):
    return [tuple(int(v) for v in o[3]) for o in state["others"]]


def section_3(source):
    out(f"=== (2.3) siege deaths: last-10-step mask/reachability/opponent-distance trace + offline #29/#30 [{source}] ===")
    out("'safe reachable tile count' = BFS-reachable tiles from self_pos (respecting the current blocked "
        "set) for which find_safe_path(tile, bfs_dist, ...) succeeds -- i.e. reachable AND survivable once "
        "there, not just open. #29/#30 computed here only for this analysis (not wired into features.py).\n")

    from agent_code.rhine.state_processing import _bomb_positions, _to_coord

    all_f29, all_f30 = [], []
    death_f29, death_f30 = [], []
    n_cases = 0
    dump = open(SOURCES[source][0] / "siege_trace.txt", "w")
    for s in SEEDS:
        payload = load(source, s)
        for r in payload["records"]:
            if not (r["metrics"]["self_kill"] or r["metrics"]["got_killed_by_opponent"]):
                continue
            n_cases += 1
            recs = r["steps"]
            tail = recs[-10:]
            dump.write(f"\n{'='*15} seed{s} round={r['round']} "
                       f"{'self_kill' if r['metrics']['self_kill'] else 'got_killed'} {'='*15}\n")
            narrow_step = None
            for x in tail:
                state = restore_state(x["state"])
                field = state["field"]
                self_pos = tuple(int(v) for v in x["state"]["self"][3])
                opponents = _extract_others(x["state"])
                bombs_blocked = frozenset(_to_coord(pos) for pos, timer in state["bombs"])
                blocked = bombs_blocked | frozenset(opponents)
                n_legal = int(np.sum(x["final_mask"]))
                danger_offsets_here = None
                # Rebuild danger_offsets as extract_semantic_state does.
                from agent_code.rhine.state_processing import compute_danger_offsets
                danger_offsets_here = compute_danger_offsets(field, state["bombs"], state["explosion_map"], cfg.BOMB_POWER)
                safe_n = _safe_reachable_count(field, blocked, self_pos, danger_offsets_here)
                opp_dists = {o: _opponent_bfs_distance(field, blocked, self_pos, o) for o in opponents}
                f29, f30 = _feat_29_30(field, blocked, self_pos, opponents)
                all_f29.append(f29)
                all_f30.append(f30)
                if narrow_step is None and safe_n <= 1:
                    narrow_step = x["state"]["step"]
                dump.write(f"step={x['state']['step']:>4} pos={self_pos} action={x['action']:<5} "
                           f"n_legal={n_legal} safe_reachable_tiles={safe_n} n_opp_alive={len(opponents)} "
                           f"opp_dists={opp_dists} f29={f29:.3f} f30={f30:.3f}\n")
            death_f29.extend([np.nan])  # Placeholder for alignment; real values captured below.
            # Means over the last five steps.
            last5 = tail[-5:]
            case_f29, case_f30 = [], []
            for x in last5:
                state = restore_state(x["state"])
                self_pos = tuple(int(v) for v in x["state"]["self"][3])
                opponents = _extract_others(x["state"])
                bombs_blocked = frozenset(_to_coord(pos) for pos, timer in state["bombs"])
                blocked = bombs_blocked | frozenset(opponents)
                f29, f30 = _feat_29_30(state["field"], blocked, self_pos, opponents)
                case_f29.append(f29)
                case_f30.append(f30)
            death_f29[-1] = float(np.mean(case_f29))
            death_f30_val = float(np.mean(case_f30))
            death_f30.append(death_f30_val)
            if narrow_step is not None:
                lead_actions = [x["action"] for x in tail if x["state"]["step"] <= narrow_step][-4:-1]
                dump.write(f"  narrowing step (safe_reachable<=1) = {narrow_step}; preceding actions = {lead_actions}\n")
    dump.close()
    out(f"cases analyzed: {n_cases}")
    out(f"death-preceding (last 5 steps) mean: #29={np.mean(death_f29):.3f} #30={np.mean(death_f30):.3f}")
    out(f"all-recorded-step mean (this source, global background): #29={np.mean(all_f29):.3f} #30={np.mean(all_f30):.3f}")
    out(f"full per-step trace for every case: {SOURCES[source][0] / 'siege_trace.txt'}")


# ============================================================= (5) late-game deadlock bomb estimate
def section_5():
    out("=== (2.5) late-game deadlock (B group, coin_collector) BOMB-placement estimate (approximate) ===")
    out("Scope: only the 36 cat2 cases (current tile IS the kill target and BOMB is mask-legal there) -- "
        "the other 14 cat3 cases have no tile where BOMB is legal at the target itself, so this estimate "
        "does not apply to them.\n")
    from agent_code.rhine.scripts import analyze_stage_d_final as A
    from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
    A.DATA_DIR = common.LOGS_DIR / "stage_d_final"

    n_cases = 0
    sample_bombs = forced_bombs = 0
    sample_hits = forced_hits = 0
    for s in SEEDS:
        payload = A.load(s, "B")
        for r in payload["records"]:
            actions = r["metrics"]["actions"]
            run_len, start = longest_oscillation_run(actions)
            if run_len < OSCILLATION_THRESHOLD:
                continue
            recs = r["steps"]
            window = recs[start:start + run_len + 1]
            feats = np.stack([w["features"] for w in window])
            period2_same = np.mean([np.array_equal(feats[k], feats[k+2]) for k in range(len(feats)-2)]) if len(feats) > 2 else 0.0
            crates_left = int(np.sum(restore_state(window[0]["state"])["field"] == 1))
            positions = [tuple(int(v) for v in w["state"]["self"][3]) for w in window]
            distinct = sorted(set(positions))
            if not (crates_left == 0 and period2_same >= 0.99 and len(distinct) == 2):
                continue
            target_tile = None
            for tile in distinct:
                rec = next(w for w in window if tuple(int(v) for v in w["state"]["self"][3]) == tile)
                f = rec["features"]
                if bool(f[21]) and f[22] == 0.0 and bool(rec["base_mask"][BOMB_IDX]):
                    target_tile = (tile, rec)
                    break
            if target_tile is None:
                continue
            n_cases += 1
            tile, rec = target_tile
            p_bomb = float(rec["probs_base"][BOMB_IDX])
            n_visits = sum(1 for p in positions if p == tile)
            sample_bombs += n_visits * p_bomb
            state = restore_state(rec["state"])
            blast = set(blast_coords(state["field"], tile, cfg.BOMB_POWER))
            opp_positions = {tuple(int(v) for v in o[3]) for o in rec["state"]["others"]}
            in_range = bool(opp_positions & blast)
            sample_hits += n_visits * p_bomb * in_range
            forced_bombs += 1
            forced_hits += in_range
    out(f"cat2 cases with a countable target-tile visit sequence: {n_cases}/36")
    out(f"(a) sample-by-P(BOMB) over the whole deadlock window: expected BOMB placements = {sample_bombs:.2f} "
        f"total ({sample_bombs/300:.3f}/round over all 300 B-group rounds); expected placements landing an "
        f"opponent in blast = {sample_hits:.2f} ({100*sample_hits/sample_bombs if sample_bombs else float('nan'):.1f}%)")
    out(f"(b) forced-once (one BOMB the first time the target tile is visited): {forced_bombs} placements "
        f"({forced_bombs/300:.3f}/round); opponent recorded in blast at that moment = {forced_hits}/{forced_bombs} "
        f"({100*forced_hits/forced_bombs if forced_bombs else float('nan'):.1f}%)")
    out("\nApproximate: does not resimulate the round after the forced/sampled bomb (real continuation, "
        "opponent movement, and whether the resulting escape is actually safe are not modeled).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["1", "2", "3", "5", "all"])
    parser.add_argument("--source", choices=["coin_collector", "rulebased"], default="rulebased")
    args = parser.parse_args()
    if args.section in ("1", "all"):
        section_1(args.source)
    if args.section in ("2", "all"):
        section_2(args.source)
    if args.section in ("3", "all"):
        section_3(args.source)
    if args.section in ("5", "all"):
        section_5()


if __name__ == "__main__":
    main()
