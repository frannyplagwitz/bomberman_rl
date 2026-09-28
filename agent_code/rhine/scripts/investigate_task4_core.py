"""Task 4 offline investigation, sections a/b/c/d.3/e, read-only on the
eval_stage_d_final.py recordings under agent_code/rhine/logs/stage_d_final/task4/
(task4_seed{0,1,2}_{breaker_only,deadlock}_A.pkl.gz, task4_seed{0,1,2}_no_breaker_B.pkl.gz).

Reuses Stage D's analysis functions, which depend only on the recording
schema (unchanged for Task 4); features #1-28 keep their indices in the
31-dim vector.

Usage:
  python -m agent_code.rhine.scripts.investigate_task4_core <section> [--out FILE]
  sections: a | b | c | d3 | e | all
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.analyze_stage_d_final import (
    F_BOMB_DIST, F_CRATES, F_HAS_BOMB_TARGET, F_HAS_KILL, F_KILL_DIST, F_KILL_VALUE,
    _classify_selfkill, _collect_kill_data, _conditional_corr_groups, _corr, _fast_opportunities,
    _final_step_blocked, _mean_std, _osc_cases, _rate, _siege_or_c1,
)
from agent_code.rhine.scripts.diagnose_kill_target_discrimination import (
    KILL_CHECK_WINDOW, WASTEFUL_MISMATCH_THRESHOLD,
)
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _dump_tail, _instrument_step
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import blast_coords

# Task 4's evaluation config. Only affects the legal_actions display of
# _instrument_step(); reported statistics use the recorded masks.
cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = False

SEEDS = (0, 1, 2)
BOMB_IDX = cfg.ACTIONS.index("BOMB")
DATA_DIR = Path(__file__).resolve().parents[1] / "logs" / "stage_d_final" / "task4"
_out = None


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


TAGS = {
    "breaker_only": ("breaker_only", "A"),
    "deadlock": ("deadlock", "A"),
    "no_breaker": ("no_breaker", "B"),
}


def load_task4(seed: int, group: str) -> dict:
    """group in TAGS.keys()."""
    label, suffix = TAGS[group]
    path = DATA_DIR / f"task4_seed{seed}_{label}_{suffix}.pkl.gz"
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"]:
        raise RuntimeError(f"{path} is a partial recording")
    return payload


# ---------------------------------------------------------------- (a) invalid actions
def section_a():
    out("=== (a) invalid_action_own_cause / contested_tile (all 3 groups x 3 seeds) ===")
    out("Reuses: EpisodeMetrics/agg fields as recorded by eval_stage_d_final.py (unchanged schema).")
    all_own = []
    for group in ("breaker_only", "deadlock", "no_breaker"):
        for s in SEEDS:
            payload = load_task4(s, group)
            agg = payload["agg"]
            own = agg["invalid_action_own_cause_count_total"]
            contested = agg["invalid_action_contested_tile_count_total"]
            total = agg["invalid_action_count_total"]
            steps = sum(r["metrics"]["steps"] for r in payload["records"])
            all_own.append(own)
            out(f"  {group:12s} seed{s}: own_cause={own} contested_tile={contested} total={total} "
                f"(other={total - own - contested}) steps={steps} per_1000_steps={1000 * total / steps:.2f}")
    out(f"\nown_cause across all 9 recordings: {all_own} -> "
        f"{'CONFIRMED all 0' if all(v == 0 for v in all_own) else 'FAIL: nonzero own_cause found'}")
    out("sample size: 9 recordings x 100 rounds = 900 rounds total")


# ---------------------------------------------------------------- (b) oscillation
def section_b():
    out("\n=== (b) oscillation pattern (new_definition: late_game_no_score flag added on top of "
        "reused _osc_cases()) ===")
    out("Reuses: analyze_stage_d_final._osc_cases()/diagnose_oscillation.longest_oscillation_run() "
        "unchanged (same Stage D oscillation convention: strict-run-length threshold + per-case "
        "crates_left/frac_has_coin/frac_has_kill_target/n_opponents_mean fields).")
    out("New definition: late_game_no_score = (crates_left == 0) and (len(state['coins']) == 0) at the "
        "run's start step -- 'no further real scoring opportunity anywhere on the board', not present "
        "verbatim in the Stage D oscillation analysis but directly implied by its own 'crates cleared, no coins' description.")

    for group_label, group in (("no_breaker (breaker OFF)", "no_breaker"), ("breaker_only (breaker ON, residual)", "breaker_only")):
        out(f"\n--- group: {group_label} ---")
        total_cases = 0
        late_game = 0
        near_opp = 0
        for s in SEEDS:
            payload = load_task4(s, group)
            cases = _osc_cases(payload)
            total_cases += len(cases)
            frac = payload["agg"]["oscillation_fraction"]
            out(f"  seed{s}: oscillation_fraction={frac:.3f} ({len(cases)}/100 rounds)")
            for c in cases:
                # Raw coin count at run start, re-read from the same recording.
                r = next(rr for rr in payload["records"] if rr["round"] == c["round"])
                start_state = r["steps"][c["start"]]["state"]
                n_coins_at_start = len(start_state["coins"])
                is_late = c["crates_left"] == 0 and n_coins_at_start == 0
                if is_late:
                    late_game += 1
                if c["n_opponents_mean"] >= 1:
                    near_opp += 1
                out(f"    round={c['round']:>3} run_start={c['start']:>3} run_len={c['run_len']:>3}/{c['total_steps']} "
                    f"tiles={c['distinct']} crates_left={c['crates_left']} coins_at_start={n_coins_at_start} "
                    f"late_game_no_score={is_late} opponents_alive_mean={c['n_opponents_mean']:.1f} "
                    f"has_kill_target_frac={c['frac_has_kill_target']:.2f} runs_to_round_end={c['runs_to_end']}")
        out(f"\n  {group_label} TOTAL: {total_cases} sustained-oscillation rounds across 3 seeds "
            f"(sample size n={total_cases}); late_game_no_score={late_game}/{total_cases} "
            f"({100 * late_game / total_cases if total_cases else float('nan'):.1f}%); "
            f"rounds with >=1 opponent alive during the run={near_opp}/{total_cases}")

    out("\nComparison to Stage D oscillation results (28-dim, 3x coin_collector_agent, "
        "B group): 80 sustained cases/300 rounds, all strict 2-tile bounces, 50/80 (62.5%) late-game "
        "deadlocks (crates cleared/no coin/kill target present but not pursued), mean 1.8 opponents alive; "
        "A group residual 2/300 (0.7%). Task 4 uses rule_based_agent (not coin_collector_agent) -- numbers "
        "not expected to match exactly, comparison is qualitative only.")


# ---------------------------------------------------------------- (c) self-kill
def section_c():
    out("\n=== (c) self-kill classification (group breaker_only, A/B/C + c1/c2/c-class) ===")
    out("Reuses: analyze_stage_d_final._classify_selfkill()/_siege_or_c1() unchanged -- same "
        "Stage D self-kill census label taxonomy (c2_contested_tile, c1_later_leg_conflict, "
        "c_new_single_opponent_siege, opp_new_bomb_after_placement, own_cause_invalid, unclassified). "
        "A/B-class (think-time forced WAIT) structurally impossible: this evaluation runs with "
        "think-time limit disabled (eval_stage_d_final.py always calls "
        "common._disable_think_time_limit()).")
    from collections import Counter
    labels = Counter()
    per_seed_selfkill = {}
    per_seed_gotkilled = {}
    dump_path = DATA_DIR / "task4_selfkill_tail_dumps.txt"
    dump_log = open(dump_path, "w")
    total_rounds = 0
    both_flag_count = 0
    for s in SEEDS:
        payload = load_task4(s, "breaker_only")
        total_rounds += len(payload["records"])
        sk_cases = [r for r in payload["records"] if r["metrics"]["self_kill"]]
        gk_cases = [r for r in payload["records"] if r["metrics"]["got_killed_by_opponent"]]
        both = [r for r in payload["records"] if r["metrics"]["self_kill"] and r["metrics"]["got_killed_by_opponent"]]
        both_flag_count += len(both)
        per_seed_selfkill[s] = len(sk_cases)
        per_seed_gotkilled[s] = len(gk_cases)
        out(f"\nseed{s}: self_kill_rate={payload['agg']['self_kill_rate']:.3f} ({len(sk_cases)}/100) "
            f"got_killed_rate={payload['agg']['got_killed_by_opponent_rate']:.3f} ({len(gk_cases)}/100) "
            f"both-flags-set={len(both)}")
        for r in sk_cases:
            label, bomb_idx, flags = _classify_selfkill(r)
            extra = ""
            if label == "unclassified" and bomb_idx is not None:
                findings = _siege_or_c1(r, bomb_idx)
                siege = [f for f in findings if f[2] == 1 and f[3] >= 2 and f[4]]
                conflict_later = [f for f in findings if f[2] >= 1]
                if siege:
                    label = "c_new_single_opponent_siege"
                elif conflict_later:
                    label = "c1_later_leg_conflict"
                extra = f" findings_tail={findings[-3:]}"
            labels[label] += 1
            recs = r["steps"]
            out(f"  round={r['round']:>3} bomb_idx={bomb_idx} label={label} "
                f"last_actions={[x['action'] for x in recs[-6:]]} death_pos={r['death']['pos'] if r['death'] else None}"
                f"{extra}")
            tail = recs[-25:]
            step_log = [_instrument_step(restore_state(x["state"])) for x in tail]
            dump_log.write(f"\n{'=' * 20} seed{s} round={r['round']} label={label} {'=' * 20}\n")
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                _dump_tail(dump_log, step_log, [x["action"] for x in tail], 20)
    dump_log.close()
    total = sum(labels.values())
    out(f"\nlabel totals across 3 seeds: {dict(labels)} (total self-kills={total}); "
        f"per-seed self-kill counts={per_seed_selfkill}")
    if total:
        c2 = labels.get("c2_contested_tile", 0)
        c_class = labels.get("c_new_single_opponent_siege", 0)
        out(f"c2 share = {c2}/{total} = {100 * c2 / total:.1f}% (Stage D 28-dim baseline 17/19=89.5%, "
            f"Stage C baseline 26/29=89.7%)")
        out(f"c-class (new single-opponent siege) count = {c_class}; "
            f"c1_later_leg_conflict = {labels.get('c1_later_leg_conflict', 0)}; "
            f"opp_new_bomb_after_placement = {labels.get('opp_new_bomb_after_placement', 0)}; "
            f"own_cause_invalid = {labels.get('own_cause_invalid', 0)}; "
            f"unclassified(after siege/c1 fallback) = {labels.get('unclassified', 0)}")
    else:
        out("No self-kill cases in this dataset -- c2/c1/c-class shares cannot be computed (data insufficient).")
    n_sk = sum(per_seed_selfkill.values())
    n_gk = sum(per_seed_gotkilled.values())
    total_death = n_sk + n_gk - both_flag_count
    out(f"\nself_kill/got_killed mutual-exclusivity check: {both_flag_count} rounds had both flags set "
        f"(out of {total_rounds} total rounds) -> "
        f"{'CONFIRMED mutually exclusive' if both_flag_count == 0 else 'NOT mutually exclusive -- see EpisodeMetrics construction'}")
    out(f"total death rate (self_kill OR got_killed_by_opponent) = {total_death}/{total_rounds} "
        f"= {100 * total_death / total_rounds:.1f}% (self_kill={n_sk}, got_killed={n_gk})")
    out(f"tail dumps for manual review: {dump_path}")


# ---------------------------------------------------------------- (d.3) kill behaviour
def _first_layer_amplification_task4(seed, feats):
    ckpt = torch.load(common.MODELS_DIR / f"task4_seed{seed}.pt", map_location="cpu")
    w = ckpt["model_state_dict"]["trunk.0.weight"].numpy()
    l2 = np.linalg.norm(w, axis=0)
    std = feats.std(axis=0)
    trained = l2 * std
    baselines = []
    for k in range(20):
        torch.manual_seed(k)
        net = ActorCriticMLP(n_features=cfg.MODEL_CONFIG.n_features, n_actions=6,
                              hidden_sizes=cfg.MODEL_CONFIG.hidden_sizes)
        baselines.append(np.linalg.norm(net.trunk[0].weight.detach().numpy(), axis=0) * std)
    base = np.mean(baselines, axis=0)
    order = np.argsort(-trained)
    rank = {int(j): int(np.where(order == j)[0][0]) + 1 for j in (F_CRATES, F_KILL_VALUE)}
    return trained, base, rank


def section_d3():
    out("\n=== (d.3) kill behaviour (group breaker_only) ===")
    out("Reuses: analyze_stage_d_final._collect_kill_data()/_conditional_corr_groups()/"
        "_fast_opportunities() and diagnose_kill_target_discrimination.is_opportunity()/"
        "_bomb_wasteful_mismatch_check() unchanged -- same Stage D kill-behavior opportunity "
        "definition (bomb_available + opponent in blast_coords(self_pos) + BOMB mask-legal). "
        "Feature index constants (#28 kill value etc.) are unchanged positions in the 31-dim vector.")
    from collections import defaultdict
    tot = defaultdict(list)
    for s in SEEDS:
        payload = load_task4(s, "breaker_only")
        opp, wasteful, n_steps = _collect_kill_data(payload)
        bombed = [e for e in opp if e["chosen"] == "BOMB"]
        not_bombed = [e for e in opp if e["chosen"] != "BOMB"]
        n_opp = len(opp)
        hit_kill = sum(1 for e in bombed if e["hit_kill"])
        out(f"\nseed{s}: kills/round={payload['agg']['opponent_kills_mean']:.2f} "
            f"bombs_placed/round={payload['agg']['bombs_dropped_mean']:.1f} "
            f"score_mean={payload['agg']['score_mean']:.2f} "
            f"coins_collected_mean={payload['agg']['coins_collected_mean']:.2f}")
        coins = payload["agg"]["coins_collected_mean"]
        kills = payload["agg"]["opponent_kills_mean"]
        crates = payload["agg"]["crates_destroyed_mean"]
        kill_share = 100 * 5 * kills / (coins + 5 * kills) if (coins + 5 * kills) else float("nan")
        out(f"  score composition (coin=1,kill=5; crates don't score directly): coins_mean={coins:.2f} "
            f"kills_mean={kills:.2f} crates_destroyed_mean={crates:.2f} score share from kills={kill_share:.0f}%")
        out(f"  opportunities={n_opp} ({n_opp / 100:.2f}/round, {1000 * n_opp / n_steps:.1f} per 1000 steps); "
            f"BOMB chosen (initiative)={len(bombed)} ({100 * len(bombed) / n_opp if n_opp else float('nan'):.1f}%)")
        out(f"  hit rate (kill within {KILL_CHECK_WINDOW} ticks)={hit_kill}/{len(bombed)} "
            f"({100 * hit_kill / len(bombed) if bombed else float('nan'):.1f}%)")
        nb_v = np.array([e["f28"] for e in not_bombed]); nb_p = np.array([e["p_bomb"] for e in not_bombed])
        r_nb = _corr(nb_v, nb_p)
        out(f"  not-bombed: n={len(not_bombed)} corr(#28,P(BOMB))={r_nb:.3f}")
        kg, cg = _conditional_corr_groups(payload)
        r_kill = _corr([a for a, _ in kg], [b for _, b in kg])
        r_crate = _corr([a for a, _ in cg], [b for _, b in cg])
        out(f"  wider group (current tile is target & BOMB legal): kill corr(#28,P(BOMB))={r_kill:.3f} (n={len(kg)}); "
            f"crate baseline corr(#18,P(BOMB))={r_crate:.3f} (n={len(cg)})")
        all_feats = np.stack([x["features"] for r in payload["records"] for x in r["steps"]])
        trained, base, rank = _first_layer_amplification_task4(s, all_feats)
        n_feat = cfg.MODEL_CONFIG.n_features
        out(f"  first-layer L2*std: #18 crates={trained[F_CRATES]:.3f} "
            f"(x{trained[F_CRATES] / base[F_CRATES]:.1f} vs random init, rank {rank[F_CRATES]}/{n_feat}); "
            f"#28 kill_value={trained[F_KILL_VALUE]:.3f} (x{trained[F_KILL_VALUE] / base[F_KILL_VALUE]:.1f}, "
            f"rank {rank[F_KILL_VALUE]}/{n_feat})")
        for key, val in (("opp", n_opp), ("bombed", len(bombed)), ("hit", hit_kill), ("steps", n_steps)):
            tot[key].append(val)
        tot["r_nb"].append(r_nb); tot["r_kill"].append(r_kill); tot["r_crate"].append(r_crate)
    n_opp, n_bombed = sum(tot["opp"]), sum(tot["bombed"])
    out(f"\nALL 3 seeds (n={n_opp} opportunities, 300 rounds): opportunities/round={n_opp / 300:.2f}, "
        f"initiative(BOMB chosen)={n_bombed}/{n_opp} ({100 * n_bombed / n_opp:.1f}%), "
        f"hit={sum(tot['hit'])}/{n_bombed} ({100 * sum(tot['hit']) / n_bombed:.1f}%)")
    out(f"corr(#28,P(BOMB)) not-bombed group by seed: {[round(v, 3) for v in tot['r_nb']]} "
        f"wider-group kill: {[round(v, 3) for v in tot['r_kill']]} crate baseline: {[round(v, 3) for v in tot['r_crate']]}")
    out("Stage D reference (28-dim, 3x coin_collector_agent, "
        "NOT a controlled comparison -- opponent type differs): 33.9 opp/round, initiative 66.0%, hit 1.6%.")


# ---------------------------------------------------------------- (e) got_killed
def section_e():
    out("\n=== (e) got_killed_by_opponent classification (group breaker_only) ===")
    out("Reuses: analyze_stage_d_final.section_gotkilled()'s raw-field extraction unchanged "
        "(killer_owners, n_distinct_owners, opponents_alive, near_opps, in_danger_last8, "
        "legal_moves_last8, contested_steps, opponent_bombs_on_board). New: an explicit category "
        "label per case (single_opponent_collision / multi_opponent_siege / c2_final_step_contested) "
        "following the same descriptive scheme the Stage D got-killed case table used, not present "
        "there as a formal function -- classification rule below.")
    out("New category rule: c2_final_step_contested if a contested-tile INVALID_ACTION occurred at the "
        "final step; else multi_opponent_siege if opponents_alive>=2 at death; else single_opponent_collision.")
    from collections import Counter
    cat_counts = Counter()
    n_cases = 0
    dump_path = DATA_DIR / "task4_gotkilled_tail_dumps.txt"
    dump_log = open(dump_path, "w")
    for s in SEEDS:
        payload = load_task4(s, "breaker_only")
        cases = [r for r in payload["records"] if r["metrics"]["got_killed_by_opponent"]]
        out(f"\nseed{s}: got_killed_rate={payload['agg']['got_killed_by_opponent_rate']:.3f} ({len(cases)}/100)")
        for r in cases:
            n_cases += 1
            recs, death = r["steps"], r["death"]
            pos = tuple(death["pos"])
            killers = sorted({owner for owner, coords, timer in death["expl"] if pos in {tuple(c) for c in coords}})
            last_state = restore_state(recs[-1]["state"])
            covering = []
            for bpos, timer, owner in recs[-1]["world"]["bombs"]:
                if pos in set(blast_coords(last_state["field"], bpos, cfg.BOMB_POWER)):
                    covering.append((owner, bpos, timer))
            contested_steps = [
                recs[k]["state"]["step"] for k in range(max(len(recs) - 6, 0), len(recs) - 1)
                if recs[k + 1]["invalid_contested"] > recs[k]["invalid_contested"]
            ]
            if _final_step_blocked(recs, death):
                contested_steps.append(recs[-1]["state"]["step"])
            n_opp_bombs = len({(tuple(b[0]), b[2]) for b in recs[-1]["world"]["bombs"] if b[2] != "rhine"})
            opponents_alive = len(recs[-1]["state"]["others"])
            legal_moves_last8 = [
                len([a for a in ("UP", "DOWN", "LEFT", "RIGHT") if x["final_mask"][cfg.ACTIONS.index(a)]])
                for x in recs[-8:]
            ]
            near_opps = sum(1 for o in recs[-1]["state"]["others"]
                             if abs(o[3][0] - pos[0]) + abs(o[3][1] - pos[1]) <= 6)
            if contested_steps:
                category = "c2_final_step_contested"
            elif opponents_alive >= 2:
                category = "multi_opponent_siege"
            else:
                category = "single_opponent_collision"
            cat_counts[category] += 1
            out(f"  round={r['round']:>3} step={death['step']} pos={pos} category={category} "
                f"killer_owners={killers} n_distinct_owners={len({c[0] for c in covering})} "
                f"opponents_alive={opponents_alive} within6={near_opps} "
                f"legal_moves_last8={legal_moves_last8} "
                f"opponent_bombs_on_board={n_opp_bombs} contested_tile_invalid_in_last5={contested_steps}")
            tail = recs[-25:]
            step_log = [_instrument_step(restore_state(x["state"])) for x in tail]
            dump_log.write(f"\n{'=' * 20} seed{s} round={r['round']} category={category} killers={killers} {'=' * 20}\n")
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                _dump_tail(dump_log, step_log, [x["action"] for x in tail], 20)
    dump_log.close()
    if n_cases == 0:
        out("\nNo got_killed_by_opponent case: nothing to classify (data insufficient -- n=0).")
    else:
        out(f"\ncategory totals (n={n_cases}): {dict(cat_counts)}")
    out(f"tail dumps for manual review: {dump_path}")


def main():
    global _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["a", "b", "c", "d3", "e", "all"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else DATA_DIR / f"investigation_core_{args.section}.txt"
    _out = open(out_path, "w")
    sections = {"a": section_a, "b": section_b, "c": section_c, "d3": section_d3, "e": section_e}
    for name in (sections if args.section == "all" else [args.section]):
        sections[name]()
    _out.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
