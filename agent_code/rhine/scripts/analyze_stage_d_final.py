"""Offline analysis of the Stage D final-evaluation recordings written by
eval_stage_d_final.py (one gzip pickle per checkpoint x breaker group).
Read-only; reuses the existing diagnostics' helpers instead of re-running
games.

Sections: tables | invalid | oscillation | selfkill | kill | gotkilled | all
Usage:
  python -m agent_code.rhine.scripts.analyze_stage_d_final <section> [--data-dir DIR]
"""
import argparse
import contextlib
import gzip
import io
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from agent_code.rhine import config as cfg
from agent_code.rhine.model import ActorCriticMLP
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_kill_target_discrimination import (
    KILL_CHECK_WINDOW, WASTEFUL_MISMATCH_THRESHOLD, _bomb_wasteful_mismatch_check, is_opportunity,
)
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _dump_tail, _instrument_step
from agent_code.rhine.scripts.eval_stage_d_final import restore_state
from agent_code.rhine.state_processing import blast_coords, crates_in_blast, extract_semantic_state

cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True

SEEDS = (0, 1, 2)
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
BOMB_IDX = cfg.ACTIONS.index("BOMB")
F_KILL_VALUE, F_HAS_KILL, F_KILL_DIST = 27, 21, 22
F_CRATES, F_HAS_BOMB_TARGET, F_BOMB_DIST = 17, 11, 12
DATA_DIR = None
ALLOW_PARTIAL = False
_out = None


def out(msg=""):
    print(msg)
    if _out:
        _out.write(msg + "\n")
        _out.flush()


def load(seed, group):
    path = DATA_DIR / f"task3_stage_d_seed{seed}_{group}.pkl.gz"
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not payload["meta"]["complete"] and not ALLOW_PARTIAL:
        raise RuntimeError(f"{path} is a partial recording")
    return payload


def _fmt(x, digits=3):
    return f"{x:.{digits}f}"


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.3f}±{arr.std(ddof=1):.3f}"


def _corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- tables
TABLE_KEYS = [
    ("score_mean", "score_mean"), ("score_total", "score_total"),
    ("opp_kills_mean", "opponent_kills_mean"), ("got_killed_rate", "got_killed_by_opponent_rate"),
    ("self_kill_rate", "self_kill_rate"), ("completion_rate", "completion_rate"),
    ("ep_len_mean", "episode_length_mean"), ("steps_to_compl_mean", "steps_to_completion_mean"),
    ("invalid_own_cause", "invalid_action_own_cause_count_total"),
    ("invalid_contested", "invalid_action_contested_tile_count_total"),
    ("oscillation_frac", "oscillation_fraction"),
]


def section_tables():
    for group, title in (("A", "A: breaker ON + no-bomb-when-cleared"), ("B", "B: breaker OFF + no-bomb-when-cleared")):
        out(f"\n### Group {title}  (eval_seed=1000, 100 rounds, 3x coin_collector_agent; std uses ddof=1)")
        rows = {s: load(s, group)["agg"] for s in SEEDS}
        out("| seed | " + " | ".join(name for name, _ in TABLE_KEYS) + " |")
        out("|---|" + "---|" * len(TABLE_KEYS))
        for s in SEEDS:
            out(f"| seed{s} | " + " | ".join(_fmt(rows[s][key], 3 if "rate" in key or "frac" in key else 2)
                                            for _, key in TABLE_KEYS) + " |")
        out("| mean±std | " + " | ".join(_mean_std([rows[s][key] for s in SEEDS]) for _, key in TABLE_KEYS) + " |")


# ---------------------------------------------------------------- invalid
def section_invalid():
    out("\n=== (a) invalid actions ===")
    for group in ("A", "B"):
        for s in SEEDS:
            payload = load(s, group)
            own = payload["agg"]["invalid_action_own_cause_count_total"]
            contested = payload["agg"]["invalid_action_contested_tile_count_total"]
            total = payload["agg"]["invalid_action_count_total"]
            steps = sum(r["metrics"]["steps"] for r in payload["records"])
            rounds_with = sum(1 for r in payload["records"] if r["metrics"]["invalid_action_contested_tile_count"] > 0)
            out(f"group {group} seed{s}: invalid_total={total} own_cause={own} contested_tile={contested} "
                f"(other={total - own - contested}) per_1000_steps={1000 * total / steps:.2f} "
                f"rounds_with_contested={rounds_with}/100")
            if own:
                for r in payload["records"]:
                    recs = r["steps"]
                    for k in range(len(recs) - 1):
                        if recs[k + 1]["invalid_own"] > recs[k]["invalid_own"]:
                            st = recs[k]["state"]
                            out(f"  OWN_CAUSE round={r['round']} step={st['step']} pos={st['self'][3]} "
                                f"action={recs[k]['action']} legal={[cfg.ACTIONS[i] for i, m in enumerate(recs[k]['base_mask']) if m]} "
                                f"others={[o[3] for o in st['others']]} bombs={st['bombs']}")


# ---------------------------------------------------------------- oscillation
def _positions(recs):
    return [tuple(int(v) for v in r["state"]["self"][3]) for r in recs]


def _osc_cases(payload):
    cases = []
    for r in payload["records"]:
        actions = r["metrics"]["actions"]
        run_len, start = longest_oscillation_run(actions)
        if run_len < OSCILLATION_THRESHOLD:
            continue
        recs = r["steps"]
        positions = _positions(recs)
        run_positions = positions[start:start + run_len + 1]
        end = start + run_len
        window = recs[start:start + run_len + 1]
        feats = np.stack([w["features"] for w in window])
        period2_same = np.mean([np.array_equal(feats[i], feats[i + 2]) for i in range(len(feats) - 2)]) if len(feats) > 2 else float("nan")
        cases.append({
            "round": r["round"], "total_steps": len(actions), "start": start, "run_len": run_len,
            "distinct": sorted(set(run_positions)), "runs_to_end": end >= len(actions) - 1,
            "post_break": actions[end:end + 10] if end < len(actions) - 1 else [],
            "frac_has_coin": float(feats[:, 4].mean()), "frac_has_bomb_target": float(feats[:, 11].mean()),
            "frac_bomb_avail": float(feats[:, 10].mean()), "frac_in_danger": float(feats[:, 18].mean()),
            "frac_has_kill_target": float(feats[:, 21].mean()),
            "n_opponents_mean": float(np.mean([len(w["state"]["others"]) for w in window])),
            "period2_identical_features": float(period2_same),
            "actions_in_run": Counter(actions[start:start + run_len + 1]),
            "crates_left": int(np.sum(restore_state(recs[start]["state"])["field"] == 1)),
        })
    return cases


def section_oscillation():
    out("\n=== (b) oscillation (group B = breaker OFF; group A shown for the breaker effect) ===")
    for s in SEEDS:
        for group in ("B", "A"):
            payload = load(s, group)
            cases = _osc_cases(payload)
            frac = payload["agg"]["oscillation_fraction"]
            interventions = 0
            if group == "A":
                interventions = sum(
                    1 for r in payload["records"] for st in r["steps"]
                    if not np.array_equal(st["base_mask"], st["final_mask"])
                )
            out(f"\ngroup {group} seed{s}: oscillation_fraction={frac:.2f} ({len(cases)}/100 rounds)"
                + (f"  breaker-intervened steps={interventions}" if group == "A" else ""))
            if group == "A" and not cases:
                continue
            for c in cases:
                kind = "strict-2-tile bounce" if len(c["distinct"]) == 2 else f"drift over {len(c['distinct'])} tiles"
                out(f"  round={c['round']:>3} run_start={c['start']:>3} run_len={c['run_len']:>3}/{c['total_steps']} {kind} "
                    f"tiles={c['distinct']} runs_to_round_end={c['runs_to_end']} "
                    f"period2_identical_features={c['period2_identical_features']:.2f} "
                    f"has_coin={c['frac_has_coin']:.2f} has_bomb_target={c['frac_has_bomb_target']:.2f} "
                    f"bomb_avail={c['frac_bomb_avail']:.2f} in_danger={c['frac_in_danger']:.2f} "
                    f"has_kill_target={c['frac_has_kill_target']:.2f} opponents_alive={c['n_opponents_mean']:.1f} "
                    f"crates_left={c['crates_left']} actions={dict(c['actions_in_run'])} post_break={c['post_break']}")


# ---------------------------------------------------------------- self-kill
def _final_step_blocked(recs, death):
    """Whether the final move left the position unchanged while another agent
    or obstacle occupies the target: the contested-tile signature, which the
    next observation cannot show because the agent died."""
    last = recs[-1]
    action = last["action"]
    if action not in DIRS or death is None:
        return False
    pos = tuple(int(v) for v in last["state"]["self"][3])
    if tuple(death["pos"]) != pos:
        return False
    target = (pos[0] + DIRS[action][0], pos[1] + DIRS[action][1])
    return any((x, y) == target for name, x, y in death["agents"] if name != "rhine")


def _new_opponent_bombs_after(recs, bomb_idx):
    """Opponent bombs placed after rhine's bomb, with their blast tiles."""
    known = {(tuple(b[0]), b[2]) for b in recs[bomb_idx]["world"]["bombs"]}
    found = {}
    for k in range(bomb_idx + 1, len(recs)):
        field = restore_state(recs[k]["state"])["field"]
        for bpos, timer, owner in recs[k]["world"]["bombs"]:
            key = (tuple(bpos), owner)
            if owner != "rhine" and key not in known and key not in found:
                found[key] = (recs[k]["state"]["step"], set(blast_coords(field, bpos, cfg.BOMB_POWER)))
    return found


def _classify_selfkill(rec):
    """Returns (label, bomb_idx, flags). Priority: own_cause invalid > c2
    contested tile > new opponent bomb after placement > unclassified (the
    caller then inspects mask internals)."""
    recs, death = rec["steps"], rec["death"]
    last = len(recs) - 1
    bomb_idx = None
    for k in range(last, max(last - 10, -1), -1):
        if recs[k]["action"] == "BOMB":
            bomb_idx = k
            break
    if bomb_idx is None:
        return "no_own_bomb_in_last_10_steps", None, {}
    contested_steps = [
        recs[k]["state"]["step"] for k in range(bomb_idx, last)
        if recs[k + 1]["invalid_contested"] > recs[k]["invalid_contested"]
    ]
    own_steps = [
        recs[k]["state"]["step"] for k in range(bomb_idx, last)
        if recs[k + 1]["invalid_own"] > recs[k]["invalid_own"]
    ]
    if _final_step_blocked(recs, death):
        contested_steps.append(recs[last]["state"]["step"])
    death_pos = tuple(death["pos"])
    near_tiles = {death_pos} | {(death_pos[0] + dx, death_pos[1] + dy) for dx, dy in DIRS.values()}
    new_bombs = _new_opponent_bombs_after(recs, bomb_idx)
    new_bomb_hits = {key: v[0] for key, v in new_bombs.items() if v[1] & near_tiles}
    flags = {"contested_steps": contested_steps, "own_steps": own_steps, "new_opp_bombs_covering_death_area": new_bomb_hits}
    if own_steps:
        return "own_cause_invalid", bomb_idx, flags
    if contested_steps:
        return "c2_contested_tile", bomb_idx, flags
    if new_bomb_hits:
        return "opp_new_bomb_after_placement", bomb_idx, flags
    return "unclassified", bomb_idx, flags


def _siege_or_c1(rec, bomb_idx):
    """For cases not explained by a contested tile or a later opponent bomb:
    inspects the mask internals after the bomb was placed."""
    recs = rec["steps"]
    findings = []
    for k in range(bomb_idx + 1, len(recs)):
        step_info = _instrument_step(restore_state(recs[k]["state"]))
        if not step_info["current_tile_in_danger"]:
            continue
        detail = step_info["detail"]
        cands = {d: i for d, i in detail.items() if i["base_safe"]}
        all_not_robust = bool(cands) and all(i["robust"] is False for i in cands.values())
        findings.append((recs[k]["state"]["step"], step_info["legal_actions"], step_info["n_conflicting"],
                         len(cands), all_not_robust, recs[k]["action"]))
    return findings


def section_selfkill():
    out("\n=== (c) self-kill (group A) ===")
    dump_log = open(DATA_DIR / "selfkill_tail_dumps.txt", "w")
    labels = Counter()
    per_seed = {}
    for s in SEEDS:
        payload = load(s, "A")
        cases = [r for r in payload["records"] if r["metrics"]["self_kill"]]
        per_seed[s] = len(cases)
        out(f"\nseed{s}: self_kill_rate={payload['agg']['self_kill_rate']:.2f} ({len(cases)}/100)")
        for r in cases:
            label, bomb_idx, flags = _classify_selfkill(r)
            extra = f" flags={flags}"
            if label == "unclassified" and bomb_idx is not None:
                findings = _siege_or_c1(r, bomb_idx)
                siege = [f for f in findings if f[2] == 1 and f[3] >= 2 and f[4]]
                conflict_later = [f for f in findings if f[2] >= 1]
                if siege:
                    label = "c_new_single_opponent_siege"
                elif conflict_later:
                    label = "c1_later_leg_conflict"
                extra += f" findings(step,legal,N_conflict,n_cand,all_not_robust,action)={findings[-4:]}"
            labels[label] += 1
            recs = r["steps"]
            out(f"  round={r['round']:>3} steps={len(recs)} bomb_idx={bomb_idx} label={label} "
                f"last_actions={[x['action'] for x in recs[-6:]]} death_pos={r['death']['pos'] if r['death'] else None}"
                f"{extra}")
            tail = recs[-25:]
            step_log = [_instrument_step(restore_state(x["state"])) for x in tail]
            dump_log.write(f"\n{'=' * 20} seed{s} round={r['round']} label={label} {'=' * 20}\n")
            with contextlib.redirect_stdout(io.StringIO()):
                _dump_tail(dump_log, step_log, [x["action"] for x in tail], 20)
    dump_log.close()
    total = sum(labels.values())
    out(f"\nlabel totals: {dict(labels)} (total {total}); per-seed counts {per_seed}")
    if total:
        c2 = labels.get("c2_contested_tile", 0)
        out(f"c2 share = {c2}/{total} = {100 * c2 / total:.1f}%  (Stage C baseline 26/29 = 89.7%, "
            f"from the Stage C replay analysis)")
        out(f"c-class new single-opponent siege count = {labels.get('c_new_single_opponent_siege', 0)}; "
            f"c1 = {labels.get('c1_later_leg_conflict', 0)}; opp_new_bomb_after_placement = "
            f"{labels.get('opp_new_bomb_after_placement', 0)}")
    out("A/B-class (think-time forced WAIT) cannot occur: think-time limit disabled in this evaluation.")
    out(f"tail dumps for manual review: {DATA_DIR / 'selfkill_tail_dumps.txt'}")


# ---------------------------------------------------------------- kill behaviour
def _collect_kill_data(payload):
    opp_entries, bomb_steps, wasteful, n_steps = [], [], [], 0
    for r in payload["records"]:
        recs = r["steps"]
        n_steps += len(recs)
        kills_seq = [x["kills"] for x in recs] + [r["metrics"]["opponent_kills"]]
        steps_seq = [x["state"]["step"] for x in recs] + [recs[-1]["state"]["step"] + 1]
        names_seq = [set(o[0] for o in x["state"]["others"]) for x in recs]
        for i, x in enumerate(recs):
            st = x["state"]
            gs = None
            if x["action"] == "BOMB" or (st["self"][2] and st["others"]):
                gs = restore_state(st)
                sem = extract_semantic_state(gs)
            else:
                continue
            if x["action"] == "BOMB":
                wasteful.append(_bomb_wasteful_mismatch_check(sem, gs))
            if not is_opportunity(sem, x["base_mask"]):
                continue
            blast = set(blast_coords(sem.field_arr, sem.self_pos, cfg.BOMB_POWER))
            in_blast_names = {o[0] for o in st["others"] if tuple(o[3]) in blast}
            entry = {
                "round": r["round"], "step": st["step"], "chosen": x["action"],
                "f28": float(x["features"][F_KILL_VALUE]), "p_bomb": float(x["probs_base"][BOMB_IDX]),
                "has_kill_target": sem.has_kill_target, "nearest_kill_distance": sem.nearest_kill_distance,
                "bomb_legal": bool(x["base_mask"][BOMB_IDX]), "n_opp_in_blast": len(in_blast_names),
            }
            if x["action"] == "BOMB":
                window = [j for j in range(i + 1, len(steps_seq)) if steps_seq[j] - st["step"] <= KILL_CHECK_WINDOW]
                entry["hit_kill"] = any(kills_seq[j] > x["kills"] for j in window)
                entry["hit_vanish"] = any(j < len(names_seq) and bool(in_blast_names - names_seq[j]) for j in window)
                bomb_steps.append(entry)
            opp_entries.append(entry)
    return opp_entries, wasteful, n_steps


def _conditional_corr_groups(payload):
    """Populations for correlating feature #28 with P(BOMB): (1) kill
    opportunity but not bombed; (2) current tile is the kill target and BOMB
    is legal, bombed or not; plus the same-shape crate baseline."""
    kill_group, crate_group = [], []
    for r in payload["records"]:
        for x in r["steps"]:
            f, legal = x["features"], x["base_mask"][BOMB_IDX]
            if not legal:
                continue
            p = float(x["probs_base"][BOMB_IDX])
            if f[F_HAS_KILL] == 1 and f[F_KILL_DIST] == 0:
                kill_group.append((float(f[F_KILL_VALUE]), p))
            if f[F_HAS_BOMB_TARGET] == 1 and f[F_BOMB_DIST] == 0:
                crate_group.append((float(f[F_CRATES]), p))
    return kill_group, crate_group


def _first_layer_amplification(seed, feats):
    ckpt = torch.load(common.MODELS_DIR / f"task3_stage_d_seed{seed}.pt", map_location="cpu")
    w = ckpt["model_state_dict"]["trunk.0.weight"].numpy()
    l2 = np.linalg.norm(w, axis=0)
    std = feats.std(axis=0)
    trained = l2 * std
    baselines = []
    for k in range(20):
        torch.manual_seed(k)
        net = ActorCriticMLP(n_features=28, n_actions=6, hidden_sizes=cfg.MODEL_CONFIG.hidden_sizes)
        baselines.append(np.linalg.norm(net.trunk[0].weight.detach().numpy(), axis=0) * std)
    base = np.mean(baselines, axis=0)
    order = np.argsort(-trained)
    rank = {int(j): int(np.where(order == j)[0][0]) + 1 for j in (F_CRATES, F_KILL_VALUE)}
    return trained, base, rank


def section_kill():
    out("\n=== (d) kill behaviour (group A) ===")
    tot = defaultdict(list)
    for s in SEEDS:
        payload = load(s, "A")
        opp, wasteful, n_steps = _collect_kill_data(payload)
        bombed = [e for e in opp if e["chosen"] == "BOMB"]
        not_bombed = [e for e in opp if e["chosen"] != "BOMB"]
        n_opp = len(opp)
        hit_kill = sum(1 for e in bombed if e["hit_kill"])
        hit_vanish = sum(1 for e in bombed if e["hit_vanish"])
        cat_b = [e for e in not_bombed if not e["bomb_legal"] or not e["has_kill_target"] or e["nearest_kill_distance"] != 0]
        cat_b_other_target = [e for e in cat_b if e["bomb_legal"] and e["has_kill_target"] and e["nearest_kill_distance"] != 0]
        out(f"\nseed{s}: kills/round={payload['agg']['opponent_kills_mean']:.2f} bombs_placed/round="
            f"{payload['agg']['bombs_dropped_mean']:.1f}")
        out(f"  opportunities={n_opp} ({n_opp / 100:.2f}/round, {1000 * n_opp / n_steps:.1f} per 1000 steps); "
            f"BOMB chosen={len(bombed)} ({100 * len(bombed) / n_opp if n_opp else float('nan'):.1f}%)")
        out(f"  hit rate (rhine kill within {KILL_CHECK_WINDOW} ticks)={hit_kill}/{len(bombed)} "
            f"({100 * hit_kill / len(bombed) if bombed else float('nan'):.1f}%); looser 'in-blast opponent vanished'="
            f"{hit_vanish}/{len(bombed)} ({100 * hit_vanish / len(bombed) if bombed else float('nan'):.1f}%)")
        out(f"  category (b) among not-bombed: {len(cat_b)}/{len(not_bombed)} "
            f"(of which kill target elsewhere, i.e. current tile qualifies but a higher-scoring target is nearer/elsewhere: "
            f"{len(cat_b_other_target)}; no kill target or mask-illegal: {len(cat_b) - len(cat_b_other_target)})")
        nb_v = np.array([e["f28"] for e in not_bombed]); nb_p = np.array([e["p_bomb"] for e in not_bombed])
        r_nb = _corr(nb_v, nb_p)
        hi, lo = nb_p[nb_v >= 0.75], nb_p[nb_v < 0.25]
        out(f"  not-bombed: n={len(not_bombed)} corr(#28,P(BOMB))={r_nb:.3f}  mean P(BOMB) value>=0.75: "
            f"{hi.mean() if len(hi) else float('nan'):.3f} (n={len(hi)}), value<0.25: {lo.mean() if len(lo) else float('nan'):.3f} (n={len(lo)}); "
            f"median P(BOMB)={np.median(nb_p) if len(nb_p) else float('nan'):.4f}")
        out(f"  #28 mean: not-bombed={nb_v.mean() if len(nb_v) else float('nan'):.3f}, "
            f"bombed={np.mean([e['f28'] for e in bombed]) if bombed else float('nan'):.3f}, "
            f"bombed&hit={np.mean([e['f28'] for e in bombed if e['hit_kill']]) if hit_kill else float('nan'):.3f}")
        kg, cg = _conditional_corr_groups(payload)
        r_kill = _corr([a for a, _ in kg], [b for _, b in kg])
        r_crate = _corr([a for a, _ in cg], [b for _, b in cg])
        out(f"  wider group (current tile is target & BOMB legal): kill corr(#28,P(BOMB))={r_kill:.3f} (n={len(kg)}); "
            f"crate baseline corr(#18,P(BOMB))={r_crate:.3f} (n={len(cg)})")
        thr = [w for w in wasteful if w["threatens_reachable_opponent"]]
        low = [w for w in thr if w["difficulties"] and max(w["difficulties"].values()) < WASTEFUL_MISMATCH_THRESHOLD]
        out(f"  wasteful-penalty mismatch: BOMB steps={len(wasteful)} threatens_opponent={len(thr)} "
            f"low-difficulty(<{WASTEFUL_MISMATCH_THRESHOLD})={len(low)} ({100 * len(low) / len(thr) if thr else float('nan'):.1f}%)")
        all_feats = np.stack([x["features"] for r in payload["records"] for x in r["steps"]])
        trained, base, rank = _first_layer_amplification(s, all_feats)
        out(f"  first-layer L2*std: #18 crates={trained[F_CRATES]:.3f} (x{trained[F_CRATES] / base[F_CRATES]:.1f} vs random init, rank {rank[F_CRATES]}/28); "
            f"#28 kill_value={trained[F_KILL_VALUE]:.3f} (x{trained[F_KILL_VALUE] / base[F_KILL_VALUE]:.1f}, rank {rank[F_KILL_VALUE]}/28); "
            f"std(#28)={all_feats[:, F_KILL_VALUE].std():.3f} std(#18)={all_feats[:, F_CRATES].std():.3f}")
        for key, val in (("opp", n_opp), ("bombed", len(bombed)), ("hit", hit_kill), ("vanish", hit_vanish),
                         ("steps", n_steps), ("thr", len(thr)), ("low", len(low))):
            tot[key].append(val)
        tot["r_nb"].append(r_nb); tot["r_kill"].append(r_kill); tot["r_crate"].append(r_crate)
    n_opp, n_bombed = sum(tot["opp"]), sum(tot["bombed"])
    out(f"\nALL 3 seeds: opportunities={n_opp} ({n_opp / 300:.2f}/round), initiative={n_bombed}/{n_opp} "
        f"({100 * n_bombed / n_opp:.1f}%), hit={sum(tot['hit'])}/{n_bombed} ({100 * sum(tot['hit']) / n_bombed:.1f}%), "
        f"vanish-variant={sum(tot['vanish'])}/{n_bombed} ({100 * sum(tot['vanish']) / n_bombed:.1f}%)")
    out(f"corr(#28,P(BOMB)) not-bombed group by seed: {[round(v, 3) for v in tot['r_nb']]}  "
        f"wider-group kill: {[round(v, 3) for v in tot['r_kill']]}  crate baseline: {[round(v, 3) for v in tot['r_crate']]}")
    out("Stage C reference: 27-dim: 4.86 opp/round, initiative 23.3%, hit 10.6%; v2 (34-dim, sum): "
        "11.1 opp/round, initiative 12.0%, hit 9.3%; v3 not-bombed corr 0.140/-0.031/0.102; v3 wider-group kill corr "
        "0.121/0.138/0.288 (crate 0.150/0.274/0.218). Feature set 34->28 and opponents 1->3 differ.")


# ---------------------------------------------------------------- kill behaviour, decomposed
def _fast_opportunities(payload):
    """Same population as diagnose_kill_target_discrimination.is_opportunity()
    (bomb available, an opponent inside the own blast, BOMB legal), read
    directly from the recorded observation without full semantic extraction."""
    entries = []
    for r in payload["records"]:
        recs = r["steps"]
        kills_seq = [x["kills"] for x in recs] + [r["metrics"]["opponent_kills"]]
        steps_seq = [x["state"]["step"] for x in recs] + [recs[-1]["state"]["step"] + 1]
        for i, x in enumerate(recs):
            st = x["state"]
            if not st["self"][2] or not st["others"] or not x["base_mask"][BOMB_IDX]:
                continue
            field = restore_state(st)["field"]
            pos = tuple(int(v) for v in st["self"][3])
            blast = set(blast_coords(field, pos, cfg.BOMB_POWER))
            n_in_blast = sum(1 for o in st["others"] if tuple(int(v) for v in o[3]) in blast)
            if not n_in_blast:
                continue
            entry = {
                "chosen": x["action"], "f28": float(x["features"][F_KILL_VALUE]),
                "p_bomb": float(x["probs_base"][BOMB_IDX]), "crates": int(crates_in_blast(field, pos, cfg.BOMB_POWER)),
                "n_in_blast": n_in_blast, "has_kill_target": bool(x["features"][F_HAS_KILL]),
                "dist0": bool(x["features"][F_KILL_DIST] == 0),
            }
            if x["action"] == "BOMB":
                window = [j for j in range(i + 1, len(steps_seq)) if steps_seq[j] - st["step"] <= KILL_CHECK_WINDOW]
                entry["hit"] = any(kills_seq[j] > x["kills"] for j in window)
            entries.append(entry)
    return entries


def _rate(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def section_killextra():
    out("\n=== (d, decomposed) opportunities split by crates in the blast (group A) ===")
    out("kill-only = no crate in the blast (a bomb there can only be an attack or wasted); "
        "crate+kill = the bomb would also clear crates, so choosing BOMB is not evidence of hunting intent.")
    pooled = []
    for s in SEEDS:
        payload = load(s, "A")
        entries = _fast_opportunities(payload)
        pooled += entries
        coins = payload["agg"]["coins_collected_mean"]
        kills = payload["agg"]["opponent_kills_mean"]
        out(f"\nseed{s}: opportunities={len(entries)} (fast path; full-semantic count above was authoritative) | "
            f"coins/round={coins:.2f} kills/round={kills:.2f} -> score share from kills={100 * 5 * kills / (coins + 5 * kills):.0f}%")
        for name, sub in (("kill-only (crates=0)", [e for e in entries if e["crates"] == 0]),
                          ("crate+kill (crates>0)", [e for e in entries if e["crates"] > 0])):
            bombed = [e for e in sub if e["chosen"] == "BOMB"]
            out(f"  {name}: n={len(sub)} BOMB chosen={_rate(len(bombed), len(sub))} "
                f"hit={_rate(sum(1 for e in bombed if e['hit']), len(bombed))}")
    out("\nPOOLED over 3 seeds:")
    for name, sub in (("kill-only (crates=0)", [e for e in pooled if e["crates"] == 0]),
                      ("crate+kill (crates>0)", [e for e in pooled if e["crates"] > 0])):
        bombed = [e for e in sub if e["chosen"] == "BOMB"]
        out(f"  {name}: n={len(sub)} ({len(sub) / 300:.1f}/round) BOMB chosen={_rate(len(bombed), len(sub))} "
            f"hit={_rate(sum(1 for e in bombed if e['hit']), len(bombed))}")
    out("\nBy #28 (expected_kill_value_at_target) bucket, pooled: BOMB rate over all opportunities, hit rate over bombed")
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sub = [e for e in pooled if lo <= e["f28"] < hi]
        bombed = [e for e in sub if e["chosen"] == "BOMB"]
        out(f"  [{lo:.2f},{min(hi, 1.0):.2f}): n={len(sub)} BOMB={_rate(len(bombed), len(sub))} "
            f"hit={_rate(sum(1 for e in bombed if e['hit']), len(bombed))} mean P(BOMB)={np.mean([e['p_bomb'] for e in sub]) if sub else float('nan'):.3f}")
    out("\nSame buckets, kill-only opportunities only:")
    ko = [e for e in pooled if e["crates"] == 0]
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sub = [e for e in ko if lo <= e["f28"] < hi]
        bombed = [e for e in sub if e["chosen"] == "BOMB"]
        out(f"  [{lo:.2f},{min(hi, 1.0):.2f}): n={len(sub)} BOMB={_rate(len(bombed), len(sub))} "
            f"hit={_rate(sum(1 for e in bombed if e['hit']), len(bombed))}")
    out("\nFirst-layer L2*std amplification vs random init, kill-group features (1-based #22-#28) and #18:")
    for s in SEEDS:
        payload = load(s, "A")
        feats = np.stack([x["features"] for r in payload["records"] for x in r["steps"]])
        trained, base, rank = _first_layer_amplification(s, feats)
        cells = ", ".join(f"#{j + 1}=x{trained[j] / base[j]:.1f}" for j in (17, 21, 22, 23, 24, 25, 26, 27))
        out(f"  seed{s}: {cells}")


# ---------------------------------------------------------------- got killed
def section_gotkilled():
    out("\n=== (e) got_killed_by_opponent (group A) ===")
    n_cases = 0
    dump_log = open(DATA_DIR / "gotkilled_tail_dumps.txt", "w")
    for s in SEEDS:
        payload = load(s, "A")
        cases = [r for r in payload["records"] if r["metrics"]["got_killed_by_opponent"]]
        out(f"\nseed{s}: got_killed_rate={payload['agg']['got_killed_by_opponent_rate']:.2f} ({len(cases)}/100)")
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
            in_danger = [bool(x["features"][18]) for x in recs[-8:]]
            legal_moves = [len([a for a in ("UP", "DOWN", "LEFT", "RIGHT") if x["final_mask"][cfg.ACTIONS.index(a)]]) for x in recs[-8:]]
            near_opps = sum(1 for o in recs[-1]["state"]["others"]
                            if abs(o[3][0] - pos[0]) + abs(o[3][1] - pos[1]) <= 6)
            out(f"  round={r['round']:>3} step={death['step']} pos={pos} killer_owners={killers} "
                f"bombs_covering_pos_at_last_act={covering} n_distinct_owners={len({c[0] for c in covering})} "
                f"opponents_alive={len(recs[-1]['state']['others'])} within6={near_opps} "
                f"in_danger_last8={in_danger} legal_moves_last8={legal_moves} last_actions={[x['action'] for x in recs[-6:]]} "
                f"rhine_bomb_in_last6={'BOMB' in [x['action'] for x in recs[-6:]]} "
                f"opponent_bombs_on_board={n_opp_bombs} contested_tile_invalid_in_last5={contested_steps}")
            tail = recs[-25:]
            step_log = [_instrument_step(restore_state(x["state"])) for x in tail]
            dump_log.write(f"\n{'=' * 20} seed{s} round={r['round']} killers={killers} {'=' * 20}\n")
            with contextlib.redirect_stdout(io.StringIO()):
                _dump_tail(dump_log, step_log, [x["action"] for x in tail], 20)
    dump_log.close()
    if n_cases == 0:
        out("\nNo got_killed_by_opponent case in group A: nothing to classify.")


def main():
    global DATA_DIR, _out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["tables", "invalid", "oscillation", "selfkill", "kill", "killextra", "gotkilled", "all"])
    parser.add_argument("--data-dir", default=str(common.LOGS_DIR / "stage_d_final"))
    parser.add_argument("--allow-partial", action="store_true", help="testing only: read unfinished recordings")
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    global ALLOW_PARTIAL
    ALLOW_PARTIAL = args.allow_partial
    _out = open(DATA_DIR / f"analysis_{args.section}.txt", "w")
    sections = {
        "tables": section_tables, "invalid": section_invalid, "oscillation": section_oscillation,
        "selfkill": section_selfkill, "kill": section_kill, "killextra": section_killextra,
        "gotkilled": section_gotkilled,
    }
    for name in (sections if args.section == "all" else [args.section]):
        sections[name]()
    _out.close()


if __name__ == "__main__":
    main()
