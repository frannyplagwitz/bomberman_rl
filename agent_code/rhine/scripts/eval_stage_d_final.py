"""Stage D final evaluation with full per-step recording, so follow-up
investigations run offline on the recorded data instead of re-running games.

One process = one (checkpoint, breaker group). The opponent-RNG pin (same
helper the kill/self-kill diagnostics use) seeds opponent RNGs once per run;
on its own it does not make runs reproducible across launches. Pass
--reproducible for step-exact reproducibility (see _reseed_round()). Per step
it records the raw observation given to act(), the chosen action, feature
vector, base/final masks, masked action probabilities, and a world snapshot
(bomb owners, explosion owners, alive agents); per round it also records
every agent's final score. Outputs a gzip pickle plus a text log; nothing
under models/ is touched.

Usage:
  python -m agent_code.rhine.scripts.eval_stage_d_final \
      --checkpoint <abs path> --group A|B --out-dir <dir> [--n-rounds 100]
Group A = oscillation breaker on, group B = breaker off. The no-bomb-when-
board-cleared mask is on by default (Stage D training config); see
--enable-no-bomb-when-cleared.
"""
import argparse
import dataclasses
import gzip
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.distributions import Categorical

from agent_code.rhine import callbacks as rhine_callbacks
from agent_code.rhine import config as cfg
from agent_code.rhine.model import masked_logits
from agent_code.rhine.scripts import common
from agent_code.rhine.scripts.diagnose_oscillation import OSCILLATION_THRESHOLD, longest_oscillation_run
from agent_code.rhine.scripts.diagnose_selfkill_bfs_trace import _pin_opponent_reseed

SCENARIO = "classic"
OPPONENTS = ["coin_collector_agent"] * 3
# Hash randomization would otherwise make any set/dict iteration over strings
# differ between launches.
REPRODUCIBLE_HASH_SEED = "0"


def _round_seed(eval_seed: int, round_index: int) -> int:
    return eval_seed * 1000 + round_index


def _reseed_round(np_seed_fn, eval_seed: int, round_index: int):
    """Reseeds the two global RNGs framework opponents draw from (Python
    `random` for shuffle(), numpy's legacy global state) at the start of
    every round, so a round's opponent behavior depends only on
    (eval_seed, round_index) and not on how many draws earlier rounds made.
    `np_seed_fn` must be the unpatched np.random.seed, since
    _pin_opponent_reseed() redirects the module attribute.
    """
    seed = _round_seed(eval_seed, round_index)
    random.seed(seed)
    np_seed_fn(seed)


def _disable_think_time_limit_all_agents(world):
    """Like common._disable_think_time_limit(), but for every agent: an
    opponent forced to WAIT by a timing overrun under CPU load would make
    the run non-reproducible."""
    for a in world.agents:
        a.available_think_time = float("inf")


def compact_state(state: dict) -> dict:
    out = dict(state)
    out["field"] = np.asarray(state["field"], dtype=np.int8)
    out["explosion_map"] = np.asarray(state["explosion_map"], dtype=np.int8)
    return out


def restore_state(compact: dict) -> dict:
    out = dict(compact)
    out["field"] = np.asarray(compact["field"], dtype=np.int64)
    out["explosion_map"] = np.asarray(compact["explosion_map"], dtype=np.float64)
    return out


def _probs(model, features, mask):
    state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(np.asarray(mask, dtype=bool)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        return Categorical(logits=masked_logits(logits, mask_t)).probs.squeeze(0).numpy().astype(np.float32)


def _snapshot(world):
    return {
        "bombs": [((int(b.x), int(b.y)), int(b.timer), b.owner.name) for b in world.bombs],
        "expl": [
            (ex.owner.name, [(int(x), int(y)) for x, y in ex.blast_coords], int(ex.timer))
            for ex in world.explosions
        ],
        "agents": [(a.name, int(a.x), int(a.y)) for a in world.active_agents],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--group", choices=["A", "B"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument(
        "--opponents", nargs="+", default=None,
        help="Opponent directory names, repeatable (e.g. rule_based_agent rule_based_agent "
             "rule_based_agent). Defaults to Stage D's 3x coin_collector_agent.",
    )
    parser.add_argument("--tag", default=None, help="Output filename stem override (default: checkpoint stem).")
    parser.add_argument(
        "--enable-deadlock-bomb", action="store_true", default=False,
        help="Sets cfg.ENABLE_DEADLOCK_BOMB=True for this run. Only takes effect together with "
             "--group A (breaker on); with --group B it has no effect (a warning is logged by "
             "callbacks.setup()). See action_mask.should_force_deadlock_bomb().",
    )
    parser.add_argument(
        "--enable-no-bomb-when-cleared", action=argparse.BooleanOptionalAction, default=True,
        help="Sets cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED. Default True matches every Stage D "
             "checkpoint's actual training+evaluation config; "
             "pass --no-enable-no-bomb-when-cleared to evaluate under the config default (False) "
             "instead, e.g. for a task whose training run kept this switch off "
             "(Task 4).",
    )
    parser.add_argument(
        "--disable-think-time-limit", action=argparse.BooleanOptionalAction, default=False,
        help="Overrides the framework's real 0.5s per-step think-time budget to unlimited (see "
             "common._disable_think_time_limit()'s docstring) -- diagnostic-only, never exercised by "
             "the official submission path. Default False: every step is subject to the real budget, "
             "matching the timing rule (real-timing runs are the only ones that count "
             "as formal report data). Pass --disable-think-time-limit to reproduce every prior Task 3/4 "
             "evaluation's behavior (all of which disabled the limit).",
    )
    parser.add_argument(
        "--reproducible", action=argparse.BooleanOptionalAction, default=False,
        help="Step-exact reproducible run: reseeds opponent RNGs per round from (eval_seed, round), "
             "fixes PYTHONHASHSEED (re-executes the process if needed) and torch to one thread. "
             "Requires --disable-think-time-limit, which is then applied to every agent.",
    )
    args = parser.parse_args()
    if args.reproducible:
        if not args.disable_think_time_limit:
            parser.error("--reproducible requires --disable-think-time-limit")
        if os.environ.get("PYTHONHASHSEED") != REPRODUCIBLE_HASH_SEED:
            os.environ["PYTHONHASHSEED"] = REPRODUCIBLE_HASH_SEED
            os.execv(sys.executable, [sys.executable] + sys.argv)
        torch.set_num_threads(1)
    opponents = args.opponents if args.opponents else OPPONENTS

    cfg.ENABLE_OSCILLATION_BREAKER = args.group == "A"
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = args.enable_no_bomb_when_cleared
    cfg.ENABLE_DEADLOCK_BOMB = args.enable_deadlock_bomb

    tag = f"{args.tag or Path(args.checkpoint).stem}_{args.group}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.pkl.gz"
    log_file, log_path = common.open_log_file(f"stage_d_final_eval_{tag}")
    common.log_print(log_file, f"Stage D final eval -- log: {log_path}")
    common.log_print(
        log_file,
        f"checkpoint={args.checkpoint} group={args.group} breaker={cfg.ENABLE_OSCILLATION_BREAKER} "
        f"no_bomb_when_cleared={cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED} "
        f"deadlock_bomb={cfg.ENABLE_DEADLOCK_BOMB} n_rounds={args.n_rounds} "
        f"eval_seed={args.eval_seed} opponents={opponents} out={out_path} "
        f"disable_think_time_limit={args.disable_think_time_limit}",
    )

    common._set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", args.checkpoint)
    common._set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    common._set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    common._set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    holder = {}
    round_steps = []
    world_ref = {}

    orig = {
        "act": rhine_callbacks.act,
        "extract": rhine_callbacks.extract_semantic_state,
        "features": rhine_callbacks.features_from_semantic,
        "mask": rhine_callbacks.mask_from_semantic,
        "select": rhine_callbacks.select_action,
    }

    def cap_extract(game_state):
        semantic = orig["extract"](game_state)
        holder["sem"] = semantic
        return semantic

    def cap_features(semantic, *a, **kw):
        features = orig["features"](semantic, *a, **kw)
        holder["features"] = np.array(features, dtype=np.float32)
        return features

    def cap_mask(semantic):
        mask = orig["mask"](semantic)
        holder["base_mask"] = np.array(mask, dtype=bool)
        return mask

    def cap_select(model, features, mask, deterministic=False):
        holder["final_mask"] = np.array(mask, dtype=bool)
        holder["probs_final"] = _probs(model, features, mask)
        return orig["select"](model, features, mask, deterministic=deterministic)

    def instrumented_act(self, game_state):
        holder.clear()
        chosen = orig["act"](self, game_state)
        if game_state is not None and "features" in holder:
            world = world_ref["world"]
            round_steps.append({
                "step": game_state["step"],
                "action": chosen,
                "state": compact_state(game_state),
                "features": holder["features"],
                "base_mask": holder["base_mask"],
                "final_mask": holder["final_mask"],
                "probs_base": _probs(self.model, holder["features"], holder["base_mask"]),
                "probs_final": holder["probs_final"],
                "world": _snapshot(world),
                "kills": world.agents[0].statistics.get("kills", 0),
                "invalid_own": self.invalid_action_own_cause_count,
                "invalid_contested": self.invalid_action_contested_tile_count,
                "deadlock_bomb_triggered": bool(getattr(self, "last_deadlock_bomb_triggered", False)),
            })
        return chosen

    rhine_callbacks.extract_semantic_state = cap_extract
    rhine_callbacks.features_from_semantic = cap_features
    rhine_callbacks.mask_from_semantic = cap_mask
    rhine_callbacks.select_action = cap_select
    rhine_callbacks.act = instrumented_act

    episodes = []
    records = []
    t_start = time.time()

    def save(complete):
        agg = common.aggregate_metrics(episodes) if episodes else {}
        if episodes:
            n_sustained = sum(
                1 for ep in episodes if longest_oscillation_run(ep.actions)[0] >= OSCILLATION_THRESHOLD
            )
            agg["oscillation_fraction"] = n_sustained / len(episodes)
        payload = {
            "meta": {
                "checkpoint": args.checkpoint, "group": args.group, "eval_seed": args.eval_seed,
                "n_rounds": args.n_rounds, "opponents": opponents, "complete": complete,
                "breaker": cfg.ENABLE_OSCILLATION_BREAKER,
                "no_bomb_when_cleared": cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED,
                "deadlock_bomb": cfg.ENABLE_DEADLOCK_BOMB,
                "disable_think_time_limit": args.disable_think_time_limit,
                "reproducible": args.reproducible,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            },
            "agg": agg, "records": records,
        }
        tmp = out_path.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out_path)

    try:
        np_seed_fn = np.random.seed
        with _pin_opponent_reseed(args.eval_seed):
            world = common.build_world(scenario=SCENARIO, seed=args.eval_seed, train=False, opponents=opponents)
            world_ref["world"] = world
            agent = world.agents[0]

            for _ in range(args.n_rounds):
                world.new_round()
                if args.reproducible:
                    _reseed_round(np_seed_fn, args.eval_seed, world.round)
                world.user_input = "WAIT"
                round_steps.clear()
                death = None
                round_t0 = time.time()
                while world.running:
                    if args.reproducible:
                        _disable_think_time_limit_all_agents(world)
                    elif args.disable_think_time_limit:
                        common._disable_think_time_limit(world)
                    world.do_step("WAIT")
                    if agent.dead and death is None:
                        death = {"step": world.step, "pos": (int(agent.x), int(agent.y)), **_snapshot(world)}

                actions = list(world.replay["actions"][agent.name])
                own, contested = common._invalid_action_breakdown(agent)
                self_kill = agent.statistics.get("suicides", 0) > 0
                episode = common.EpisodeMetrics(
                    round_index=world.round, steps=world.step,
                    coins_collected=agent.statistics.get("coins", 0),
                    completed=common.default_success_fn(world),
                    invalid_action_count=agent.statistics.get("invalid", 0),
                    invalid_action_own_cause_count=own, invalid_action_contested_tile_count=contested,
                    wait_count=actions.count("WAIT"),
                    immediate_reverse_count=sum(
                        1 for p, c in zip(actions, actions[1:]) if common.OPPOSITE_DIRECTION.get(p) == c
                    ),
                    crates_destroyed=agent.statistics.get("crates", 0),
                    bombs_dropped=agent.statistics.get("bombs", 0),
                    self_kill=self_kill, opponent_kills=agent.statistics.get("kills", 0),
                    got_killed_by_opponent=agent.dead and not self_kill, actions=actions,
                )
                episodes.append(episode)
                records.append({
                    "round": world.round, "metrics": dataclasses.asdict(episode),
                    "steps": list(round_steps), "death": death, "round_wall_seconds": time.time() - round_t0,
                    "final_scores": [(a.name, a.code_name, int(a.score)) for a in world.agents],
                })
                common.log_print(
                    log_file,
                    f"round {len(episodes)}/{args.n_rounds}: steps={episode.steps} coins={episode.coins_collected} "
                    f"kills={episode.opponent_kills} self_kill={episode.self_kill} "
                    f"got_killed={episode.got_killed_by_opponent} invalid(own/contested)="
                    f"{own}/{contested} elapsed={time.time() - t_start:.0f}s",
                )
                if len(episodes) % 20 == 0 and len(episodes) < args.n_rounds:
                    save(complete=False)
            world.end()
    finally:
        for name, key in [("act", "act"), ("extract_semantic_state", "extract"),
                          ("features_from_semantic", "features"), ("mask_from_semantic", "mask"),
                          ("select_action", "select")]:
            setattr(rhine_callbacks, name, orig[key])

    save(complete=True)
    agg = common.aggregate_metrics(episodes)
    if cfg.ENABLE_DEADLOCK_BOMB:
        n_triggered = sum(
            1 for rec in records for step in rec["steps"] if step.get("deadlock_bomb_triggered")
        )
        common.log_print(log_file, f"\ndeadlock_bomb_triggered steps: {n_triggered}")
    common.log_print(log_file, f"\nDONE {tag}: rounds={len(episodes)} elapsed={time.time() - t_start:.0f}s")
    for key in ("score_mean", "score_total", "opponent_kills_mean", "got_killed_by_opponent_rate",
                "self_kill_rate", "completion_rate", "episode_length_mean", "steps_to_completion_mean",
                "invalid_action_own_cause_count_total", "invalid_action_contested_tile_count_total"):
        common.log_print(log_file, f"  {key}={agg[key]}")
    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
