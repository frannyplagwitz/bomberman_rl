"""Evaluate one agent against three rule_based_agent opponents.

Runs N rounds of the classic scenario with the standard framework
(environment.py / agents.py / settings.py) and the framework's default
think-time limit, then writes per-round details and aggregate metrics.

Opponent randomness is pinned from outside the opponent code: numpy's
np.random.seed() (called without arguments by rule_based_agent.setup())
is redirected to the eval seed, and Python's `random` plus numpy's global
RNG are reseeded at the start of every round from (seed, round). Hash
randomization is fixed by re-executing with PYTHONHASHSEED=0. The map
layout comes from the world's own RNG, seeded with the eval seed.

Usage (from the framework root):
  python eval_kit/evaluate.py --agent my_agent [--n-rounds 100] [--seed 1000]
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HASH_SEED = "0"
SCENARIO = "classic"
OPPONENT = "rule_based_agent"
N_OPPONENTS = 3
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
# A round counts as oscillating if its longest back-and-forth run reaches this many actions.
OSCILLATION_THRESHOLD = 10


def find_framework_root(start: Path) -> Path:
    """Returns the nearest directory at or above `start` that contains
    environment.py and agent_code/. Exits if none is found."""
    for d in [start, *start.parents]:
        if (d / "environment.py").is_file() and (d / "agent_code").is_dir():
            return d
    sys.exit(f"Could not find the framework root (environment.py + agent_code/) above {start}. "
             f"Pass --framework-root.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an agent against 3x rule_based_agent.")
    parser.add_argument("--agent", required=True,
                        help="Agent folder: a name under agent_code/ or a path to a folder inside agent_code/.")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: eval_kit/results/<agent>_seed<seed>_n<rounds>).")
    parser.add_argument("--framework-root", default=None,
                        help="Framework root (default: searched upward from this script's folder).")
    return parser.parse_args()


@contextmanager
def pinned_opponent_seeding(np_module, seed: int):
    """Redirects every np.random.seed(...) call to `seed` while active and
    seeds Python's `random` once up front. Restores both on exit."""
    original_seed = np_module.random.seed
    original_state = random.getstate()
    np_module.random.seed = lambda *a, **kw: original_seed(seed)
    random.seed(seed)
    try:
        yield original_seed
    finally:
        np_module.random.seed = original_seed
        random.setstate(original_state)


def longest_oscillation_run(actions) -> int:
    """Length (in actions) of the longest run alternating between two
    opposite moves, e.g. LEFT RIGHT LEFT RIGHT. Returns 0 for fewer than
    3 actions."""
    if len(actions) < 3:
        return 0
    best, run_start = 0, 0
    for i in range(2, len(actions)):
        if not (actions[i] == actions[i - 2] and OPPOSITE.get(actions[i - 1]) == actions[i]):
            best = max(best, i - 1 - run_start)
            run_start = i - 1
    return max(best, len(actions) - run_start)


def make_time_to_stop(s):
    """Round-end rule used for these evaluations: identical to the
    framework's, except that "one agent left and board cleared" only ends
    the round once every opponent of the evaluated agent (agents[0]) is dead.
    Scores are unaffected; it keeps round lengths comparable across runs."""
    def time_to_stop(self):
        if len(self.active_agents) == 0:
            return True
        evaluated = self.agents[0]
        opponents_alive = any(a is not evaluated for a in self.active_agents)
        if (not opponents_alive
                and len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all(not c.collectable for c in self.coins)
                and len(self.bombs) + len(self.explosions) == 0):
            return True
        if any(a.train for a in self.agents) and not self.args.continue_without_training:
            if not any(a.train for a in self.active_agents):
                return True
        return self.step >= s.MAX_STEPS
    return time_to_stop


def instrument_think_time(agent, tracker: dict):
    """Wraps agent.wait_for_act to record each act() duration and whether
    it exceeded the time the framework allowed for that step. Does not
    change the returned action or timing."""
    original = agent.wait_for_act

    def wait_for_act():
        allowed = agent.available_think_time
        action, think_time = original()
        tracker["times"].append(think_time)
        if think_time > allowed:
            tracker["timeouts"] += 1
        return action, think_time

    agent.wait_for_act = wait_for_act


def rank_of_first(scores) -> float:
    """Rank of scores[0] among all scores (1 = best); ties share the
    average of their rank positions."""
    own = scores[0]
    higher = sum(1 for x in scores[1:] if x > own)
    equal = sum(1 for x in scores[1:] if x == own)
    return higher + 1 + equal / 2.0


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    """Sample standard deviation (ddof=1); nan for fewer than 2 values."""
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def median(values):
    if not values:
        return float("nan")
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def summarize(rounds: list, n_coins: int) -> dict:
    """Aggregate metrics over all rounds. Rates are fractions of rounds."""
    n = len(rounds)
    scores = [r["score"] for r in rounds]
    completed = [r for r in rounds if r["completed"]]
    opp_scores = [r["opponent_scores"] for r in rounds]
    best_opp = [max(o) for o in opp_scores]
    act_times = [t for r in rounds for t in r["_act_times"]]
    return {
        "n_rounds": n,
        "score_mean": mean(scores),
        "score_std": std(scores),
        "score_se": std(scores) / n ** 0.5 if n > 1 else float("nan"),
        "score_total": sum(scores),
        "coins_collected_mean": mean([r["coins"] for r in rounds]),
        "opponent_kills_mean": mean([r["kills"] for r in rounds]),
        "self_kill_rate": mean([float(r["self_kill"]) for r in rounds]),
        "got_killed_by_opponent_rate": mean([float(r["got_killed_by_opponent"]) for r in rounds]),
        "survival_rate": mean([float(not r["dead"]) for r in rounds]),
        "completion_rate": len(completed) / n,
        "steps_to_completion_mean": mean([r["steps"] for r in completed]),
        "steps_to_completion_median": median([r["steps"] for r in completed]),
        "episode_length_mean": mean([r["steps"] for r in rounds]),
        "steps_per_coin_mean": mean([r["steps"] / r["coins"] for r in rounds if r["coins"] > 0]),
        "wait_fraction_mean": mean([r["wait_count"] / r["steps"] for r in rounds if r["steps"] > 0]),
        "invalid_action_count_total": sum(r["invalid_actions"] for r in rounds),
        "immediate_reverse_count_mean": mean([r["immediate_reverses"] for r in rounds]),
        "crates_destroyed_mean": mean([r["crates_destroyed"] for r in rounds]),
        "bombs_dropped_mean": mean([r["bombs_dropped"] for r in rounds]),
        "bombs_placed_mean": mean([r["bomb_actions"] for r in rounds]),
        "oscillation_fraction": mean([float(r["longest_oscillation"] >= OSCILLATION_THRESHOLD) for r in rounds]),
        "opponent_score_mean": mean([x for o in opp_scores for x in o]),
        "best_opponent_score_mean": mean(best_opp),
        "strict_first_rate": mean([float(r["score"] > b) for r, b in zip(rounds, best_opp)]),
        "shared_first_rate": mean([float(r["score"] >= b) for r, b in zip(rounds, best_opp)]),
        "mean_rank": mean([r["rank"] for r in rounds]),
        "margin_vs_best_opponent_mean": mean([r["score"] - b for r, b in zip(rounds, best_opp)]),
        "think_time_exceeded_total": sum(r["think_time_exceeded"] for r in rounds),
        "skipped_steps_total": sum(r["skipped_steps"] for r in rounds),
        "opponent_think_time_exceeded_total": sum(sum(r["opponent_think_time_exceeded"]) for r in rounds),
        "act_time_mean_ms": 1000 * mean(act_times),
        "act_time_max_ms": 1000 * max(act_times) if act_times else float("nan"),
        "expected_coins_per_round": n_coins,
    }


def main():
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        os.environ["PYTHONHASHSEED"] = HASH_SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)

    kit_dir = Path(__file__).resolve().parent
    root = Path(args.framework_root).resolve() if args.framework_root else find_framework_root(kit_dir)
    agent_path = Path(args.agent)
    agent_name = agent_path.resolve().name
    if len(agent_path.parts) > 1 and agent_path.resolve().parent != root / "agent_code":
        sys.exit(f"{agent_path} is not directly inside {root / 'agent_code'}.")
    if not (root / "agent_code" / agent_name / "callbacks.py").is_file():
        sys.exit(f"{root / 'agent_code' / agent_name} has no callbacks.py; "
                 f"the agent folder must be inside {root / 'agent_code'}.")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        kit_dir / "results" / f"{agent_name}_seed{args.seed}_n{args.n_rounds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Framework modules use paths relative to the framework root.
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import settings as s
    from environment import BombeRLeWorld, WorldArgs

    n_coins = s.SCENARIOS[SCENARIO]["COIN_COUNT"]
    world_args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=False, log_dir=str(out_dir), save_stats=False,
        match_name="eval_kit", seed=args.seed, silence_errors=False, scenario=SCENARIO,
    )
    print(f"agent={agent_name} n_rounds={args.n_rounds} seed={args.seed} "
          f"think_time_limit={s.TIMEOUT}s out={out_dir}", flush=True)

    rounds = []
    t_start = time.time()
    with pinned_opponent_seeding(np, args.seed) as np_seed:
        world = BombeRLeWorld(world_args, [(agent_name, False)] + [(OPPONENT, False)] * N_OPPONENTS)
        world.time_to_stop = make_time_to_stop(s).__get__(world, BombeRLeWorld)
        agent, opponents = world.agents[0], world.agents[1:]
        trackers = {a.name: {"times": [], "timeouts": 0} for a in world.agents}
        for a in world.agents:
            instrument_think_time(a, trackers[a.name])

        for _ in range(args.n_rounds):
            world.new_round()
            round_seed = args.seed * 1000 + world.round
            random.seed(round_seed)
            np_seed(round_seed)
            for tr in trackers.values():
                tr["times"], tr["timeouts"] = [], 0
            skipped = 0
            death_step = None
            round_t0 = time.time()
            while world.running:
                if agent in world.active_agents and agent.available_think_time <= 0:
                    skipped += 1
                world.do_step("WAIT")
                if agent.dead and death_step is None:
                    death_step = world.step

            actions = list(world.replay["actions"][agent.name])
            stats = agent.statistics
            self_kill = stats.get("suicides", 0) > 0
            scores = [int(a.score) for a in world.agents]
            coins_all = world.round_statistics.get(world.round_id, {}).get("coins", 0)
            rounds.append({
                "round": world.round,
                "score": scores[0],
                "coins": stats.get("coins", 0),
                "kills": stats.get("kills", 0),
                "dead": bool(agent.dead),
                "death_step": death_step,
                "self_kill": self_kill,
                "got_killed_by_opponent": bool(agent.dead and not self_kill),
                "steps": world.step,
                "completed": coins_all == n_coins,
                "invalid_actions": stats.get("invalid", 0),
                "wait_count": actions.count("WAIT"),
                "immediate_reverses": sum(1 for p, c in zip(actions, actions[1:]) if OPPOSITE.get(p) == c),
                "crates_destroyed": stats.get("crates", 0),
                "bombs_dropped": stats.get("bombs", 0),
                "bomb_actions": actions.count("BOMB"),
                "longest_oscillation": longest_oscillation_run(actions),
                "opponent_scores": scores[1:],
                "rank": rank_of_first(scores),
                "think_time_exceeded": trackers[agent.name]["timeouts"],
                "skipped_steps": skipped,
                "opponent_think_time_exceeded": [trackers[o.name]["timeouts"] for o in opponents],
                "act_time_mean_ms": 1000 * mean(trackers[agent.name]["times"]),
                "act_time_max_ms": 1000 * max(trackers[agent.name]["times"], default=float("nan")),
                "round_wall_seconds": time.time() - round_t0,
                "actions": actions,
                "_act_times": list(trackers[agent.name]["times"]),
            })
            r = rounds[-1]
            print(f"round {len(rounds)}/{args.n_rounds}: score={r['score']} opp={r['opponent_scores']} "
                  f"steps={r['steps']} self_kill={r['self_kill']} got_killed={r['got_killed_by_opponent']} "
                  f"timeouts={r['think_time_exceeded']} elapsed={time.time() - t_start:.0f}s", flush=True)
        world.end()

    summary = summarize(rounds, n_coins)
    summary["wall_seconds"] = time.time() - t_start
    meta = {"agent": agent_name, "opponents": [OPPONENT] * N_OPPONENTS, "scenario": SCENARIO,
            "n_rounds": args.n_rounds, "seed": args.seed, "think_time_limit_s": s.TIMEOUT,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    for r in rounds:
        del r["_act_times"]

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2)
    with open(out_dir / "rounds.json", "w") as f:
        json.dump(rounds, f)
    csv_fields = [k for k in rounds[0] if k != "actions"]
    with open(out_dir / "rounds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rounds:
            writer.writerow({**r, "opponent_scores": " ".join(map(str, r["opponent_scores"])),
                             "opponent_think_time_exceeded": " ".join(map(str, r["opponent_think_time_exceeded"]))})

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key:36s} {value:.4f}" if isinstance(value, float) else f"  {key:36s} {value}")
    print(f"\nWrote {out_dir / 'summary.json'}, rounds.json, rounds.csv, game.log")


if __name__ == "__main__":
    main()
