"""6-stage curriculum runner for a single (behavior, model_type, weight_source) chain.

Each stage warm-starts from the previous stage's checkpoint via NECKAR_WARM_START and saves
its own checkpoint (kept, not overwritten) so every stage's weights stay recoverable, not
just the final one. Runs one full chain per invocation - launch several in parallel
(different --chain-name) for several behaviors/weight-sources at once, see
AGGRESSIVE_MLP_TRAINING.md "Curriculum launch" section.

Every chain's stage checkpoint/episode-CSV filenames are disambiguated by NECKAR_RUN_ID (train.py
now suffixes the checkpoint .pt with self.run_id - see AGGRESSIVE_MLP_TRAINING.md "checkpoint
filename collision" note), so several chains can run concurrently without clobbering each
other's files even though they all write into the same neckar/ directory: agents.py's
per-call os.chdir(.../agent_code/<name>/) (see agents.py:305) means every subprocess writes
relative to the *real* neckar directory regardless of what cwd it was launched with -
giving a chain its own subprocess cwd does NOT isolate its file writes.
"""
import argparse
import fcntl
import glob
import os
import sys
import shutil
import subprocess
import time

from pathlib import Path

import pandas as pd

RUN_CWD = Path(__file__).resolve().parent
MAIN_PY = (RUN_CWD / "../../main.py").resolve()
CHAINS_DIR = RUN_CWD / "curriculum_runs"
README = RUN_CWD / "AGGRESSIVE_MLP_TRAINING.md"
STATS_MARKER = "<!-- CURRICULUM_STATS_TABLE -->"

# loot-crate inserted before classic - see AGGRESSIVE_MLP_TRAINING.md "Curriculum design":
# same CRATE_DENSITY as classic, differs only in COIN_COUNT (50 vs 9), so it's a coin-rich
# stepping stone for crate/bombing mechanics before classic's scarcity.
STAGES = [
    dict(name="1_coin_heaven", scenario="coin-heaven", agents=[], n_rounds=300),
    dict(name="2_lootcrate_solo", scenario="loot-crate", agents=[], n_rounds=400),
    dict(name="3_lootcrate_opposition", scenario="loot-crate",
         agents=["peaceful_agent", "coin_collector_agent"], n_rounds=500),
    dict(name="4_classic_solo", scenario="classic", agents=[], n_rounds=400),
    dict(name="5_classic_opposition", scenario="classic",
         agents=["peaceful_agent", "coin_collector_agent"], n_rounds=500),
    dict(name="6_classic_rule_based", scenario="classic", agents=["rule_based_agent"], n_rounds=2000),
]

# Opponent-difficulty curriculum, orthogonal to STAGES above (which varies scenario/coin
# density, not opponent skill). Purpose: rule_based_agent evades well enough that real kills
# are near-zero regardless of reward tuning (see "Curriculum run complete" results below) -
# this ramps opponent evasion skill from none at all (stationary_agent, added specifically
# for this - agent_code/stationary_agent, always WAITs) up to rule_based_agent, so the agent
# can find and lock in a kill strategy against an easy target before facing one that actively
# dodges. All stages use `classic` throughout - scenario isn't the variable here, opponent
# skill is.
ENEMY_FOCUS_STAGES = [
    dict(name="1_stationary", scenario="classic", agents=["stationary_agent"], n_rounds=200),
    dict(name="2_peaceful", scenario="classic", agents=["peaceful_agent"], n_rounds=300),
    dict(name="3_random", scenario="classic", agents=["random_agent"], n_rounds=300),
    dict(name="4_coin_collector", scenario="classic", agents=["coin_collector_agent"], n_rounds=300),
    dict(name="5_rule_based", scenario="classic", agents=["rule_based_agent"], n_rounds=2000),
]

# Same opponent-skill ramp as ENEMY_FOCUS_STAGES, but stage 1 uses `empty` instead of
# `classic` - settings.SCENARIOS' "empty" is CRATE_DENSITY=0, COIN_COUNT=0, i.e. exactly
# coin-heaven (CRATE_DENSITY=0) with the coins removed (COIN_COUNT=0 instead of 50) - already
# defined, no new scenario needed. Paired with aggressive_5 (agent_behavior.py), whose reward
# is zero for everything except KILLED_OPPONENT/TRAPPED_ENEMY and the KILL/TRAP potentials -
# removing coins from the very first stage means there is *nothing else in the environment*
# to interact with besides the opponent, so there's no ambiguity about what the reward is
# for even before the reward-only-cares-about-combat design is exercised at all.
ENEMY_FOCUS_KILL_ONLY_STAGES = [
    dict(name="1_stationary_no_coin", scenario="empty", agents=["stationary_agent"], n_rounds=200),
    dict(name="2_peaceful", scenario="classic", agents=["peaceful_agent"], n_rounds=300),
    dict(name="3_random", scenario="classic", agents=["random_agent"], n_rounds=300),
    dict(name="4_coin_collector", scenario="classic", agents=["coin_collector_agent"], n_rounds=300),
    dict(name="5_rule_based", scenario="classic", agents=["rule_based_agent"], n_rounds=2000),
]

CURRICULA = {"scenario": STAGES, "enemy_focus": ENEMY_FOCUS_STAGES,
             "enemy_focus_kill_only": ENEMY_FOCUS_KILL_ONLY_STAGES}

# Convergence check for each curriculum's final (hardest-opponent) stage: if the death rate
# over the last 100 episodes is still this high, the agent hasn't stabilized against
# rule_based_agent yet - run more rounds from where it left off instead of calling it done at
# a fixed round count.
CONVERGENCE_DEATH_RATE = 0.55
CONVERGENCE_EXTENSION_ROUNDS = 800
MAX_EXTENSIONS = 2


def checkpoint_name(model_type, behavior, scenario, n_rounds, n_opponents, run_id):
    return (f"actor-critic-{model_type}-{behavior}-{scenario}"
            f"-{n_rounds}-rounds-{n_opponents}-opponents-{run_id}.pt")


def build_cmd(stage, n_rounds):
    # sys.executable, not "python": a bare "python" may not exist on PATH (conda env).
    return [sys.executable, str(MAIN_PY), "play", "--agents", "neckar", *stage["agents"],
            "--train", "1", "--n-rounds", str(n_rounds),
            "--scenario", stage["scenario"], "--no-gui"]


def load_bo_best(behavior, model_type, task):
    """Best (score, params) across all seed CSVs bo_search.py wrote for this behavior/task -
    each seed ran an independent TPE search (see bo_search.py docstring), so pooling means
    taking the single best trial found across all of them, not averaging.
    """
    # bo_sweep.sh moves its result CSVs into bo_results/ once every seed process finishes
    # (to keep them from cluttering the neckar dir) - fall back to RUN_CWD itself in
    # case this is ever pointed at a sweep that wrote its CSVs straight to bo_search.py's
    # own default location instead.
    filename = f"bo_search_results_{task}_{behavior}_{model_type}_seed*.csv"
    files = sorted(glob.glob(str(RUN_CWD / "bo_results" / filename)))
    if not files:
        files = sorted(glob.glob(str(RUN_CWD / filename)))
    if not files:
        raise FileNotFoundError(f"No BO result CSVs matched {filename!r} in bo_results/ or {RUN_CWD}")

    best_val, best_row = -1e18, None
    for f in files:
        df = pd.read_csv(f)
        df = df[df["value"] > -1e8]  # drop crashed trials (score() returns -1e9)
        if df.empty:
            continue
        row = df.loc[df["value"].idxmax()]
        if row["value"] > best_val:
            best_val, best_row = row["value"], row

    if best_row is None:
        raise RuntimeError(f"All BO trials crashed for behavior={behavior!r}, task={task!r}")

    params = {c[len("params_"):]: best_row[c] for c in best_row.index if c.startswith("params_")}
    return params, float(best_val)


EMPTY_STATS = dict(death_rate=float("nan"), useful_rate=float("nan"), crates=float("nan"),
                    mean_reward=None, map_coverage=float("nan"), win_rate=float("nan"),
                    win_total=0, n_episodes=0)


def stage_stats(episodes_csv, window=100):
    """Stats for one stage's episode CSV. death_rate/useful_rate/crates/map_coverage/win_rate
    are means over the last `window` episodes (or all episodes if fewer than `window`) - a
    snapshot of current behavior, not a lifetime total. win_total is the one exception -
    summed over the *entire* stage's episodes, not just the window, since "how many rounds
    did it win in this stage" is naturally a running total, not a recent-behavior snapshot.
    """
    df = pd.read_csv(episodes_csv)
    tail = df.tail(window) if len(df) >= window else df
    if tail.empty:
        return dict(EMPTY_STATS, n_episodes=0)
    death_rate = float(((tail["killed_self"] == 1) | (tail["got_killed"] == 1)).mean())
    useful_rate = float((tail["bombs_useful"] / tail["bombs_dropped"].replace(0, pd.NA))
                         .fillna(0.0).mean())
    crates = float(tail["crates_destroyed"].mean())
    mean_reward = tail["mean_reward"].dropna()
    mean_reward = float(mean_reward.iloc[-1]) if not mean_reward.empty else None
    map_coverage = float(tail["map_coverage"].mean()) if "map_coverage" in tail else float("nan")
    win_rate = float(tail["won_round"].mean()) if "won_round" in tail else float("nan")
    win_total = int(df["won_round"].sum()) if "won_round" in df else 0
    return dict(death_rate=death_rate, useful_rate=useful_rate, crates=crates,
                mean_reward=mean_reward, map_coverage=map_coverage, win_rate=win_rate,
                win_total=win_total, n_episodes=len(df))


def append_readme_row(chain, stage_name, n_rounds, wall_s, stats, note=""):
    """Append one row to AGGRESSIVE_MLP_TRAINING.md's stats table, file-locked since several
    chains' processes call this concurrently.
    """
    def fmt(key, spec=".2f"):
        val = stats.get(key)
        return format(val, spec) if val == val and val is not None else "n/a"  # val==val excludes NaN

    death = fmt("death_rate")
    useful = fmt("useful_rate")
    crates = fmt("crates")
    reward = fmt("mean_reward", ".3f")
    coverage = fmt("map_coverage")
    win_rate = fmt("win_rate")
    win_total = stats.get("win_total", 0)
    line = (f"| {chain} | {stage_name} | {n_rounds} | {wall_s:.0f}s | {death} | {useful} "
            f"| {crates} | {coverage} | {win_rate} | {win_total} | {reward} | {note} |\n")

    lock_path = str(README) + ".lock"
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            text = README.read_text()
            if STATS_MARKER not in text:
                text += (
                    "\n## Curriculum run stats\n\n"
                    "Appended automatically by `curriculum_run.py` as each stage finishes - "
                    "death/useful/crates/map_coverage/win_rate are means over the last 100 "
                    "episodes of that stage (or all episodes if fewer than 100); win_total is "
                    "summed over the *entire* stage, not just that window. map_coverage is the "
                    "fraction of the map's walkable tiles visited by episode end (mean); "
                    "win_rate is the fraction of those episodes with the highest score at "
                    "round end (`WON_ROUND` - see train.py).\n\n"
                    f"{STATS_MARKER}\n"
                    "| chain | stage | rounds | wall_time | death_rate | bomb_useful_rate "
                    "| crates_destroyed | map_coverage | win_rate | win_total | mean_reward | note |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
                )
            text += line
            README.write_text(text)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def run_stage(env, stage, n_rounds, run_id, warm_start):
    stage_env = env.copy()
    stage_env["NECKAR_RUN_ID"] = run_id
    if warm_start is not None:
        stage_env["NECKAR_WARM_START"] = str(warm_start)

    cmd = build_cmd(stage, n_rounds)
    t0 = time.time()
    # No cwd override: agents.py chdir's into the real agent_code/<name>/ directory per
    # callback invocation regardless of subprocess cwd (see module docstring), so running
    # from RUN_CWD (matching bo_search.py's already-proven pattern) is equivalent and simpler.
    result = subprocess.run(cmd, env=stage_env, capture_output=True, text=True)
    wall = time.time() - t0
    return result, wall


def run_chain(behavior, model_type, weight_source, chain_name, stages=STAGES,
              task_for_bo="task_4", bo_model="mlp"):
    chain_dir = CHAINS_DIR / chain_name
    chain_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NECKAR_BEHAVIOR"] = behavior
    env["NECKAR_MODEL_TYPE"] = model_type
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    if weight_source == "bo":
        params, best_val = load_bo_best(behavior, bo_model, task_for_bo)
        for key, value in params.items():
            env[f"NECKAR_{behavior.upper()}_{key}"] = str(value)
        first_note = "BO weights (score {:.3f}): {}".format(
            best_val, ", ".join(f"{k}={v:.4f}" for k, v in params.items()))
    else:
        first_note = "hand-tuned defaults (no env overrides)"

    prev_ckpt = None
    for i, stage in enumerate(stages, start=1):
        run_id = f"{chain_name}-s{i}"
        result, wall = run_stage(env, stage, stage["n_rounds"], run_id, prev_ckpt)

        if result.returncode != 0:
            (chain_dir / f"stage{i}_FAILED.log").write_text(
                f"CMD FAILED rc={result.returncode}\n\nSTDOUT tail:\n{result.stdout[-4000:]}"
                f"\n\nSTDERR tail:\n{result.stderr[-4000:]}\n")
            append_readme_row(chain_name, stage["name"], stage["n_rounds"], wall,
                               dict(EMPTY_STATS),
                               note=f"CRASHED rc={result.returncode}, chain aborted")
            return

        n_opp = len(stage["agents"])
        ckpt_src = RUN_CWD / checkpoint_name(model_type, behavior, stage["scenario"],
                                              stage["n_rounds"], n_opp, run_id)
        ckpt_dst = chain_dir / f"stage{i}_{stage['name']}.pt"
        shutil.copy(ckpt_src, ckpt_dst)
        prev_ckpt = ckpt_dst.resolve()

        ep_csv = RUN_CWD / "logs" / (
            f"episodes-{model_type}-{behavior}-{stage['scenario']}-{stage['n_rounds']}"
            f"-rounds-{n_opp}-opponents-{run_id}.csv")
        stats = stage_stats(ep_csv)
        append_readme_row(chain_name, stage["name"], stage["n_rounds"], wall, stats,
                           note=(first_note if i == 1 else ""))

        if i == len(stages):
            extension = 0
            while stats["death_rate"] == stats["death_rate"] and \
                    stats["death_rate"] > CONVERGENCE_DEATH_RATE and extension < MAX_EXTENSIONS:
                extension += 1
                ext_name = f"{stage['name']}_ext{extension}"
                ext_run_id = f"{chain_name}-s{i}-ext{extension}"
                result, wall = run_stage(env, stage, CONVERGENCE_EXTENSION_ROUNDS,
                                          ext_run_id, prev_ckpt)
                if result.returncode != 0:
                    append_readme_row(chain_name, ext_name, CONVERGENCE_EXTENSION_ROUNDS, wall,
                                       dict(EMPTY_STATS),
                                       note=f"CRASHED rc={result.returncode}, extension aborted")
                    break

                ckpt_src = RUN_CWD / checkpoint_name(model_type, behavior, stage["scenario"],
                                                      CONVERGENCE_EXTENSION_ROUNDS, n_opp, ext_run_id)
                ckpt_dst = chain_dir / f"stage{i}_{ext_name}.pt"
                shutil.copy(ckpt_src, ckpt_dst)
                prev_ckpt = ckpt_dst.resolve()

                ep_csv = RUN_CWD / "logs" / (
                    f"episodes-{model_type}-{behavior}-{stage['scenario']}"
                    f"-{CONVERGENCE_EXTENSION_ROUNDS}-rounds-{n_opp}-opponents-{ext_run_id}.csv")
                stats = stage_stats(ep_csv)
                append_readme_row(chain_name, ext_name, CONVERGENCE_EXTENSION_ROUNDS, wall, stats,
                                   note=f"convergence extension {extension}/{MAX_EXTENSIONS} "
                                        f"(death_rate threshold {CONVERGENCE_DEATH_RATE})")

    append_readme_row(chain_name, "DONE", 0, 0.0, dict(EMPTY_STATS),
                       note="chain complete, final checkpoint = last stage listed above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--model", choices=["mlp", "mlp_sparse"], default="mlp")
    parser.add_argument("--weight-source", choices=["hand", "bo"], default="hand")
    parser.add_argument("--curriculum", choices=sorted(CURRICULA), default="scenario",
                         help="'scenario' = original 6-stage coin-heaven->classic+rule_based "
                              "progression; 'enemy_focus' = fixed classic scenario, ramping "
                              "opponent skill from stationary_agent up to rule_based_agent")
    parser.add_argument("--chain-name", required=True,
                         help="Unique name for this chain's working dir/log rows, "
                              "e.g. aggressive_1-bo or aggressive_1-hand")
    parser.add_argument("--task-for-bo", default="task_4",
                         help="Which bo_search.py TASK_CONFIGS entry's result CSVs to pool "
                              "when --weight-source=bo")
    parser.add_argument("--bo-model", default="mlp",
                         help="model_type used when the BO sweep was run (for locating its "
                              "result CSVs) - independent of --model, which is what this "
                              "curriculum chain itself trains")
    args = parser.parse_args()

    run_chain(args.behavior, args.model, args.weight_source, args.chain_name,
               stages=CURRICULA[args.curriculum],
               task_for_bo=args.task_for_bo, bo_model=args.bo_model)
