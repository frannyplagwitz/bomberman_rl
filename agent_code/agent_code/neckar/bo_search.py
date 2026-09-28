import glob
import os
import sys
import subprocess
import time

from pathlib import Path

import optuna
import pandas as pd
from tqdm import tqdm

# Works both as `python -m agent_code.neckar.bo_search` (relative) and as
# `python bo_search.py` from inside this directory (absolute), which the usage line in the
# docstring below still shows.
try:
    from .agent_behavior import BASE_REWARDS, POTENTIAL_WEIGHTS
except ImportError:
    from agent_behavior import BASE_REWARDS, POTENTIAL_WEIGHTS

""" Bayesian-Optimization sweep over reward/potential weights using Optuna TPE sampler

1. Each trial samples a candidate value for each weight being searched
2. Candidate values are exported as NECKAR_<BEHAVIOR>_<KEY> and are read by 
      apply_env_overrides in agent_behavior.py at import time

3. Training subprocess is launched with the environment variables set and produces a per-episode CSV
4. Per-episode CSV is loaded and the scalar score is computed from the last N episodes
5. Optuna uses score to pick weights for the next trial

Can select task to search with --task

python bo_search.py --task task1_tournament --n-trials 20

Search ranges are NOT fixed absolute numbers baked into TASK_CONFIGS - each task instead
lists which *keys* to search (search_keys), and build_search_space() resolves each into an
actual (low, high) range centered on --behavior's own current value for that key (see
relative_range()). A fixed absolute range only makes sense for the one behavior it was
tuned around: aggressive_1's TRAP potential weight (0.3) sits nowhere near peaceful's
(0.005), so a range written for peaceful would search a region far from aggressive_1's
actual design point instead of refining around it. Centering per-behavior at runtime means
the same task config searches sensibly for peaceful, any aggressive_N variant, or trapper.
"""


BEHAVIOR = "peaceful"
MODEL_TYPE = "mlp"
N_TRIALS = 20
EPISODES_PER_UPDATE = 4
DEFAULT_SPREAD = 0.6  # see relative_range()
SPREAD = DEFAULT_SPREAD


def relative_range(current: float, spread: float = DEFAULT_SPREAD) -> tuple:
    """Build a search range centered on `current`, spanning +/- `spread` fraction of its
    magnitude (spread=0.6 -> roughly [0.4x, 1.6x] for a positive current, mirrored so a
    negative current's range still stays on the negative side). Falls back to a small
    fixed window around 0 if current is exactly 0, since a relative range around 0 has no
    well-defined width (multiplying 0 by anything is still 0).
    """
    if current == 0:
        return (-0.05, 0.05)

    lo = current - spread * abs(current)
    hi = current + spread * abs(current)
    return (min(lo, hi), max(lo, hi))


def build_search_space(behavior: str, keys: list, spread: float = DEFAULT_SPREAD) -> dict:
    """Resolve a list of key names into an actual {key: (low, high)} search space, each
    centered on `behavior`'s own current value (whichever of BASE_REWARDS/
    POTENTIAL_WEIGHTS actually defines that key - see relative_range()). A key not
    defined for this particular behavior is silently skipped, so one task's key list can
    be shared across behaviors that don't all define every key.
    """
    space = {}
    for key in keys:
        if key in BASE_REWARDS.get(behavior, {}):
            current = BASE_REWARDS[behavior][key]
        elif key in POTENTIAL_WEIGHTS.get(behavior, {}):
            current = POTENTIAL_WEIGHTS[behavior][key]
        else:
            continue
        space[key] = relative_range(current, spread)
    return space


# Logging files 
MAIN_PY = Path(__file__).resolve().parent / "../../main.py"
RUN_CWD = Path(__file__).resolve().parent
LOG_DIR = RUN_CWD / "logs"


# Per-task profile: contains all the parameters needed for a task 

# Create profiles for settings needed for each task 
# Per-task profile: all parameters needed for a task.
#
# Tasks list which *keys* to search (search_keys); build_search_space() resolves each into
# an actual (low, high) range centred on --behavior's own current value for that key. A
# fixed absolute range only suits the one behavior it was tuned around - aggressive_1's
# TRAP potential weight (0.3) sits nowhere near peaceful's (0.005), so a range written for
# peaceful would search far from aggressive_1's design point instead of refining around it.
TASK_CONFIGS = {

    "task_1_smoke": dict(
        scenario="coin-heaven",
        agents=[],
        n_rounds=50,
        # POTENTIAL_WEIGHTS: coin shaping + exploration bonus
        search_keys=["COIN", "EXPLORE"],
        warm_start=None),

    "task_1": dict(
        scenario="coin-heaven",
        agents=[],
        n_rounds=150,
        search_keys=["COIN", "EXPLORE"],
        warm_start=None),

    "task_2_smoke": dict(
        scenario="classic",
        agents=[],
        n_rounds=50,
        # BASE_REWARDS: bomb usefulness. POTENTIAL_WEIGHTS: crate shaping.
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE"],
        warm_start=None),

    "task_2": dict(
        scenario="classic",
        agents=[],
        n_rounds=150,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE"],
        warm_start=None),

    "task_3_smoke": dict(
        scenario="classic",
        agents=["peaceful_agent", "coin_collector_agent"],
        n_rounds=50,
        # + TRAP: POTENTIAL_WEIGHTS - shaping towards trapping opponents
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "TRAP"],
        warm_start=None),

    "task_3": dict(
        scenario="classic",
        agents=["peaceful_agent", "coin_collector_agent"],
        n_rounds=150,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "TRAP"],
        warm_start=None),

    # task_4/task_4_smoke: classic vs rule_based_agent, the actual target for
    # aggressive_N/trapper tuning. KILL and KILLED_OPPONENT - the two most central
    # "how aggressive" levers - are searched here alongside the bombing/crate/survival/
    # trap knobs. Every range is resolved per-behavior at runtime, not fixed absolutes.
    "task_4_smoke": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=50,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "SURVIVE", "TRAP",
                     "KILL", "KILLED_OPPONENT"],
        warm_start=None),

    "task_4": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=500,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "SURVIVE", "TRAP",
                     "KILL", "KILLED_OPPONENT"],
        warm_start=None),

    # The *_aggressive variants keep their own narrower key set. With per-behavior ranges
    # they no longer need their own absolute bounds - pass --behavior aggressive_N and the
    # ranges centre on that variant's own values.
    "task_4_aggressive_smoke": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=50,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "SURVIVE", "KILL", "TRAP"],
        warm_start=None),

    "task_4_aggressive": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=150,
        search_keys=["BOMB_WASTEFUL", "BOMB_USEFUL", "CRATE", "SURVIVE", "TRAP",
                     "KILL", "KILLED_OPPONENT"],
        warm_start=None),
}
    
    
          
def get_seed_from_env(task_config: dict, behavior: str) -> dict: 
    """Grab seeded values from system and start BO search with those values for first trial
       If nothing exported, will run with default values (start of range)

    Args:
        task_config (dict): configuration for task
        behavior (str): agent behavior (peaceful or aggressive)

    Returns:
        dict: seeded trial
    """
    prefix = f"NECKAR_{behavior.upper()}_"
    seed = {}
    
    # For each key in search space, look for a value in the environment
    for key in task_config["search_keys"]:
        raw = os.environ.get(f"{prefix}{key}")
        if raw is None:
            continue
        try:
            seed[key] = float(raw)
        except ValueError:
            print(f"[bo_search] WARNING: {prefix}{key}={raw!r} is set but isn't "
                  f"a valid float - skipping it as a seed value and will use starting value ")
 
    return seed

def build_train_cmd(task_config: dict) -> list[str]:
    """Build training command 

    Args:
        task_config (dict): task configuration properties

    Returns:
        The command 
    """
    # sys.executable, not "python": the trial must run under the same interpreter as this
    # sweep (a bare "python" may not exist on PATH, e.g. inside a conda env).
    cmd = [sys.executable, str(MAIN_PY.resolve()), "play", "--agents" ,"neckar", *task_config["agents"],
           "--train", "1", "--n-rounds", str(task_config["n_rounds"]), 
           "--scenario", task_config["scenario"], "--no-gui"]
    
    return cmd 
    
    
def load_episodes_for_run(run_id: str) -> "pd.DataFrame":
    """Returns the per-episode CSV written by the training subprocess that used this exact
    run_id (train.py's setup_training reads NECKAR_RUN_ID - see objective() below).

    Deliberately does NOT fall back to "the newest matching file": when several
    bo_search.py sweeps run concurrently (the whole point of parallelizing trials across
    cores), each writes its own episodes-*.csv into the same shared logs/ directory at
    close to the same time, and "newest file since trial start" could silently pick up a
    *different* sweep's trial instead of this one's - a wrong score attributed to the
    wrong config, with no error to notice it by. Matching on the unique run_id embedded
    in the filename (see train.py's episode_log_path) makes that impossible instead of
    just unlikely.

    Args:
        run_id (str): the id this trial pinned via NECKAR_RUN_ID

    Returns:
        pd.DataFrame: that run's per-episode rows
    """
    matches = glob.glob(str(LOG_DIR / f"episodes-*-{run_id}.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No episodes CSV found for run_id={run_id!r} in {LOG_DIR} - "
            f"check NECKAR_RUN_ID reached train.py's setup_training")

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous: multiple CSVs matched run_id={run_id!r}: {matches}")

    return pd.read_csv(matches[0])


def score_episodes(df: "pd.DataFrame", window: int = 15) -> float: 
    """
    Score a window of episodes at the end of a round
    """
    
    tail = df.tail(window) if len(df) >= window else df 
    
    if tail.empty:
        return -1e9 # no episodes, worst possible score 
    
    # Find averages of metrics based on values in window 
    coins_collected = tail["coins_collected"].mean()
    crates_destroyed = tail["crates_destroyed"].mean()
    bombs_dropped = tail["bombs_dropped"]
    useful_rate = (tail["bombs_useful"] / bombs_dropped.replace(0, pd.NA)).fillna(0.0).mean()
    death_rate = ((tail["killed_self"] == 1) | (tail["got_killed"] == 1)).mean()
    
    # Return score, 
    # return crates_destroyed + 5.0 * useful_rate - 15.0 * death_rate for mlp-sparse and aggressive
    return coins_collected + crates_destroyed + 5.0 * useful_rate - 15.0 * death_rate
    
def objective(trial: optuna.Trial, task_config: dict) -> float:
    env = os.environ.copy()

    # Resolved per-behavior, per-trial: BEHAVIOR/SPREAD are fixed for the whole sweep, but
    # this keeps the search space construction next to where it's used rather than
    # computed once and passed around.
    search_space = build_search_space(BEHAVIOR, task_config["search_keys"], spread=SPREAD)

    for key, (low, high) in search_space.items():
        value = trial.suggest_float(key, low, high)
        env[f"NECKAR_{BEHAVIOR.upper()}_{key}"] = str(value)
        env[f"NECKAR_{BEHAVIOR.upper()}_MOVED_FROM_SAFETY"] = str(-value) # Mirror MOVED_TO_SAFETY

    # Store behavior and model types from arguments
    env["NECKAR_BEHAVIOR"] = BEHAVIOR
    env["NECKAR_MODEL_TYPE"] = MODEL_TYPE
    env["NECKAR_EPISODES_PER_UPDATE"] = str(EPISODES_PER_UPDATE)
    # Unique per (process, trial) - not just per trial - so concurrent bo_search.py
    # sweeps (different PIDs) never collide even though Optuna numbers each sweep's
    # trials independently starting from 0. This is what load_episodes_for_run() below
    # matches on instead of "the newest CSV file", which isn't safe once several
    # sweeps are writing into the same logs/ directory at once.
    run_id = f"bo-{os.getpid()}-{trial.number}"
    env["NECKAR_RUN_ID"] = run_id

    # thread through warm start if one is added to callback.py's setup
    # lets trial start from prior-stage checkpoint
        
    if task_config.get("warm_start"):
        env["NECKAR_WARM_START"] = task_config["warm_start"]
    # Track start of trial and build/run command
    trial_start = time.time()
    train_cmd = build_train_cmd(task_config)


    try: 
        result = subprocess.run(train_cmd, env=env, capture_output=True, text=True)

    except Exception as exc:
        trial.set_user_attr("failure", f"subprocess.run raised {type(exc).__name__}: {exc}")
        print(f"\n[trial {trial.number}] FAILED TO LAUNCH training subprocess: {type(exc).__name__}: {exc}")
        print(f"[trial {trial.number}] command was: {train_cmd}\n")
        return -1e9

    if result.returncode != 0:
        trial.set_user_attr("failure", f"training subprocess exited {result.returncode}")
        trial.set_user_attr("stderr_tail", result.stderr[-2000:])

        print(f"[trial {trial.number}] subprocess exited {result.returncode}")
        print(f"[trial {trial.number}] stdout tail:\n{result.stdout[-800:]}")
        print(f"[trial {trial.number}] stderr tail:\n{result.stderr[-800:]}\n")

        return -1e9 # training crashed under config

    # Load the result from the training
    try: 
        df = load_episodes_for_run(run_id)
        score=score_episodes(df)

        trial.set_user_attr("n_episodes", len(df))
        trial.set_user_attr("death_rate", 
                            float(((df["killed_self"] == 1) | 
                                (df["got_killed"] == 1)).mean()))

    except Exception as exc:
        trial.set_user_attr("failure", f"result processing raised {type(exc).__name__}: {exc}")
        print(f"\n[trial {trial.number}] training exited 0 but result processing failed: "
              f"{type(exc).__name__}: {exc}")
        print(f"[trial {trial.number}] run_id={run_id!r} log_dir={str(LOG_DIR)!r}\n")
        return -1e9

    return score

# Main 

if __name__ == "__main__":
    import argparse 
    import functools 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="task_4")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    # "aggressive" was renamed to aggressive_1..aggressive_4/trapper in agent_behavior.py;
    # the old bare "aggressive" key no longer exists there, so passing it here would make
    # apply_env_overrides/BASE_REWARDS.get(behavior, {}) silently fall back to an empty
    # dict - zero reward shaping, not an error, so it wouldn't have been obvious it broke.
    parser.add_argument("--behavior",
                        choices=["peaceful", "aggressive_1", "aggressive_2", "aggressive_3",
                                 "aggressive_4", "aggressive_5", "aggressive_6", "aggressive_7", "trapper"],
                        default="peaceful")
    # neckar's callbacks.build_model() handles mlp/cnn; "lut" lives in the elsenz agent.
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--seed", type=int, default=42,
                        help="TPESampler seed - vary this when running several bo_search.py "
                             "sweeps in parallel so they don't retrace identical trial sequences")
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD,
                        help="Search range width as a fraction of each key's current value "
                             "for --behavior, e.g. 0.6 -> roughly [0.4x, 1.6x] (see relative_range())")
    # Kept from main: this is threaded to the agent as NECKAR_EPISODES_PER_UPDATE and drives
    # the PPO update cadence plus the LR/entropy decay schedules (train.py), so it is
    # explicit here rather than left to silently fall back to per-episode updates.
    parser.add_argument("--episodes-per-update", type=int, default=EPISODES_PER_UPDATE,
                        help="Rollout-accumulation batch size threaded through as "
                             "NECKAR_EPISODES_PER_UPDATE (default: %(default)s).")

    args = parser.parse_args()


    # Guard against task/behavior mismatch (if --task task_4_aggressive, shouldn't get --bevhavior peaceful )
    task_is_aggressive = "aggressive" in args.task
    
    # startswith, not ==: the bare "aggressive" behavior was replaced by the
    # aggressive_1..7 variants, so an equality test against "aggressive" can never hold
    # and rejected every aggressive task regardless of --behavior.
    behavior_is_aggressive = args.behavior.startswith("aggressive")

    if task_is_aggressive and not behavior_is_aggressive:
        parser.error(f"--task {args.task!r} is an aggressive-only task; "
                      f"pass --behavior aggressive_N (got {args.behavior!r})")

    if not task_is_aggressive and args.task.startswith("task_4") and behavior_is_aggressive:
        parser.error(f"--task {args.task!r} is a peaceful-only task; "
                      f"pass --behavior peaceful, or use {args.task}_aggressive")
    

    # Get behavior and model type from arguments, these will be passed into
    # the other agent files
    BEHAVIOR = args.behavior
    SPREAD = args.spread
    MODEL_TYPE = args.model 
    EPISODES_PER_UPDATE = args.episodes_per_update
    
    task_config = TASK_CONFIGS[args.task]
    
    # Print out the specs for the optimization trial, displays the values being updated
    # during the trial 
    print(f"Running Bayesian Optimization for {args.task!r} with the following properties:\n")
    
    print(f" scenario={task_config['scenario']!r},"
          f" behavior={BEHAVIOR!r},"
          f" model={MODEL_TYPE!r}"
          f" agents={task_config['agents']}, n_rounds={task_config['n_rounds']}"
          f" episodes_per_update={EPISODES_PER_UPDATE},"
          f" warm_start={task_config.get('warm_start')!r}\n")
    
    resolved_space = build_search_space(BEHAVIOR, task_config["search_keys"], spread=args.spread)
    print(f"Search Space (centered on {BEHAVIOR!r}'s current values, spread={args.spread}):")
    for key, (low, high) in resolved_space.items():
        print(f"  {key}: ({low:.4f}, {high:.4f})")
    
    
    # Setup optuna and progress bar, one tick per completed trial
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))

    # Seed trial 0 with current best known configuration
    seed = get_seed_from_env(task_config, BEHAVIOR)

    if seed:
        study.enqueue_trial(seed)
        print(f"Seeded trial 0 with known-good config: {seed}\n")
    
    
    progress_bar = tqdm(total=args.n_trials, desc=f"BO {args.task}", unit="trial")
    
    # Callback for running optuna, once a trial is done, will add a tick to progress bar
    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        score = "crashed" if trial.value is not None and trial.value <= -1e9 else f"{trial.value:.3f}"
        progress_bar.set_postfix(score=score, best=f"{study.best_value: .3f}")
        
        try: 
            best = f"{study.best_value:.3f}"
        except ValueError: 
            best = "n/a"
        
        progress_bar.set_postfix(score=score, best=best)
        progress_bar.update(1)
        
    # Run bayesian optimization via optuna 
    try: 
        study.optimize(functools.partial(
            objective, task_config=task_config), 
            n_trials=args.n_trials, 
            callbacks=[_progress_callback])    
        
    finally: 
        progress_bar.close()
        
    # Obtain dataframe and output into csv. The filename includes the seed so parallel
    # sweeps (same task/behavior/model, different --seed) each get their own results file
    # instead of silently overwriting one shared name.
    trials_df = study.trials_dataframe()
    out_csv = (f"bo_search_results_{args.task}_{args.behavior}_{args.model}"
               f"_seed{args.seed}.csv")
    trials_df.to_csv(out_csv, index=False)
    print(f"\nFull trial history written to {out_csv}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and t.value > -1e9]

    if not completed:
        print("\n No trial produced a usable score - every trial either crashed or "
              "returned the -1e9 sentinel. Per-trial failure reasons:")

        for t in study.trials:
            reason = t.user_attrs.get("failure") or t.user_attrs.get("stderr_tail", "")[-300:]
            print(f"  trial {t.number}: state={t.state.name} value={t.value} reason={reason!r}")

        print(f"\nSee {out_csv} for the full table, and the [trial N] diagnostics printed "
              f"above for stdout/stderr tails from each subprocess.")

    else:
        # Obtain best trial and its parameters
        print("\n Best Trial: ")
        print(f" score: {study.best_value:.3f}")
        print(f" params: {study.best_params}")

        # Print out the top 3 best trials
        top = trials_df.sort_values("value", ascending=False).head(3)

        print("\nTop 3 trials by score")
        print(top[[c for c in top.columns if c.startswith("params_") or c == "value"]].to_string(index=False))
