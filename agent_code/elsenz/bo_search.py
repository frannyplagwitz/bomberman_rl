import glob
import os 
import subprocess
import time

from pathlib import Path

import optuna
import pandas as pd
from tqdm import tqdm

""" Bayesian-Optimization sweep over reward/potential weights using Optuna TPE sampler 

1. Each trial samples a candidate value for each weight being searched
2. Candidate values are exported as ELSENZ_<BEHAVIOR>_<KEY> and are read by 
      apply_env_overrides in agent_behavior.py at import time

3. Training subprocess is launched with the environment variables set and produces a per-episode CSV
4. Per-episode CSV is loaded and the scalar score is computed from the last N episodes
5. Optuna uses score to pick weights for the next trial 

Can select task to search with --task 

python bo_search.py --task task1_tournament --n-trials 20 
"""


BEHAVIOR = "peaceful"
N_TRIALS = 20
EPISODES_PER_UPDATE = 4


# Logging files 
MAIN_PY = Path(__file__).resolve().parent / "../../main.py"
RUN_CWD = Path(__file__).resolve().parent
LOG_DIR = RUN_CWD / "logs"
CSV_GLOB = str(LOG_DIR / "episodes*.csv")


# Per-task profile: contains all the parameters needed for a task 

TASK_CONFIGS = {
    
    "task_1_smoke": dict(
        scenario="coin-heaven",
        agents=[], 
        n_rounds=50, 
        search_space={
            "COIN": (0.02, 0.20),    # POTENTIAL_WEIGHTS - shaping towards revealed coins
            "EXPLORE": (0.001, 0.02) # POTENTIAL_WEIGHTS - exploration bonus 
        }),
    
    "task_1": dict(
        scenario="coin-heaven", 
        agents=[],
        n_rounds=150,
        search_space={
            "COIN": (0.02, 0.20),    
            "EXPLORE": (0.001, 0.02) 
        }),
    
    
    "task_2_smoke": dict(
        scenario="classic",
        agents=[],
        n_rounds=50,
        search_space={
            "BOMB_WASTEFUL": (-0.5, -0.3),  # BASE_REWARDS - shaping away from bomb not destroying crate/endangering opponent
            "BOMB_USEFUL": (0.10, 0.45),    # BASE_REWARDS - shaping towards bomb destroying crate/endangers opponent
            "MOVED_TO_SAFETY":(0.05, 0.35), # BASE_REWARDS - shaping towards moving to safety if in danger
            "CRATE": (0.03, 0.10)           # POTENTIAL_WEIGHTS - shaping towards destroying crates
        }),
        
    "task_2": dict(
        scenario="classic",
        agents=[],
        n_rounds=150,
        search_space={
            "BOMB_WASTEFUL": (-0.5, -0.3),
            "BOMB_USEFUL": (0.10, 0.45),
            "MOVED_TO_SAFETY":(0.05, 0.35),
            "CRATE": (0.03, 0.10)
        }),
    
    "task_3_smoke": dict(
        scenario="classic",
        agents=["peaceful_agent", "coin_collector_agent"],
        n_rounds=50, 
        search_space={
            "BOMB_WASTEFUL": (-0.5, -0.3), 
            "BOMB_USEFUL": (0.10, 0.45),
            "MOVED_TO_SAFETY":(0.05, 0.35),
            "CRATE": (0.03, 0.10), 
            "TRAP": (0.002, 0.02) # POTENTIAL_WEIGHTS - shaping towards trapping opponents
        }),
    
            
    "task_3": dict(
        scenario="classic",
        agents=["peaceful_agent", "coin_collector_agent"],
        n_rounds=150, 
        search_space={
            "BOMB_WASTEFUL": (-0.8, -0.3), 
            "BOMB_USEFUL": (0.10, 0.45),
            "MOVED_TO_SAFETY":(0.05, 0.35),
            "CRATE": (0.03, 0.10), 
            "TRAP": (0.002, 0.02)
        }),
    
    "task_4_smoke": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=50, 
        search_space={
            "BOMB_WASTEFUL": (-0.8, -0.3), 
            "BOMB_USEFUL": (0.10, 0.45),
            "MOVED_TO_SAFETY":(0.05, 0.35),
            "CRATE": (0.03, 0.08), 
            "SURVIVE": (0.8, 1.0), # POTENTIAL_WEIGHTS - shaping towards surviving 
            "TRAP": (0.002, 0.02)  # POTENTIAL_WEIGHTS - shaping towards trapping an opponent
        }),
    
    "task_4": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=500, 
        search_space={
            "BOMB_WASTEFUL": (-0.8, -0.3), 
            "BOMB_USEFUL": (0.10, 0.45),
            "MOVED_TO_SAFETY":(0.05, 0.35),
            "CRATE": (0.03, 0.08), 
            "SURVIVE": (0.8, 1.0),
            "TRAP": (0.002, 0.02)
        }),
    
    "task_4_aggressive_smoke": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=50,
        search_space={
            "BOMB_WASTEFUL": (-0.8, -0.3),
            "BOMB_USEFUL": (0.10, 0.45), 
            "CRATE": (0.05, 0.25),
            "SURVIVE":(0.0, 0.3), # Starting at 0, because aggressive doesn't care about survival;
            "KILL": (0.3, 1.5), # POTENTIAL_WEIGHT: Shaping towards killing opponents
            "TRAP": (0.1, 0.5), # High bounds for kill and trap   
        }),
    
    "task_4_aggressive": dict(
        scenario="classic",
        agents=["rule_based_agent"],
        n_rounds=150,
        search_space={
            "BOMB_WASTEFUL": (-0.8, -0.3),
            "BOMB_USEFUL": (0.10, 0.45), 
            "CRATE": (0.05, 0.25), 
            "SURVIVE":(0.0, 0.3), 
            "KILL": (0.3, 1.5),
            "TRAP": (0.1, 0.5), 
        })
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
    prefix = f"ELSENZ_{behavior.upper()}_"
    seed = {}
    
    # For each key in search space, look for a value in the environment
    for key in task_config["search_space"]:
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
    cmd = ["python", str(MAIN_PY.resolve()), "play", "--agents" ,"elsenz", *task_config["agents"],
           "--train", "1", "--n-rounds", str(task_config["n_rounds"]), 
           "--scenario", task_config["scenario"], "--no-gui"]
    
    return cmd 
    
    
def load_latest_episodes(csv_glob: str, since_ts: float) -> "pd.DataFrame":
    """Returns per-episode CSV written by training subprocess that just finished
    Is identified as the newest file matching csv_glob with mtime since_ts

    Args:
        csv_glob (str): csv global 
        since_ts (float): mtime since last file read

    Returns:
        pd.DataFrame: DataFrame containing latest episodes
    """
    candidates = [f for f in glob.glob(csv_glob) if os.path.getmtime(f) >= since_ts]
    
    if not candidates: 
        raise FileNotFoundError(
            f"No CSV file matching {csv_glob!r} was written after trial started"
            f"Check CSV_GLOB matches where episode_logger.py actually writes to")
    
    newest = max(candidates, key=os.path.getmtime)
    return pd.read_csv(newest)


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
    return coins_collected + crates_destroyed + 5.0 * useful_rate - 15.0 * death_rate
    
def objective(trial: optuna.Trial, task_config: dict) -> float: 
    env = os.environ.copy()
    
    # Obtain values from search space that are also in the environment  
    for key, (low, high) in task_config["search_space"].items():
        value = trial.suggest_float(key, low, high)
        env[f"ELSENZ_{BEHAVIOR.upper()}_{key}"] = str(value)
        env[f"ELSENZ_{BEHAVIOR.upper()}_MOVED_FROM_SAFETY"] = str(-value) # Mirror MOVED_TO_SAFETY
    
    # Store behavior and updatesfrom arguments
    env["ELSENZ_BEHAVIOR"] = BEHAVIOR
    env["ELSENZ_EPISODES_PER_UPDATE"] = str(EPISODES_PER_UPDATE)

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
        df = load_latest_episodes(CSV_GLOB, since_ts=trial_start)
        score=score_episodes(df)

        trial.set_user_attr("n_episodes", len(df))
        trial.set_user_attr("death_rate", 
                            float(((df["killed_self"] == 1) | 
                                (df["got_killed"] == 1)).mean()))

    except Exception as exc:
        trial.set_user_attr("failure", f"result processing raised {type(exc).__name__}: {exc}")
        print(f"\n[trial {trial.number}] training exited 0 but result processing failed: "
              f"{type(exc).__name__}: {exc}")
        print(f"[trial {trial.number}] CSV_GLOB={CSV_GLOB!r}\n")
        return -1e9

    return score

# Main 

if __name__ == "__main__":
    import argparse 
    import functools 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="task_4")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--behavior", choices=["peaceful", "aggressive"], default = "peaceful")
    parser.add_argument("--episodes-per-update", type=int, default=EPISODES_PER_UPDATE,
                         help="Rollout-accumulation batch size threaded through as "
                              "ELSENZ_EPISODES_PER_UPDATE (default: %(default)s). Explicit "
                              "so a sweep can't silently fall back to per-episode updates.")
    
    args = parser.parse_args()


    # Guard against task/behavior mismatch (if --task task_4_aggressive, shouldn't get --bevhavior peaceful )
    task_is_aggressive = "aggressive" in args.task
    
    if task_is_aggressive and args.behavior != "aggressive":
        parser.error(f"--task {args.task!r} is an aggressive-only task; "
                      f"pass --behavior aggressive (got {args.behavior!r})")
        
    if not task_is_aggressive and args.task.startswith("task_4") and args.behavior == "aggressive":
        parser.error(f"--task {args.task!r} is a peaceful-only task; "
                      f"pass --behavior peaceful, or use {args.task}_aggressive")
    

    # Get behavior and model type from arguments, these will be passed into 
    # the other agent files 
    BEHAVIOR = args.behavior
    EPISODES_PER_UPDATE = args.episodes_per_update
    
    task_config = TASK_CONFIGS[args.task]
    
    # Print out the specs for the optimization trial, displays the values being updated
    # during the trial 
    print(f"Running Bayesian Optimization for {args.task!r} with the following properties:\n")
    
    print(f" scenario={task_config['scenario']!r},"
          f" behavior={BEHAVIOR!r},"
          f" model=lut"
          f" agents={task_config['agents']}, n_rounds={task_config['n_rounds']}"
          f" episodes_per_update={EPISODES_PER_UPDATE}")
    
    print(f"Search Space: {task_config['search_space']}")
    
    
    # Setup optuna and progress bar, one tick per completed trial
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=42))

    # Seed trial 0 with current best known configuration
    seed = get_seed_from_env(task_config, BEHAVIOR)

    if seed:
        study.enqueue_trial(seed)
        print(f"Seeded trial 0 with known-good config: {seed}\n")
    
    
    progress_bar = tqdm(total=args.n_trials, desc=f"BO {args.task}", unit="trial")
    
    
    # Callback for running optuna, once a trial is done, will add a tick to progress bar
    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        score = "crashed" if trial.value is not None and trial.value <= -1e9 else f"{trial.value:.3f}"
        
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
        
    # Obtain dataframe  and output into csv
    trials_df = study.trials_dataframe()
    out_csv = f"bo_search_results_{args.task}_{args.behavior}_lut.csv"
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
        # Obtain best trial and sent parameters
        print("\n Best Trial: ")
        print(f" score: {study.best_value:.3f}")
        print(f" params: {study.best_params}")
        
        # Print out the top 3 best trials
        top = trials_df.sort_values("value", ascending=False).head(3)
    
        print("\nTop 3 trials by score")
        print(top[[c for c in top.columns if c.startswith("params_") or c == "value"]].to_string(index=False))