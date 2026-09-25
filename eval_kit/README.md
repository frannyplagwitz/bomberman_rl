# eval_kit

Evaluates one agent against three `rule_based_agent` opponents on the `classic` scenario and writes per-round results plus a summary. This is the shared evaluation setup for the report (classic, 100 rounds, seed 1000), so results from different agents can be compared directly.

## Setup

Copy the `eval_kit/` folder into the root of your bomberman_rl framework copy, next to `main.py`:

```
bomberman_rl/
  main.py
  environment.py
  agents.py
  settings.py
  agent_code/
    rule_based_agent/
    your_agent/
  eval_kit/
    evaluate.py
    README.md
```

The agent to evaluate has to be a normal agent folder inside `agent_code/` (at least `callbacks.py` with `setup` and `act`, plus whatever files it loads).

## Requirements

- Python 3 with `numpy` (already needed by the framework).
- The unmodified framework files: `environment.py`, `agents.py`, `settings.py`, `events.py`, `items.py`, `fallbacks.py`, and `agent_code/rule_based_agent/`.
- Whatever your own agent imports (e.g. `torch`).

No other packages are needed. The script does not modify any framework or agent files.

## Usage

Run from the framework root:

```
python eval_kit/evaluate.py --agent your_agent
python eval_kit/evaluate.py --agent agent_code/your_agent --n-rounds 100 --seed 1000
python eval_kit/evaluate.py --agent your_agent --n-rounds 20 --seed 2000 --out-dir results/test
```

| Argument | Default | Meaning |
|---|---|---|
| `--agent` | required | Folder name under `agent_code/`, or a path to it |
| `--n-rounds` | 100 | Number of rounds |
| `--seed` | 1000 | Evaluation seed |
| `--out-dir` | `eval_kit/results/<agent>_seed<seed>_n<rounds>/` | Where results go |
| `--framework-root` | found automatically | Only needed if `eval_kit/` is not inside the framework |

Run time depends mostly on your agent: with about 20 ms per `act()` call, 100 rounds took about 10 minutes. Progress is printed after every round.

## What is fixed

- Scenario `classic`, your agent plus 3x `rule_based_agent`, your agent always in the first slot.
- The think-time limit is the framework default (`TIMEOUT` in `settings.py`). Slow steps are turned into `WAIT` exactly as in a normal game, and the number of times this happened is reported.
- Map layout and start positions come from the world RNG seeded with `--seed`.
- Opponent randomness is pinned without touching the opponent code: `np.random.seed()` calls are redirected to the seed, Python's `random` and numpy's global RNG are reseeded at the start of every round from `(seed, round)`, and the script re-runs itself with `PYTHONHASHSEED=0`.
- One small difference from `main.py`: a round is not ended early when only one agent is left on a cleared board while one of your opponents is still alive. This has no effect on scores.

With the same agent, seed and number of rounds, two runs give the same games, as long as no step hits the time limit. Once a step times out, later rounds can differ between runs.

The framework measures think time with the system wall clock (`time.time()`). If the system clock gets adjusted during a step (this happens regularly under WSL2), that step can be counted as a timeout although the agent was not slow. The time-limit counters and `act_time_max_ms` include such cases.

**Deterministic policy is the responsibility of each agent.** If your agent samples actions, uses its own unseeded RNG, or depends on timing, results will vary between runs. The script does not seed or configure your agent's code (for example, it does not set torch seeds or thread counts).

## Output

All files go to the output directory:

- `summary.json`: run settings (`meta`) and aggregate metrics (`summary`).
- `rounds.csv`: one row per round.
- `rounds.json`: same as the CSV plus the full action sequence of your agent per round.
- `game.log`: the framework's own game log for this run.

The agent's own log is written by the framework to `agent_code/<agent>/logs/` as usual.

### Summary metrics

Rates are fractions of rounds. "Score" is the official game score (1 per coin, 5 per kill).

| Key | Meaning |
|---|---|
| `score_mean`, `score_std`, `score_se`, `score_total` | Your agent's score per round: mean, sample std, standard error, sum |
| `coins_collected_mean`, `opponent_kills_mean` | Coins and kills per round |
| `self_kill_rate` | Rounds in which your agent died by its own bomb |
| `got_killed_by_opponent_rate` | Rounds in which your agent died by another agent's bomb |
| `survival_rate` | Rounds in which your agent was alive at the end |
| `completion_rate` | Rounds in which all coins were collected (by anyone) |
| `steps_to_completion_mean/median` | Round length over completed rounds |
| `episode_length_mean` | Round length over all rounds |
| `steps_per_coin_mean` | Round length divided by own coins, over rounds with at least one coin |
| `wait_fraction_mean` | Share of `WAIT` actions |
| `invalid_action_count_total` | Invalid actions over all rounds |
| `immediate_reverse_count_mean` | Moves directly undone by the next move (e.g. LEFT then RIGHT), per round |
| `crates_destroyed_mean`, `bombs_dropped_mean` | Per round, from the framework statistics |
| `bombs_placed_mean` | `BOMB` actions per round, from the action sequence |
| `oscillation_fraction` | Rounds containing a back-and-forth run (e.g. LEFT RIGHT LEFT ...) of at least 10 actions |
| `opponent_score_mean`, `best_opponent_score_mean` | Mean score of a single opponent; mean score of the best opponent in each round |
| `strict_first_rate`, `shared_first_rate` | Rounds where your agent scored more than every opponent; more than or equal to |
| `mean_rank` | Average placement among the 4 agents (1 = best; ties share the average rank) |
| `margin_vs_best_opponent_mean` | Your score minus the best opponent's score, averaged |
| `think_time_exceeded_total` | Steps where your agent exceeded the time limit (action replaced by `WAIT`) |
| `skipped_steps_total` | Steps your agent was not asked to act because of an earlier overrun |
| `opponent_think_time_exceeded_total` | The same as `think_time_exceeded_total`, summed over the three opponents |
| `act_time_mean_ms`, `act_time_max_ms` | Time spent in your `act()` per step |
| `wall_seconds` | Total run time |

### Per-round columns

`round`, `score`, `coins`, `kills`, `dead`, `death_step`, `self_kill`, `got_killed_by_opponent`, `steps`, `completed`, `invalid_actions`, `wait_count`, `immediate_reverses`, `crates_destroyed`, `bombs_dropped`, `bomb_actions`, `longest_oscillation`, `opponent_scores` (space-separated in the CSV), `rank`, `think_time_exceeded`, `skipped_steps`, `opponent_think_time_exceeded`, `act_time_mean_ms`, `act_time_max_ms`, `round_wall_seconds`. `rounds.json` also has `actions`.
