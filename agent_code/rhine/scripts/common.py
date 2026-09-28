"""Shared helpers for training/evaluation driver scripts.

Builds BombeRLeWorld directly instead of using main.py's CLI, which cannot
configure opponents freely or expose per-round metrics.
"""
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np

from environment import BombeRLeWorld, WorldArgs
import settings as s

from .. import config as _cfg
from ..action_mask import mask_from_semantic as _mask_from_semantic
from ..state_processing import extract_semantic_state as _extract_semantic_state

AGENT_CODE_NAME = "rhine"
AGENT_DIR = PROJECT_ROOT / "agent_code" / AGENT_CODE_NAME
LOGS_DIR = AGENT_DIR / "logs"
MODELS_DIR = AGENT_DIR / "models"

OPPOSITE_DIRECTION = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


@dataclass
class EpisodeMetrics:
    round_index: int
    steps: int
    coins_collected: int
    completed: bool
    invalid_action_count: int
    wait_count: int
    immediate_reverse_count: int
    # Invalid actions split by cause (state_processing.classify_invalid_action()).
    # own_cause is a mask failure; contested_tile is the known same-tick race.
    # -1 means not available.
    invalid_action_own_cause_count: int = -1
    invalid_action_contested_tile_count: int = -1
    crates_destroyed: int = 0
    bombs_dropped: int = 0
    self_kill: bool = False
    opponent_kills: int = 0
    # Killed by an explosion without self-killing, regardless of bomb owner.
    got_killed_by_opponent: bool = False
    # Per-round action sequence for post-hoc diagnostics.
    actions: Optional[List[str]] = None

    @property
    def wait_fraction(self) -> float:
        return self.wait_count / self.steps if self.steps > 0 else 0.0

    @property
    def score(self) -> float:
        """Official scoring from settings.py, independent of training rewards."""
        return self.coins_collected * s.REWARD_COIN + self.opponent_kills * s.REWARD_KILL


def _disable_think_time_limit(world: BombeRLeWorld):
    """Removes the per-step think-time budget for diagnostic evaluation, so
    actions are never forced to WAIT by timing. Must be re-applied before
    every world.do_step(), since environment.py resets the budget each step.
    """
    world.agents[0].available_think_time = float("inf")


def _time_to_stop_no_early_clear(world_self):
    """Per-instance replacement for BombeRLeWorld.time_to_stop() in worlds
    with opponents: skips the "one agent left and board cleared" early stop
    while any opponent is alive. Other stopping conditions are unchanged.
    Only used by worlds built here, never by main.py.
    """
    if len(world_self.active_agents) == 0:
        return True
    opponents_alive = any(a.code_name != AGENT_CODE_NAME for a in world_self.active_agents)
    if not opponents_alive:
        if (len(world_self.active_agents) == 1
                and (world_self.arena == 1).sum() == 0
                and all(not c.collectable for c in world_self.coins)
                and len(world_self.bombs) + len(world_self.explosions) == 0):
            return True
    if any(a.train for a in world_self.agents) and not world_self.args.continue_without_training:
        if not any([a.train for a in world_self.active_agents]):
            return True
    if world_self.step >= s.MAX_STEPS:
        return True
    return False


def build_world(
    scenario: str, seed: int, train: bool, match_name: str = "task1_run", log_dir: str = "logs",
    opponents: Optional[List[str]] = None,
) -> BombeRLeWorld:
    """Builds a world with the rhine agent plus optional opponents.

    Args:
        opponents: non-trainable agent directory names (train=False); names
            may repeat. None/empty gives a solo world. When non-empty,
            time_to_stop() is replaced by _time_to_stop_no_early_clear().
    """
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=log_dir, save_stats=False,
        match_name=match_name, seed=seed, silence_errors=False, scenario=scenario,
    )
    agents = [(AGENT_CODE_NAME, train)] + [(name, False) for name in (opponents or [])]
    world = BombeRLeWorld(args, agents)
    if opponents:
        world.time_to_stop = _time_to_stop_no_early_clear.__get__(world, BombeRLeWorld)
    return world


def expected_coin_count(scenario: str) -> int:
    return s.SCENARIOS[scenario]["COIN_COUNT"]


def default_success_fn(world: BombeRLeWorld) -> bool:
    """Episode success: every coin was collected by any agent. A post-hoc
    label only; it does not affect when the round stops.
    """
    total_coins_collected = world.round_statistics.get(world.round_id, {}).get("coins", 0)
    return total_coins_collected == expected_coin_count(world.args.scenario)


def _invalid_action_breakdown(agent) -> Tuple[int, int]:
    """(own_cause_count, contested_tile_count) from the agent's per-round
    counters; (-1, -1) if the agent never ran our act()."""
    fake_self = agent.backend.runner.fake_self
    return (
        getattr(fake_self, "invalid_action_own_cause_count", -1),
        getattr(fake_self, "invalid_action_contested_tile_count", -1),
    )


def run_one_round(world: BombeRLeWorld, success_fn: Optional[Callable[[BombeRLeWorld], bool]] = None) -> EpisodeMetrics:
    agent = world.agents[0]

    world.new_round()
    round_index = world.round
    while world.running:
        world.do_step("WAIT")

    actions = list(world.replay["actions"][agent.name])
    invalid_action_count = agent.statistics.get("invalid", 0)
    invalid_own_cause, invalid_contested_tile = _invalid_action_breakdown(agent)
    coins_collected = agent.statistics.get("coins", 0)
    crates_destroyed = agent.statistics.get("crates", 0)
    bombs_dropped = agent.statistics.get("bombs", 0)
    self_kill = agent.statistics.get("suicides", 0) > 0
    opponent_kills = agent.statistics.get("kills", 0)
    got_killed_by_opponent = agent.dead and not self_kill
    wait_count = actions.count("WAIT")

    reverse_count = sum(
        1 for prev, cur in zip(actions, actions[1:]) if OPPOSITE_DIRECTION.get(prev) == cur
    )

    completed = (success_fn or default_success_fn)(world)

    return EpisodeMetrics(
        round_index=round_index,
        steps=world.step,
        coins_collected=coins_collected,
        completed=completed,
        invalid_action_count=invalid_action_count,
        invalid_action_own_cause_count=invalid_own_cause,
        invalid_action_contested_tile_count=invalid_contested_tile,
        wait_count=wait_count,
        immediate_reverse_count=reverse_count,
        crates_destroyed=crates_destroyed,
        bombs_dropped=bombs_dropped,
        self_kill=self_kill,
        opponent_kills=opponent_kills,
        got_killed_by_opponent=got_killed_by_opponent,
        actions=actions,
    )


def _run_one_round_traced(
    world: BombeRLeWorld, success_fn: Optional[Callable[[BombeRLeWorld], bool]] = None,
    disable_think_time_limit: bool = False,
) -> Tuple[EpisodeMetrics, List[Dict], deque]:
    """Like run_one_round(), but also records a per-step trace (position,
    mask, danger) for self-kill diagnosis; not used on the training path.

    Also returns a rolling in-memory buffer of recent do_step() wall-clock
    durations together with available_think_time before each step. That
    budget stays negative after an overrun until repaid, so it shows whether
    a step was forced to WAIT by an earlier overrun.
    """
    agent = world.agents[0]
    world.new_round()
    world.user_input = "WAIT"  # Required by get_state_for_agent() before any do_step().
    round_index = world.round

    trace: List[Dict] = []
    recent_step_timings: deque = deque(maxlen=5)
    while world.running:
        state = world.get_state_for_agent(agent)
        if state is not None:
            semantic = _extract_semantic_state(state)
            mask = _mask_from_semantic(semantic)
            trace.append({
                "step": state["step"],
                "self_pos": semantic.self_pos,
                "bomb_available": semantic.bomb_available,
                "current_tile_in_danger": semantic.current_tile_in_danger,
                "nearest_threat_timer": semantic.nearest_threat_timer,
                "legal_actions": [_cfg.ACTIONS[i] for i, m in enumerate(mask) if m],
            })
        step_number = state["step"] if state is not None else None
        if disable_think_time_limit:
            _disable_think_time_limit(world)
        think_time_before_step = agent.available_think_time
        _t0 = time.perf_counter()
        world.do_step("WAIT")
        duration_ms = (time.perf_counter() - _t0) * 1000
        if step_number is not None:
            recent_step_timings.append((step_number, duration_ms, think_time_before_step))

    actions = list(world.replay["actions"][agent.name])
    for step_record, action in zip(trace, actions):
        step_record["chosen_action"] = action

    invalid_action_count = agent.statistics.get("invalid", 0)
    invalid_own_cause, invalid_contested_tile = _invalid_action_breakdown(agent)
    coins_collected = agent.statistics.get("coins", 0)
    crates_destroyed = agent.statistics.get("crates", 0)
    bombs_dropped = agent.statistics.get("bombs", 0)
    self_kill = agent.statistics.get("suicides", 0) > 0
    opponent_kills = agent.statistics.get("kills", 0)
    got_killed_by_opponent = agent.dead and not self_kill
    wait_count = actions.count("WAIT")
    reverse_count = sum(
        1 for prev, cur in zip(actions, actions[1:]) if OPPOSITE_DIRECTION.get(prev) == cur
    )
    completed = (success_fn or default_success_fn)(world)

    metrics = EpisodeMetrics(
        round_index=round_index,
        steps=world.step,
        coins_collected=coins_collected,
        completed=completed,
        invalid_action_count=invalid_action_count,
        invalid_action_own_cause_count=invalid_own_cause,
        invalid_action_contested_tile_count=invalid_contested_tile,
        wait_count=wait_count,
        immediate_reverse_count=reverse_count,
        crates_destroyed=crates_destroyed,
        bombs_dropped=bombs_dropped,
        self_kill=self_kill,
        opponent_kills=opponent_kills,
        got_killed_by_opponent=got_killed_by_opponent,
        actions=actions,
    )
    return metrics, trace, recent_step_timings


def write_self_kill_trace(
    trace: List[Dict], episode: EpisodeMetrics, context: Dict,
    recent_step_timings: Optional[deque] = None,
) -> Path:
    """Appends one self-kill's per-step trace to a new timestamped file under
    logs/. If recent_step_timings is given, also records each recent step's
    do_step() duration and think-time budget next to the chosen action.
    """
    from datetime import datetime

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = LOGS_DIR / f"selfkill_trace_{timestamp}.log"
    with open(path, "w") as f:
        f.write(f"Self-kill trace -- context: {context}\n")
        f.write(f"round_index={episode.round_index} steps={episode.steps} "
                f"coins_collected={episode.coins_collected} crates_destroyed={episode.crates_destroyed}\n\n")
        for record in trace:
            f.write(
                f"step={record['step']:>3} pos={record['self_pos']} "
                f"bomb_available={record['bomb_available']} "
                f"current_tile_in_danger={record['current_tile_in_danger']} "
                f"nearest_threat_timer={record['nearest_threat_timer']} "
                f"legal_actions={record['legal_actions']} "
                f"chosen_action={record.get('chosen_action')}\n"
            )
        if recent_step_timings:
            f.write("\nLast steps' do_step() wall-clock duration and available_think_time "
                    "(real historical values, measured live -- not a replay):\n")
            trace_by_step = {r["step"]: r for r in trace}
            for step_number, duration_ms, think_time_before in recent_step_timings:
                r = trace_by_step.get(step_number, {})
                deficit_flag = (
                    " <-- FORCED WAIT: think-time budget already in deficit before this step"
                    if think_time_before is not None and think_time_before <= 0 else ""
                )
                slow_flag = " <-- this step itself exceeded 500ms" if duration_ms > 500 else ""
                f.write(
                    f"  step={step_number:>3} duration={duration_ms:.1f}ms "
                    f"available_think_time_before={think_time_before:.3f}s "
                    f"legal_actions={r.get('legal_actions')} chosen_action={r.get('chosen_action')}"
                    f"{deficit_flag}{slow_flag}\n"
                )
    return path


def run_evaluation_with_self_kill_tracing(
    n_rounds: int,
    scenario: str,
    seed: int,
    init_checkpoint: Optional[str],
    success_fn: Optional[Callable[[BombeRLeWorld], bool]] = None,
    disable_think_time_limit: bool = False,
    opponents: Optional[List[str]] = None,
) -> Tuple[List[EpisodeMetrics], BombeRLeWorld, List[Path]]:
    """Deterministic evaluation (train=False) that saves a per-step trace for
    every round ending in a self-kill.

    Returns:
        (episodes, world, trace_file_paths); the paths list is empty unless
        some round self-killed.
    """
    _set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", init_checkpoint)
    _set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", None)
    _set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", None)
    _set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", None)

    world = build_world(scenario=scenario, seed=seed, train=False, opponents=opponents)
    episodes: List[EpisodeMetrics] = []
    trace_paths: List[Path] = []
    for _ in range(n_rounds):
        ep, trace, recent_step_timings = _run_one_round_traced(
            world, success_fn=success_fn, disable_think_time_limit=disable_think_time_limit,
        )
        episodes.append(ep)
        if ep.self_kill:
            context = {"scenario": scenario, "seed": seed, "init_checkpoint": init_checkpoint}
            trace_paths.append(write_self_kill_trace(trace, ep, context, recent_step_timings))
    world.end()
    return episodes, world, trace_paths


def _set_or_clear_env(name: str, value: Optional[str]):
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def run_episodes(
    n_rounds: int,
    scenario: str,
    seed: int,
    train: bool,
    init_checkpoint: Optional[str] = None,
    save_checkpoint: Optional[str] = None,
    rollout_steps_override: Optional[int] = None,
    reward_override: Optional[Dict] = None,
    ppo_override: Optional[Dict] = None,
    steps_already_trained: Optional[int] = None,
    success_fn: Optional[Callable[[BombeRLeWorld], bool]] = None,
    opponents: Optional[List[str]] = None,
) -> Tuple[List[EpisodeMetrics], BombeRLeWorld]:
    """Runs n_rounds rounds on a freshly built world.

    Args:
        success_fn: overrides how EpisodeMetrics.completed is determined
            (default: default_success_fn).
        reward_override / ppo_override: JSON field overrides for
            RewardConfig / PPOConfig, read once at agent setup.
        steps_already_trained: steps from earlier bursts, for entropy/LR
            decay progress.

    Returns:
        (episode metrics, world).
    """
    _set_or_clear_env("PPO_AGENT_INIT_CHECKPOINT", init_checkpoint)
    _set_or_clear_env("PPO_AGENT_SAVE_CHECKPOINT", save_checkpoint)
    _set_or_clear_env("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE", str(rollout_steps_override) if rollout_steps_override else None)
    _set_or_clear_env("PPO_AGENT_REWARD_OVERRIDE", json.dumps(reward_override) if reward_override else None)
    _set_or_clear_env("PPO_AGENT_PPO_OVERRIDE", json.dumps(ppo_override) if ppo_override else None)
    _set_or_clear_env(
        "PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE",
        str(steps_already_trained) if steps_already_trained is not None else None,
    )

    world = build_world(scenario=scenario, seed=seed, train=train, opponents=opponents)

    episodes = [run_one_round(world, success_fn=success_fn) for _ in range(n_rounds)]
    world.end()
    return episodes, world


def training_reward_mean(world: BombeRLeWorld) -> Optional[float]:
    """Mean per-step training reward of a train=True burst; None otherwise."""
    fake_self = world.agents[0].backend.runner.fake_self
    history = getattr(fake_self, "reward_history", None)
    if not history:
        return None
    return float(np.mean(history))


def training_loss_means(world: BombeRLeWorld) -> Tuple[Optional[float], Optional[float]]:
    """Mean (policy_loss, value_loss) over all PPO updates in a train=True
    burst; (None, None) if not training or no update fired.
    """
    fake_self = world.agents[0].backend.runner.fake_self
    history = getattr(fake_self, "loss_history", None)
    if not history:
        return None, None
    return (
        float(np.mean([h["policy_loss"] for h in history])),
        float(np.mean([h["value_loss"] for h in history])),
    )


def training_loss_stats(world: BombeRLeWorld) -> Optional[Dict[str, float]]:
    """Mean/min/max over all PPO updates in a train=True burst for
    policy_loss, value_loss, entropy, gradient_norm and explained_variance.
    Exact burst-wide statistics, since every update uses the same number of
    minibatches. None if not training or no update fired.
    """
    fake_self = world.agents[0].backend.runner.fake_self
    history = getattr(fake_self, "loss_history", None)
    if not history:
        return None

    stats: Dict[str, float] = {}
    for metric in ("policy_loss", "value_loss", "entropy", "gradient_norm"):
        stats[f"{metric}_mean"] = float(np.mean([h[metric] for h in history]))
        stats[f"{metric}_min"] = float(np.min([h[f"{metric}_min"] for h in history]))
        stats[f"{metric}_max"] = float(np.max([h[f"{metric}_max"] for h in history]))
    ev_values = [h["explained_variance"] for h in history]
    stats["explained_variance_mean"] = float(np.nanmean(ev_values))
    stats["explained_variance_min"] = float(np.nanmin(ev_values))
    stats["explained_variance_max"] = float(np.nanmax(ev_values))
    return stats


def aggregate_metrics(episodes: List[EpisodeMetrics]) -> dict:
    n = len(episodes)
    completions = [ep for ep in episodes if ep.completed]
    steps_per_coin = [ep.steps / ep.coins_collected for ep in episodes if ep.coins_collected > 0]
    return {
        "n_episodes": n,
        "coins_collected_mean": float(np.mean([ep.coins_collected for ep in episodes])) if n else float("nan"),
        "completion_rate": len(completions) / n if n else float("nan"),
        "steps_to_completion_mean": float(np.mean([ep.steps for ep in completions])) if completions else float("nan"),
        "steps_to_completion_median": float(np.median([ep.steps for ep in completions])) if completions else float("nan"),
        "steps_to_completion_std": float(np.std([ep.steps for ep in completions])) if completions else float("nan"),
        "steps_per_coin_mean": float(np.mean(steps_per_coin)) if steps_per_coin else float("nan"),
        "wait_fraction_mean": float(np.mean([ep.wait_fraction for ep in episodes])) if n else float("nan"),
        "invalid_action_count_total": int(np.sum([ep.invalid_action_count for ep in episodes])) if n else 0,
        # Only own_cause counts as a failure; contested_tile is reported for visibility.
        "invalid_action_own_cause_count_total": int(np.sum([ep.invalid_action_own_cause_count for ep in episodes])) if n else 0,
        "invalid_action_contested_tile_count_total": int(np.sum([ep.invalid_action_contested_tile_count for ep in episodes])) if n else 0,
        "immediate_reverse_count_mean": float(np.mean([ep.immediate_reverse_count for ep in episodes])) if n else float("nan"),
        "episode_length_mean": float(np.mean([ep.steps for ep in episodes])) if n else float("nan"),
        "self_kill_rate": (sum(1 for ep in episodes if ep.self_kill) / n) if n else float("nan"),
        "crates_destroyed_mean": float(np.mean([ep.crates_destroyed for ep in episodes])) if n else float("nan"),
        "bombs_dropped_mean": float(np.mean([ep.bombs_dropped for ep in episodes])) if n else float("nan"),
        # BOMB actions chosen per round (from the action sequence), distinct
        # from bombs_dropped_mean (bombs actually placed).
        "bombs_placed_mean": (
            float(np.mean([(ep.actions or []).count("BOMB") for ep in episodes])) if n else float("nan")
        ),
        "opponent_kills_mean": float(np.mean([ep.opponent_kills for ep in episodes])) if n else float("nan"),
        "got_killed_by_opponent_rate": (sum(1 for ep in episodes if ep.got_killed_by_opponent) / n) if n else float("nan"),
        # Official scoring rule, independent of training rewards.
        "score_mean": float(np.mean([ep.score for ep in episodes])) if n else float("nan"),
        "score_total": float(np.sum([ep.score for ep in episodes])) if n else float("nan"),
        # Deprecated: ignores the travel cost to the bombing position. Kept for
        # existing CSV columns; use bombing_target_optimal_rate instead.
        "bomb_efficiency": (
            float(np.mean([ep.crates_destroyed for ep in episodes])) / float(np.mean([ep.bombs_dropped for ep in episodes]))
            if n and float(np.mean([ep.bombs_dropped for ep in episodes])) > 0 else float("nan")
        ),
    }


def ring_bell():
    """Rings the terminal bell."""
    print("\a", end="", flush=True)


def open_log_file(stage_name: str):
    """Opens agent_code/rhine/logs/<stage>_run_<timestamp>.log for appending."""
    from datetime import datetime

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{stage_name}_run_{timestamp}.log"
    return open(log_path, "a"), log_path


def log_print(log_file, message: str):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


# --- metrics CSV persistence + PNG plotting ---
# Charts are read back from CSV so they can be regenerated independently.

def write_metrics_csv_row(csv_path: Path, row: Dict):
    """Appends one row to a CSV, writing the header only for a new file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_episode_csv(csv_path: Path, episodes: List[EpisodeMetrics]):
    """Writes one CSV row per episode."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round_index", "steps", "coins_collected", "completed",
        "invalid_action_count", "invalid_action_own_cause_count", "invalid_action_contested_tile_count",
        "wait_count", "wait_fraction", "immediate_reverse_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            writer.writerow({
                "round_index": ep.round_index,
                "steps": ep.steps,
                "coins_collected": ep.coins_collected,
                "completed": ep.completed,
                "invalid_action_count": ep.invalid_action_count,
                "invalid_action_own_cause_count": ep.invalid_action_own_cause_count,
                "invalid_action_contested_tile_count": ep.invalid_action_contested_tile_count,
                "wait_count": ep.wait_count,
                "wait_fraction": round(ep.wait_fraction, 4),
                "immediate_reverse_count": ep.immediate_reverse_count,
            })


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def plot_training_curves(csv_path: Path, png_path: Path, x_col: str, y_cols: List[str], title: str = ""):
    """Plots each y_col against x_col from a CSV, one subplot per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv_rows(csv_path)
    x = [float(r[x_col]) for r in rows]

    fig, axes = plt.subplots(len(y_cols), 1, figsize=(8, 3 * len(y_cols)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, y_col in zip(axes, y_cols):
        y = [float(r[y_col]) if r[y_col] not in ("", "nan") else float("nan") for r in rows]
        ax.plot(x, y, marker="o")
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(x_col)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def plot_comparison(csv_paths: Dict[str, Path], png_path: Path, x_col: str, y_cols: List[str], title: str = ""):
    """Overlays several runs' CSVs, one line per label and one subplot per y_col."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(y_cols), 1, figsize=(8, 3 * len(y_cols)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for label, csv_path in csv_paths.items():
        rows = _read_csv_rows(csv_path)
        x = [float(r[x_col]) for r in rows]
        for ax, y_col in zip(axes, y_cols):
            y = [float(r[y_col]) if r[y_col] not in ("", "nan") else float("nan") for r in rows]
            ax.plot(x, y, marker="o", markersize=3, label=label)

    for ax, y_col in zip(axes, y_cols):
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(x_col)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def plot_episode_distribution(csv_path: Path, png_path: Path, title: str = ""):
    """Plots per-episode distributions (coins collected, episode length) from
    a per-episode CSV written by write_episode_csv().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv_rows(csv_path)
    coins = [float(r["coins_collected"]) for r in rows]
    steps = [float(r["steps"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(coins, bins=range(0, 11))
    axes[0].set_xlabel("coins_collected")
    axes[0].set_ylabel("episode count")
    axes[0].set_title("Coins collected per episode")

    axes[1].hist(steps, bins=20)
    axes[1].set_xlabel("steps")
    axes[1].set_title("Episode length")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
