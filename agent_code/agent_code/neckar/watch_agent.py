"""Watch a trained neckar checkpoint play, with the GUI on.

Playing a specific checkpoint by hand is fiddly for two reasons, and this script handles
both so the command is just "watch this file":

1. callbacks.setup() only ever loads one exact filename,
   "actor-critic-<model>-<behavior>-<scenario>-<n>-rounds-<m>-opponents.pt", derived from
   the arguments the game is launched with. A checkpoint saved under any other name - a
   training checkpoint ("...-ep500.pt"), a sweep result with a suffix, or anything renamed -
   is invisible to it. This script stages a copy under the name the run will look for, in a
   scratch directory pointed at by NECKAR_RUN_DIR, so the original file is never touched and
   the agent directory stays clean.

2. CNN checkpoints from before the 8 spatial scalars were added store their conv/fc layers
   under the old Sequential-indexed names (actor_base.0 ... actor_base.7). setup() does a
   strict load and rejects those. models.CNN.warm_start_from_old_cnn() already knows how to
   translate them, so this script detects that format and converts it once while staging.
   The strict load in setup() is left alone deliberately: it is what stops a silently
   untrained network from being mistaken for a working agent.

Usage:
    python watch_agent.py --checkpoint actor-critic-cnn-peaceful-classic.pt
    python watch_agent.py --checkpoint <path> --policy greedy --turn-based
    python watch_agent.py --checkpoint <path> --scenario coin-heaven --opponents 0
    python watch_agent.py --checkpoint <path> --n-rounds 3 --save-replay

With the GUI on, main.py pauses on an end-of-round screen until a key is pressed before
starting the next round - that pause is not a hang. Closing the window quits immediately and
skips whatever rounds are left.

--model/--behavior/--scenario are inferred from a standard "actor-critic-..." filename when
they are not given, so usually only --checkpoint is needed.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

AGENT_DIR = Path(__file__).resolve().parent
MAIN_PY = AGENT_DIR / ".." / ".." / "main.py"
AGENT_NAME = "neckar"
MODEL_TYPES = ("mlp", "cnn")
N_ACTIONS = 6
N_SCALARS = 8


def known_behaviors() -> list:
    """Behavior names defined in agent_behavior, longest first.

    Longest first so "aggressive_1" is matched whole rather than being truncated to the
    shorter "aggressive" that is also a valid key.
    """
    try:
        from .agent_behavior import BASE_REWARDS, POTENTIAL_WEIGHTS
    except ImportError:
        from agent_behavior import BASE_REWARDS, POTENTIAL_WEIGHTS
    return sorted(set(BASE_REWARDS) | set(POTENTIAL_WEIGHTS), key=len, reverse=True)


def scenarios() -> list:
    """Scenario names the engine accepts, longest first (same reason as behaviors)."""
    sys.path.insert(0, str((AGENT_DIR / ".." / "..").resolve()))
    import settings as s
    return sorted(s.SCENARIOS, key=len, reverse=True)


def infer_from_filename(name: str) -> dict:
    """Pull model/behavior/scenario out of a standard checkpoint filename.

    Matched against the real name lists rather than by splitting on "-", because both
    behaviors ("aggressive_1") and scenarios ("coin-heaven") contain separators themselves.
    Returns only the fields it could identify; anything absent is left to the CLI defaults.
    """
    found = {}
    m = re.match(r"actor-critic-([a-z_]+)-(.+)", name)
    if not m:
        return found

    if m.group(1) in MODEL_TYPES:
        found["model"] = m.group(1)

    rest = m.group(2)
    for behavior in known_behaviors():
        if rest.startswith(behavior + "-"):
            found["behavior"] = behavior
            rest = rest[len(behavior) + 1:]
            break

    for scenario in scenarios():
        if rest.startswith(scenario):
            found["scenario"] = scenario
            break

    return found


def is_legacy_cnn(state_dict: dict) -> bool:
    """Whether this is a CNN checkpoint using the pre-spatial-scalars layer names."""
    return any("_base." in k for k in state_dict)


def stage_checkpoint(src: Path, dest: Path, model_type: str) -> None:
    """Copy src to dest under the name setup() will look for, converting if needed."""
    state_dict = torch.load(src, map_location="cpu")

    if model_type == "cnn" and is_legacy_cnn(state_dict):
        from importlib import import_module
        try:
            CNN = import_module(".models.CNN", package=__package__)
        except (ImportError, TypeError):
            CNN = import_module("models.CNN")

        print(f"[watch] {src.name} uses the old CNN layer names - converting while staging")
        model = CNN.CNNActorCritic(action_dim=N_ACTIONS, n_scalars=N_SCALARS)
        # Raises if any parameter would be left uninitialised, so a silently-partial
        # conversion cannot slip through.
        CNN.warm_start_from_old_cnn(model, str(src))
        torch.save(model.state_dict(), dest)
    else:
        shutil.copy(src, dest)

    print(f"[watch] staged -> {dest}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint to watch. Any filename - it is staged under the "
                             "name setup() expects, and the original is left untouched.")
    parser.add_argument("--model", choices=MODEL_TYPES, default=None,
                        help="Defaults to the model type in the checkpoint's filename, else cnn")
    parser.add_argument("--behavior", default=None,
                        help="Defaults to the behavior in the checkpoint's filename, else peaceful")
    parser.add_argument("--scenario", default=None,
                        help="Defaults to the scenario in the checkpoint's filename, else classic")
    parser.add_argument("--opponents", default="random_agent,random_agent,random_agent",
                        help="Comma-separated opponent agent names, or a count to fill with "
                             "random_agent. Use 0 or '' to play alone. (default: %(default)s)")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Rounds to watch (default: %(default)s)")
    parser.add_argument("--policy", choices=["greedy", "sample"], default="greedy",
                        help="Action selection at inference, via NECKAR_EVAL_POLICY. greedy "
                             "shows the policy's preferred move; sample shows the stochastic "
                             "behavior it was trained with. (default: %(default)s)")
    parser.add_argument("--update-interval", type=float, default=0.3,
                        help="Seconds between steps - raise it to follow the agent by eye "
                             "(default: %(default)s)")
    parser.add_argument("--turn-based", action="store_true",
                        help="Advance one step per key press, to inspect single decisions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the world's RNG so a run can be replayed identically")
    parser.add_argument("--save-replay", action="store_true",
                        help="Also store the game for later 'main.py replay <file>'")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (for scripted checks rather than watching)")
    parser.add_argument("--run-dir", default=None,
                        help="Where to stage the checkpoint. Defaults to a temporary "
                             "directory that is removed afterwards.")
    args = parser.parse_args()

    src = Path(args.checkpoint).expanduser()
    if not src.is_file():
        # A bare filename most likely means a checkpoint sitting in the agent directory.
        candidate = AGENT_DIR / args.checkpoint
        if candidate.is_file():
            src = candidate
        else:
            parser.error(f"checkpoint not found: {args.checkpoint}")
    src = src.resolve()

    inferred = infer_from_filename(src.name)
    model_type = args.model or inferred.get("model", "cnn")
    behavior = args.behavior or inferred.get("behavior", "peaceful")
    scenario = args.scenario or inferred.get("scenario", "classic")

    only_from_defaults = {k for k in ("model", "behavior", "scenario")
                          if getattr(args, k) is None and k not in inferred}
    print(f"[watch] checkpoint : {src.name}")
    print(f"[watch] model={model_type!r} behavior={behavior!r} scenario={scenario!r}"
          + (f"  (not in filename, using defaults for: {sorted(only_from_defaults)})"
             if only_from_defaults else "  (inferred from filename)"))

    # Opponents: a count fills with random_agent, a list is used as given.
    raw = args.opponents.strip()
    if raw.isdigit():
        opponents = ["random_agent"] * int(raw)
    else:
        opponents = [a.strip() for a in raw.split(",") if a.strip()]

    # The filename setup() will look up, built from the very arguments passed below so the
    # two cannot disagree.
    staged_name = (f"actor-critic-{model_type}-{behavior}-{scenario}"
                   f"-{args.n_rounds}-rounds-{len(opponents)}-opponents.pt")

    tmp = None
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="neckar-watch-")
        run_dir = Path(tmp)

    try:
        stage_checkpoint(src, run_dir / staged_name, model_type)

        env = os.environ.copy()
        # NECKAR_RUN_DIR is what callbacks joins the checkpoint name onto, so the staged
        # copy is found without putting anything in the agent directory.
        env["NECKAR_RUN_DIR"] = str(run_dir)
        env["NECKAR_MODEL_TYPE"] = model_type
        env["NECKAR_BEHAVIOR"] = behavior
        env["NECKAR_SCENARIO"] = scenario
        env["NECKAR_EVAL_POLICY"] = args.policy
        # Never let a stale warm start from the calling shell alter what is being observed.
        env.pop("NECKAR_WARM_START", None)

        # sys.executable, not "python": a bare "python" may not be on PATH in a conda env.
        cmd = [sys.executable, str(MAIN_PY.resolve()), "play",
               "--agents", AGENT_NAME, *opponents,
               "--scenario", scenario,
               "--n-rounds", str(args.n_rounds),
               "--update-interval", str(args.update_interval)]
        if args.turn_based:
            cmd.append("--turn-based")
        if args.no_gui:
            cmd.append("--no-gui")
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.save_replay:
            cmd.append("--save-replay")

        print(f"[watch] policy={args.policy} opponents={opponents or ['(none)']} "
              f"rounds={args.n_rounds}")
        if not args.no_gui:
            # Worth stating plainly: main.py's world_controller blocks on an end-of-round
            # screen until a key is pressed, which looks exactly like a freeze if you are
            # not expecting it. Closing the window instead returns immediately, so the
            # remaining rounds are skipped rather than the run being "finished".
            print(f"[watch] GUI mode: after each round the end screen waits for a key press "
                  f"before the next one starts (press a movement key or Escape). "
                  f"{args.n_rounds} round(s) queued.")
            print("[watch] closing the window quits straight away and skips any rounds left")
        if args.turn_based:
            print("[watch] turn-based: each step also needs a movement key press")
        print(f"[watch] running: {' '.join(cmd)}\n")

        return subprocess.run(cmd, env=env).returncode

    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
