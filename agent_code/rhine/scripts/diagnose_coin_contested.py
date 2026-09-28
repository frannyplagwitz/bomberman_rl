"""Read-only diagnostic: coin_contested trigger rate in a deterministic
evaluation of a trained checkpoint.

Hooks callbacks.extract_semantic_state to record each step's coin_contested
value together with whether a coin was collected, so the trigger rate can be
compared with actual pickup outcomes.

Usage:
  python -m agent_code.rhine.scripts.diagnose_coin_contested \\
      --checkpoint models/task3_stage_c_seed0.pt \\
      --opponents coin_collector_agent --eval-seed 1000 --n-rounds 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_code.rhine import callbacks as callbacks_module
from agent_code.rhine import config as cfg
from agent_code.rhine.scripts import common

SCENARIO = "classic"


def _install_coin_contested_recorder():
    recorded = []
    original = callbacks_module.extract_semantic_state

    def wrapped(game_state):
        semantic = original(game_state)
        recorded.append(semantic.coin_contested)
        return semantic

    callbacks_module.extract_semantic_state = wrapped

    def restore():
        callbacks_module.extract_semantic_state = original

    return recorded, restore


def run_for_checkpoint(checkpoint_path: str, opponents, eval_seed: int, n_rounds: int):
    cfg.ENABLE_OSCILLATION_BREAKER = True
    cfg.ENABLE_NO_BOMB_WHEN_BOARD_CLEARED = True
    cfg.ENABLE_BOMBING_TARGET_SCORING_FORMULA = True

    recorded, restore = _install_coin_contested_recorder()
    try:
        episodes, _ = common.run_episodes(
            n_rounds=n_rounds, scenario=SCENARIO, seed=eval_seed,
            train=False, init_checkpoint=checkpoint_path, opponents=opponents,
        )
    finally:
        restore()

    n_true = sum(1 for v in recorded if v)
    n_total = len(recorded)
    return {
        "n_total": n_total,
        "n_true": n_true,
        "rate": n_true / n_total if n_total else float("nan"),
        "coins_collected_mean": sum(ep.coins_collected for ep in episodes) / len(episodes) if episodes else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--n-rounds", type=int, default=100)
    args = parser.parse_args()

    log_file, log_path = common.open_log_file("coin_contested_scan")
    common.log_print(log_file, f"coin_contested scan -- log: {log_path}")
    common.log_print(log_file, f"scenario={SCENARIO} opponents={args.opponents} eval_seed={args.eval_seed} "
                                f"n_rounds={args.n_rounds}")

    for checkpoint in args.checkpoint:
        r = run_for_checkpoint(checkpoint, args.opponents, args.eval_seed, args.n_rounds)
        common.log_print(
            log_file,
            f"\n=== {checkpoint} ===\n"
            f"coin_contested: True on {r['n_true']}/{r['n_total']} steps ({r['rate']:.4f}) "
            f"coins_collected_mean={r['coins_collected_mean']:.2f}",
        )

    log_file.close()
    common.ring_bell()


if __name__ == "__main__":
    main()
