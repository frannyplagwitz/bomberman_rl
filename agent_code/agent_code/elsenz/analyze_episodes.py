"""Aggregate one or more episodes-*.csv logs (episode_logger.py) into round-number
buckets, to check whether a LUT training run is actually learning.

Usage:
    python -m agent_code.qfiac_agent.analyze_episodes --bucket-size 100 FILE.csv [FILE2.csv ...]

Multiple files are concatenated first (e.g. a phase split across run_ids because several
workers happened to start in the same second - see LUT_AGENT.md), then bucketed by each
row's own "episodes" (per-worker round number), so results reflect training progress
within a run regardless of how many files it's split across.
"""

import argparse
import csv
from collections import defaultdict


def to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def avg(values):
    return sum(values) / len(values) if values else float("nan")


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def summarize(rows, bucket_size: int):
    buckets = defaultdict(list)
    for row in rows:
        episode = int(row["episodes"])
        buckets[(episode - 1) // bucket_size].append(row)

    summary = []
    for bucket in sorted(buckets.keys()):
        bucket_rows = buckets[bucket]
        lo, hi = bucket * bucket_size + 1, bucket * bucket_size + bucket_size

        reward = [to_float(r["mean_reward"]) for r in bucket_rows if to_float(r["mean_reward"]) is not None]
        loss = [to_float(r["total_loss"]) for r in bucket_rows if to_float(r["total_loss"]) is not None]
        steps = [to_float(r["steps"]) for r in bucket_rows if to_float(r["steps"]) is not None]
        coins = [to_float(r["coins_collected"]) for r in bucket_rows if to_float(r["coins_collected"]) is not None]
        survived = [to_float(r["survived_round"]) for r in bucket_rows if to_float(r["survived_round"]) is not None]
        killed_self = [to_float(r["killed_self"]) for r in bucket_rows if to_float(r["killed_self"]) is not None]
        got_killed = [to_float(r["got_killed"]) for r in bucket_rows if to_float(r["got_killed"]) is not None]

        summary.append({
            "lo": lo, "hi": hi, "n": len(bucket_rows),
            "mean_reward": avg(reward), "td_loss": avg(loss), "steps": avg(steps),
            "coins": avg(coins), "survived_pct": 100.0 * avg(survived),
            "deaths_pct": 100.0 * avg([s + g for s, g in zip(killed_self, got_killed)])
                if killed_self and got_killed else float("nan"),
        })
    return summary


def print_summary(summary, title=None):
    if title:
        print(f"=== {title} ===")
    header = f'{"rounds":>12} | {"n":>5} | {"reward":>8} | {"TD_loss":>8} | {"steps":>7} | {"coins":>6} | {"survived%":>9} | {"deaths%":>8}'
    print(header)
    for row in summary:
        rng = f'{row["lo"]}-{row["hi"]}'
        print(f'{rng:>12} | {row["n"]:5d} | {row["mean_reward"]:8.4f} | {row["td_loss"]:8.4f} | '
              f'{row["steps"]:7.1f} | {row["coins"]:6.2f} | {row["survived_pct"]:8.1f}% | {row["deaths_pct"]:7.1f}%')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="episodes-*.csv file(s) to aggregate together")
    parser.add_argument("--bucket-size", type=int, default=100, help="rounds per bucket (default 100)")
    parser.add_argument("--title", default=None, help="label printed above the table")
    args = parser.parse_args()

    rows = load_rows(args.files)
    print_summary(summarize(rows, args.bucket_size), title=args.title or f"{len(rows)} episode-rows")


if __name__ == "__main__":
    main()
