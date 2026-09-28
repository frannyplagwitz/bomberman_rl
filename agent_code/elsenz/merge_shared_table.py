"""Standalone entry point for combining one training run's parallel workers' end-of-run
LUT contributions into a single canonical table.

Each worker (a `main.py play` process running qfiac_agent with model_type=lut) trains its
own private table with zero cross-process I/O during training, then - once, right before it
exits - drops a (weight, visit_counts) "contribution" file (see train.py:end_of_round and
models/LUT.py:save_contribution). This script folds every contribution left behind by one
training run into the canonical table those workers warm-started from, weighted by how many
times each visited a given (state, action) cell, so it's safe to run after every worker has
exited (train_lut_parallel.sh does exactly that, right after `wait`).

Usage:
    python -m agent_code.qfiac_agent.merge_shared_table \
        --behavior peaceful --scenario classic --num-opponents 3

(ELSENZ_TABLE_NAME, if set in the environment, overrides behavior/scenario/num-opponents for
path purposes exactly like it does for the agents themselves - see
callbacks.get_shared_table_path.)
"""

import argparse
from types import SimpleNamespace

from . import callbacks
from . import LUT as lut_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--num-opponents", type=int, required=True)
    args = parser.parse_args()

    # get_shared_table_path()/get_contributions_dir() only read these 3 attributes off
    # `self` (plus ELSENZ_TABLE_NAME from the environment). Reusing them here - instead of
    # re-deriving the same path format a second time - is what keeps this script from
    # silently drifting out of sync with the paths the agents themselves actually use.
    ns = SimpleNamespace(behavior=args.behavior, scenario=args.scenario, num_opponents=args.num_opponents)

    canonical_path = callbacks.get_shared_table_path(ns)
    contributions_dir = callbacks.get_contributions_dir(ns)

    num_merged, num_states = lut_models.merge_tables(canonical_path, contributions_dir)
    print(f"Merged {num_merged} contribution(s) into {canonical_path} ({num_states} states)")


if __name__ == "__main__":
    main()
