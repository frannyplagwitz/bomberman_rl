"""Read-only diagnostic: compares two stricter bombing-escape checks with the
production rule (at least one safe escape path, has_safe_escape_after_bombing()).

Both are reimplemented locally on the same hypothetical-danger construction
as the production check.

Scheme A ("full-path robustness"): at least two escape paths that are
pairwise tile-disjoint beyond the start tile (greedy search over first-step
direction pairs, not an exact disjoint-paths solve).

Scheme B ("first-step robustness"): at least two first-step directions that
each lead to a safe continuation, without requiring disjoint paths.

Samples are real bomb_available=True decision points from a `classic` rollout
(random-init policy, one peaceful_agent and two coin_collector_agent).
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from agent_code.rhine import config as cfg
from agent_code.rhine import train as train_module
from agent_code.rhine.scripts import common
from agent_code.rhine.state_processing import (
    DIRECTIONS,
    SAFETY_HORIZON,
    blast_coords,
    extract_semantic_state,
    is_free,
    neighbor_tile,
)


def _exists_safe_path_with_parent(start, start_offset, field_arr, blocked, danger_offsets, max_offset):
    """Same search as state_processing.exists_safe_path(), but returns the
    (tile, offset) path to the first permanently safe state, or None.
    """
    if start_offset in danger_offsets.get(start, ()):
        return None
    if not {o for o in danger_offsets.get(start, ()) if o > start_offset}:
        return [(start, start_offset)]

    parent = {}
    visited = {(start, start_offset)}
    queue = deque([(start, start_offset)])
    while queue:
        pos, offset = queue.popleft()
        if offset >= max_offset:
            continue
        next_offset = offset + 1
        candidates = [pos] + [
            neighbor_tile(pos, d)
            for d in DIRECTIONS
            if is_free(field_arr, *neighbor_tile(pos, d), blocked=blocked)
        ]
        for nxt in candidates:
            if next_offset in danger_offsets.get(nxt, ()):
                continue
            state = (nxt, next_offset)
            if state in visited:
                continue
            visited.add(state)
            parent[state] = (pos, offset)
            if not {o for o in danger_offsets.get(nxt, ()) if o > next_offset}:
                path = [state]
                cur = state
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            queue.append(state)
    return None


def _first_step_candidates(tile, field_arr, blocked):
    return [
        neighbor_tile(tile, d)
        for d in DIRECTIONS
        if is_free(field_arr, *neighbor_tile(tile, d), blocked=blocked)
    ]


def evaluate_schemes(tile, field_arr, blocked, danger_offsets, max_offset):
    """Returns (scheme_b_pass, scheme_a_pass) for `tile`; `blocked` and
    `danger_offsets` must already include the hypothetical bomb at `tile`.
    """
    neighbors = _first_step_candidates(tile, field_arr, blocked)
    passing_paths = {}
    for n in neighbors:
        path = _exists_safe_path_with_parent(n, 1, field_arr, blocked, danger_offsets, max_offset)
        if path is not None:
            passing_paths[n] = path

    scheme_b_pass = len(passing_paths) >= 2

    scheme_a_pass = False
    keys = list(passing_paths.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            tiles_i = {t for t, _ in passing_paths[keys[i]]}
            tiles_j = {t for t, _ in passing_paths[keys[j]]}
            # Disjoint beyond the shared starting tile itself.
            if tiles_i.isdisjoint(tiles_j - {tile}) and tiles_j.isdisjoint(tiles_i - {tile}):
                scheme_a_pass = True
                break
        if scheme_a_pass:
            break

    return scheme_b_pass, scheme_a_pass


def main():
    log_file, log_path = common.open_log_file("evaluate_escape_robustness")
    common.log_print(log_file, f"Escape-route robustness comparison -- log file: {log_path}")

    torch.manual_seed(0)
    samples = []

    original_extract = train_module.extract_semantic_state

    def wrapped(gs):
        sem = original_extract(gs)
        if sem.bomb_available:
            samples.append((gs["field"], sem.self_pos, sem.blocked, sem.danger_offsets))
        return sem

    train_module.extract_semantic_state = wrapped
    episodes, world = common.run_episodes(
        n_rounds=15,
        scenario="classic",
        seed=127,
        train=True,
        init_checkpoint=None,
        save_checkpoint=str(common.LOGS_DIR / "evaluate_escape_robustness_scratch.pt"),
        rollout_steps_override=512,
        opponents=["peaceful_agent", "coin_collector_agent", "coin_collector_agent"],
    )
    train_module.extract_semantic_state = original_extract

    common.log_print(log_file, f"Total bomb_available=True decision points sampled: {len(samples)}")

    n_currently_legal = 0
    n_scheme_b_pass = 0
    n_scheme_a_pass = 0

    for field_arr, tile, blocked, danger_offsets in samples:
        hypothetical_danger = {t: set(offsets) for t, offsets in danger_offsets.items()}
        for blast_tile in blast_coords(field_arr, tile, cfg.BOMB_POWER):
            hypothetical_danger.setdefault(blast_tile, set()).update({cfg.BOMB_TIMER, cfg.BOMB_TIMER + 1})
        blocked_with_new_bomb = blocked | {tile}

        currently_legal = _exists_safe_path_with_parent(
            tile, 0, field_arr, blocked_with_new_bomb, hypothetical_danger, SAFETY_HORIZON
        ) is not None
        if not currently_legal:
            continue
        n_currently_legal += 1

        scheme_b_pass, scheme_a_pass = evaluate_schemes(
            tile, field_arr, blocked_with_new_bomb, hypothetical_danger, SAFETY_HORIZON
        )
        n_scheme_b_pass += scheme_b_pass
        n_scheme_a_pass += scheme_a_pass

    common.log_print(log_file, f"\nCurrently legal to BOMB (>=1 safe escape path, existing rule): {n_currently_legal}")
    if n_currently_legal:
        b_loss = n_currently_legal - n_scheme_b_pass
        a_loss = n_currently_legal - n_scheme_a_pass
        common.log_print(
            log_file,
            f"Scheme B (>=2 first-step directions): still legal={n_scheme_b_pass} "
            f"({100*n_scheme_b_pass/n_currently_legal:.1f}%), newly illegal={b_loss} "
            f"({100*b_loss/n_currently_legal:.1f}%)",
        )
        common.log_print(
            log_file,
            f"Scheme A (>=2 disjoint full paths): still legal={n_scheme_a_pass} "
            f"({100*n_scheme_a_pass/n_currently_legal:.1f}%), newly illegal={a_loss} "
            f"({100*a_loss/n_currently_legal:.1f}%)",
        )

    common.ring_bell()


if __name__ == "__main__":
    main()
