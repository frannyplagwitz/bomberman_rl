"""Rotational symmetry folding for the LUT state key.

The classic/loot-crate maps are symmetric under 90-degree rotation (same wall
skeleton, same crate density everywhere), so from the LUT's point of view "an
enemy 2 tiles to my north-east, standing near the top-right corner" and "an
enemy 2 tiles to my north-east, standing near the top-left corner" are the same
underlying situation, just viewed from a different starting corner. Without
folding these together the table has to independently learn the same policy
four times, once per corner the agent happens to start in.

canonicalize() rotates the whole game state so the agent's start corner is
always treated as top-left, before get_table_key()/action_mask() ever see it.
Everything downstream - danger/escape/coin/enemy directions, the action mask -
is computed on this canonical view and so is automatically self-consistent;
only the final chosen action needs to be rotated back (uncanonicalize_action)
into real-world directions before it's returned to the game engine, since the
Q-table itself only ever operates in the canonical frame.

All rotations here are expressed as a single quarter-turn (dx, dy) -> (-dy, dx)
(or, for an absolute position, its grid-folded counterpart) applied k times,
rather than four hand-written per-corner coordinate formulas - one rule, reused
for every corner, rather than four cases that can silently drift out of sync
with each other.
"""

import numpy as np

# Quarter-turns needed to bring each starting corner to the top-left, i.e. to
# canonical form. Verified against np.rot90's rotation direction: rotating the
# field by ROTATION_BY_CORNER[corner] quarter-turns moves that corner's
# content into the top-left.
ROTATION_BY_CORNER = {
    "top-left": 0,
    "bottom-left": 1,
    "bottom-right": 2,
    "top-right": 3,
}

# Movement actions as (dx, dy) unit vectors, and the reverse lookup. BOMB/WAIT
# have no direction and are always rotation-invariant.
ACTION_VECTORS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
VECTOR_TO_ACTION = {v: a for a, v in ACTION_VECTORS.items()}


def get_start_corner(agent_pos: tuple, rows: int, cols: int) -> str:
    """Which quadrant of the board an agent spawned in, used to fix the rotation
    for the rest of that episode (the corner is only meaningful at step 1 -
    the agent will walk away from it immediately)."""

    x, y = agent_pos
    if y < rows // 2:
        return "top-left" if x < cols // 2 else "top-right"
    return "bottom-left" if x < cols // 2 else "bottom-right"


def _rotate_xy(x: int, y: int, k: int, n: int = None) -> tuple:
    """Rotate (x, y) by k quarter-turns counter-clockwise.

    n=None rotates a direction vector around the origin (used for actions).
    n=<grid side length> rotates an absolute position, folding it around an
    n x n grid the same way np.rot90(array, k=k) folds the array itself.
    """

    k %= 4
    m = None if n is None else n - 1

    for _ in range(k):
        x, y = (-y, x) if m is None else (m - y, x)

    return x, y


def rotate_action(action: str, k: int) -> str:
    """Rotate a movement action by k quarter-turns; BOMB/WAIT pass through unchanged."""

    vector = ACTION_VECTORS.get(action)
    if vector is None:
        return action

    return VECTOR_TO_ACTION[_rotate_xy(*vector, k)]


def uncanonicalize_action(action: str, rotation_k: int) -> str:
    """Inverse of rotate_action: map a canonical-frame action back to the real
    (unrotated) direction the game engine expects."""

    return rotate_action(action, -rotation_k)


def canonicalize(game_state: dict, rotation_k: int) -> dict:
    """Return a copy of game_state rotated by rotation_k quarter-turns so the
    agent's start corner always appears top-left. rotation_k=0 returns the
    state unchanged (still a shallow-ish copy, safe to hand to read-only code)."""

    if rotation_k % 4 == 0:
        return game_state

    field = game_state["field"]
    n = field.shape[0]

    def rotate_pos(pos):
        return _rotate_xy(pos[0], pos[1], rotation_k, n)

    name, score, bomb_available, self_pos = game_state["self"]

    return {
        **game_state,
        "field": np.rot90(field, k=rotation_k).copy(),
        "explosion_map": np.rot90(game_state["explosion_map"], k=rotation_k).copy(),
        "coins": [rotate_pos(c) for c in game_state["coins"]],
        "bombs": [(rotate_pos(pos), timer) for pos, timer in game_state["bombs"]],
        "others": [(n_, s_, b_, rotate_pos(pos)) for n_, s_, b_, pos in game_state["others"]],
        "self": (name, score, bomb_available, rotate_pos(self_pos)),
    }
