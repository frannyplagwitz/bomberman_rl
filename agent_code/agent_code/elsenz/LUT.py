import os
import contextlib
import uuid
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is POSIX-only
    fcntl = None


class TabularQAgent(nn.Module): 
    """Lookup table (LUT) for Tabular Q-Learning reinforcement learning 
       Manages table of action-state values (Q(s,a))
       Q: state_id -> Q(s,a) for each action 
       
       Action selection is epsilon-greedy and is handled by callbacks
       Learning is done via TD update 
    """
    

    # Actions will be ["UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"]
    # Actions will be ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"] <- used
        
    def __init__(self, num_states: int, action_dim: int, lr: float = 0.1):
        
        super().__init__()
        
        self.num_states = num_states
        self.action_dim = action_dim 
        
        # Initialize q_table and embed with weights
        self.q_table = nn.Embedding(num_states, action_dim) # (state idx -> action-state value)
        nn.init.zeros_(self.q_table.weight)

        # How many times each (state, action) cell has been updated by *this* agent.
        # Not used for the update rule itself - only as a weight when several
        # independently-trained tables are merged afterwards (see merge_tables()):
        # a cell one process visited 200 times should count for more in the merge
        # than the same cell another process only stumbled into twice.
        self.register_buffer("visit_counts", torch.zeros(num_states, action_dim))


    def forward(self, state: torch.Tensor):
        """Returns Q(s, a) for the given state index/indices. shape = [batch, action_dim]"""
        return self.q_table(state)
        
              
    def get_value(self, state: torch.Tensor) -> torch.Tensor: 
        """Get state value from the best result stored at the passed state"""
        return self.q_table(state).max(dim=-1).values

        
    @torch.no_grad()
    def td_update(self, state_idx: int, action: int, reward: float, next_state_idx: "int | None", 
                  alpha: float = 0.1, gamma: float = 0.99,
                  next_legal_mask: "torch.Tensor | None" = None) -> float:
        
        """One-step TD tabular Q-learning
        
        Value-based update for Q-learning is defined as Q(s,a) <-- Q(s,a) + α[Target - Q(s,a)]
        We are using TD(0) as the target, which is defined as r + γ max_a' Q(s', a')
        
        How it works: 
        
        1. Grab the action-state value using the passed state index and current action 
        2. Grab the next action-state value using the next state index and the current action
        3. Calculate the update target (if no next state, will just be the reward)
        4. Calculate the difference between the target and the current action state value 
        5. Multiply by learning rate and add to current action state value 
        
        Since this is Q learning, and Q-learning behaves greedily, the next action will the action that results in 
        the max score
        
        Params: 
            state_idx: state index that values are stored at 
            action: action that was taken 
            reward:
            next_state_idx: next state index where values are stored at 
            alpha: learning rate, 0.1 by default
            gamma: discount factor, 0.99 by default
            next_legal_mask: optional bool tensor of shape (action_dim, ), 
                True where action is legal in the next state
        """
        
        # Grab the current action-state value. .item() takes a snapshot: indexing the
        # weight returns a view, so reading it after the in-place update below would see
        # the new value rather than the pre-update one.
        state_action = self.q_table.weight[state_idx, action].item()
        
        if next_state_idx is None:
            target = reward
        
        else: 
            next_q = self.q_table.weight[next_state_idx] # Grab the q-value tensor stored at next state idx
            
            # Need to check if that action is masked though before committing.
            # If mask, choose the unmasked action that results in the best Q-Value
            if next_legal_mask is not None:
                if next_legal_mask.any():
                    next_state_action = next_q[next_legal_mask].max().item()
                else:
                    next_state_action = next_q[5].item()   # WAIT, same fallback as act()
                    
            else:
                # No mask given: plain Q-learning, max over all actions
                next_state_action = next_q.max().item()
                
            target = reward + gamma * next_state_action
            
        # Find the td error and add value * learning rate to value in table
        td_error = target - state_action
        self.q_table.weight[state_idx, action] += alpha * td_error
        self.visit_counts[state_idx, action] += 1
        
        return td_error



class SharedTabularQAgent(TabularQAgent):
    """Tabular Q-agent whose table lives on disk instead of only in this process's memory.

    The game engine (agents.py) launches every agent - including several copies of the
    same qfiac_agent code for self-play - as its own OS process (multiprocessing.Process),
    so plain Python object state can't be shared between them. To let many bots pool their
    experience into ONE table, this class treats a file on disk as the source of truth,
    accessed through a memory-mapped array (np.memmap, MAP_SHARED) rather than by repeatedly
    torch.save()/torch.load()-ing the whole table:

    - refresh() copies the mmap's *current* content into the in-memory nn.Embedding used for
      forward passes. No lock needed: other processes' writes are visible through the shared
      OS page cache as soon as they happen, and reading a table mid-update by someone else can
      at worst observe one row's old-or-new (never torn - aligned float32 writes are atomic on
      every architecture this runs on) value a moment early or late, which is an acceptable
      staleness for choosing an action.
    - td_update() is the only place that needs the exclusive file lock, and only for as long
      as it takes to read the 2 rows involved (this state, and next_state for the bootstrap
      target) and write 1 updated value back - not the whole table - so lock hold time no
      longer scales with table size.

    An earlier version of this class did full-table torch.save/torch.load on every single
    refresh() and td_update() call. That works when only 1-2 bots share a table, but at 24-way
    concurrency (see train_lut_parallel.sh) it became the dominant cost: 24 processes each
    paying ~1-2ms to (de)serialize a 480KB tensor, entirely serialized behind one lock, pushed
    average per-step latency to 20-60ms and turned a ~80s/500-round game into hours. Switching
    to per-row mmap access (this version) removes that scaling problem at its root - lock hold
    time is now independent of table size.

    See LUT_AGENT.md ("Shared table for parallel training") for the full design rationale.
    """

    def __init__(self, num_states: int, action_dim: int, table_path: str, lr: float = 0.1):

        super().__init__(num_states, action_dim, lr)

        if fcntl is None:
            raise RuntimeError("SharedTabularQAgent requires a POSIX system (fcntl) for file locking")

        self.table_path = table_path
        self.lock_path = table_path + ".lock"

        os.makedirs(os.path.dirname(self.table_path) or ".", exist_ok=True)
        if not os.path.exists(self.lock_path):
            open(self.lock_path, "a").close()

        # Create the backing file (right size, zero-filled) if it doesn't exist yet. Locked
        # because many processes may race to do this on a brand-new table simultaneously;
        # np.memmap'ing a file another process has only partially written would be corrupt.
        with self._locked():
            expected_bytes = num_states * action_dim * 4  # float32
            if not os.path.exists(self.table_path) or os.path.getsize(self.table_path) != expected_bytes:
                np.zeros((num_states, action_dim), dtype=np.float32).tofile(self.table_path)

        # Kept open for the agent's lifetime (re-mmap'ing every call would add back exactly
        # the per-call overhead this class exists to avoid). MAP_SHARED (memmap's default for
        # mode="r+") is what makes this visible across processes, not just within one.
        self._mmap = np.memmap(self.table_path, dtype=np.float32, mode="r+",
                                shape=(num_states, action_dim))

        self.refresh()

    @contextlib.contextmanager
    def _locked(self):
        """Hold an exclusive OS file lock for the duration of the `with` block.

        Blocks (does not busy-wait) until any other process's read-modify-write on this
        same table file has finished, so a given row is never read-modify-written by two
        processes at once.
        """
        with open(self.lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def refresh(self):
        """Pull in updates other bots have made since this process last looked.

        Called before every action selection (see callbacks.act) so that a bot always
        chooses greedily/epsilon-greedily with respect to the most up-to-date shared
        knowledge, not a stale local snapshot. Lock-free by design - see class docstring.
        """
        self.q_table.weight.data.copy_(torch.from_numpy(np.asarray(self._mmap)))

    @torch.no_grad()
    def td_update(self, state_idx: int, action: int, reward: float, next_state_idx: "int | None",
                  alpha: float = 0.1, gamma: float = 0.99,
                  next_legal_mask: "torch.Tensor | None" = None) -> float:
        """One-step TD(0) tabular Q-learning, applied directly to the shared mmap.

        Same rule as TabularQAgent.td_update (Q(s,a) += alpha * [target - Q(s,a)]), but reads
        the current Q(s,a) and the next-state row straight from the shared table (not this
        process's possibly-stale local copy) and writes the single updated value back, all
        under one exclusive lock - the smallest critical section that's still race-free for a
        concurrent read-modify-write.
        """

        with self._locked():
            state_action = float(self._mmap[state_idx, action])

            if next_state_idx is None:
                target = reward
            else:
                # Mirror TabularQAgent.td_update: bootstrap over the next state's LEGAL
                # actions only, so the shared table learns the same target the private one
                # would, falling back to WAIT when nothing is legal (same as act()).
                next_q = self._mmap[next_state_idx]
                if next_legal_mask is not None:
                    legal = np.asarray(next_legal_mask, dtype=bool)
                    next_state_action = float(next_q[legal].max()) if legal.any() else float(next_q[5])
                else:
                    next_state_action = float(next_q.max())
                target = reward + gamma * next_state_action

            td_error = target - state_action
            self._mmap[state_idx, action] = state_action + alpha * td_error
            self._mmap.flush()

        # Keep this process's own in-memory copy consistent with what it just wrote, so a
        # forward pass immediately after td_update (without an intervening refresh()) still
        # reflects it.
        self.q_table.weight.data[state_idx, action] = float(self._mmap[state_idx, action])

        return td_error


# --- Train independently, merge afterwards -----------------------------------------------
#
# SharedTabularQAgent (above) synchronizes on every single step, which is correct and fast
# enough for a handful of bots, but even a cheap per-step lock doesn't scale cleanly to
# dozens of fully independent OS processes: measured at 24-way parallelism (see
# LUT_AGENT.md), the processes spent most of their time asleep waiting for that lock rather
# than computing - fixing the lock's own cost (mmap instead of whole-table torch.save) helped,
# but contention is still O(processes) by construction.
#
# The functions below take the opposite approach for that many-independent-bots case: each
# process trains an ordinary, private TabularQAgent with zero cross-process I/O during
# training at all (so it runs exactly as fast as one process alone would), then - once, at
# the very end of its run - writes its local (weight, visit_counts) as one "contribution".
# merge_tables() then combines every contributor's table into a single visit-count-weighted
# average, so a cell one process visited 200 times counts for more than the same cell another
# process only stumbled into a couple of times. This is what train_lut_parallel.sh now uses.

def save_contribution(agent: "TabularQAgent", contributions_dir: str, worker_id: str = None) -> str:
    """Dump this process's locally-trained (weight, visit_counts) as one contribution file
    for merge_tables() to combine later. No locking needed: every worker writes its own
    uniquely-named file, so concurrent writers never touch the same path.
    """

    os.makedirs(contributions_dir, exist_ok=True)
    worker_id = worker_id or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path = os.path.join(contributions_dir, f"{worker_id}.contribution")
    torch.save({"weight": agent.q_table.weight.data.clone(), "visits": agent.visit_counts.clone()}, path)
    return path


def load_canonical_table(agent: "TabularQAgent", canonical_path: str) -> bool:
    """Warm-start `agent`'s Q-values from a previously merged table, if one exists at
    canonical_path (see merge_tables()). Returns whether a table was actually loaded.

    Deliberately does NOT copy the canonical table's visit_counts into `agent` - those stay
    at their fresh-initialized zero. agent.visit_counts must end this run holding only the
    visits *this* process racked up, not the prior phase's already-accumulated total: every
    worker warm-starts from the same canonical table, so if each of them also inherited its
    visit_counts, every one of them would re-report that same prior evidence as if it were
    their own fresh contribution, and merge_tables() would count the prior phase's evidence
    once per worker instead of once, total - inflating its weight by ~(number of workers)
    on every single merge, compounding phase over phase. (Caught this from the numbers: after
    two 24-worker phases, cumulative visits were ~120M against a ~4.8M theoretical max for one
    phase's own new steps - which is explained almost exactly by 24 x phase 1's ~4.6M total.)
    Leaving visit_counts at zero keeps each worker's contribution to only what it actually
    observed this run; merge_tables() still folds the untouched canonical table in as the
    single holder of all prior evidence.
    """

    if not os.path.exists(canonical_path):
        return False

    payload = torch.load(canonical_path, map_location="cpu")
    agent.q_table.weight.data.copy_(payload["weight"])
    return True


def merge_tables(canonical_path: str, contributions_dir: str) -> "tuple[int, int]":
    """Combine every contribution file in contributions_dir - plus the existing canonical
    table at canonical_path, if any, so merging is incremental across phases/runs rather than
    destructive - into one visit-count-weighted average, written back to canonical_path.
    Deletes the contribution files it consumed once they're folded in.

        merged[s, a] = sum_i(visits_i[s, a] * value_i[s, a]) / sum_i(visits_i[s, a])

    for any cell with at least one visit across all contributors; a cell nobody visited
    stays 0, same as a freshly-initialized table.

    Meant to be run once, by the shell script that launched the parallel training workers,
    right after it has `wait`-ed for all of them to exit - not by the workers themselves - so
    in practice every contribution file it reads is already complete and no locking is needed
    here either.

    Returns (num_contributions_merged, num_states) for logging.
    """

    if os.path.isdir(contributions_dir):
        contribution_paths = [os.path.join(contributions_dir, f)
                               for f in os.listdir(contributions_dir) if f.endswith(".contribution")]
    else:
        contribution_paths = []

    if not contribution_paths and not os.path.exists(canonical_path):
        raise FileNotFoundError(
            f"No contributions found in {contributions_dir} and no existing table at {canonical_path}")

    weighted_sum = None
    visit_sum = None

    def accumulate(weight, visits):
        nonlocal weighted_sum, visit_sum
        if weighted_sum is None:
            weighted_sum = torch.zeros_like(weight)
            visit_sum = torch.zeros_like(visits)
        weighted_sum += weight * visits
        visit_sum += visits

    if os.path.exists(canonical_path):
        prior = torch.load(canonical_path, map_location="cpu")
        accumulate(prior["weight"], prior["visits"])

    for path in contribution_paths:
        payload = torch.load(path, map_location="cpu")
        accumulate(payload["weight"], payload["visits"])

    merged_weight = torch.where(visit_sum > 0, weighted_sum / visit_sum.clamp(min=1),
                                 torch.zeros_like(weighted_sum))

    os.makedirs(os.path.dirname(canonical_path) or ".", exist_ok=True)
    torch.save({"weight": merged_weight, "visits": visit_sum}, canonical_path)

    for path in contribution_paths:
        os.remove(path)

    return len(contribution_paths), merged_weight.shape[0]

