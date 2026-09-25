"""Per-Episode metric logger

Will write CSV row per completed episode tagged with run's configuration so that training runs
can be reloaded and compared later through plot_comparison.py
"""

import csv
import os
import time 
from typing import Any, Dict

EPISODE_LOG_FIELDS = [
    "run_id", "model_type", "behavior", "scenario", "num_rounds", "opponents",
    "episodes", "steps", "bombs_dropped", "bombs_useful", "bombs_wasteful",
    "useful_rate", "crates_destroyed", "coins_collected", "killed_self",
    "got_killed", "killed_opponent", "opponent_killed", "survived_round",
    "terminal_action", "shaped_reward", "critic_value_pred", "value_error",
    "total_loss", "critic_loss", "mean_reward", "entropy", "bomb_masked",
    "bomb_legal_not_taken", "trapped_enemy", "trapped_self"
]

class EpisodeCSVLogger:
    """Adds one row per episode to CSV file"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        

        # Write header if the file is new/empty
        needs_header = not os.path.isfile(filepath) or os.path.getsize(filepath) == 0
        if needs_header:
            with open(filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=EPISODE_LOG_FIELDS).writeheader()
                
        
    def log_episode(self, row: Dict[str, Any]):
        """Append one episode's metrics. Any field not in `row` will be blank."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_LOG_FIELDS)
            writer.writerow({k: row.get(k, "") for k in EPISODE_LOG_FIELDS})

def make_run_id() -> str:
    """ Make a sortable id for a training run based on process start time"""
    return time.strftime("%Y%m%d-%H%M%S")
