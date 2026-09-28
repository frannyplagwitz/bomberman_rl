"""Training callbacks: setup_training / game_events_occurred / end_of_round.

On a round's final step environment.py calls game_events_occurred() and then
end_of_round() with the same (state, action). To avoid recording that step
twice, game_events_occurred() stores the step it recorded, and end_of_round()
only marks that transition as done, appending a new one only if the step
differs (defensive fallback).

Environment variable overrides (set by driver scripts; config.py defaults
are never changed):
  PPO_AGENT_ROLLOUT_STEPS_OVERRIDE       int   - smaller rollout_steps for smoke tests
  PPO_AGENT_PPO_OVERRIDE                 json  - PPOConfig field overrides
  PPO_AGENT_REWARD_OVERRIDE              json  - RewardConfig field overrides
  PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE int - steps trained in earlier bursts,
                                                  for entropy/LR decay progress
  PPO_AGENT_SAVE_CHECKPOINT              path  - checkpoint save path (required
                                                  when training)
"""
import dataclasses
import json
import os
from collections import deque

import numpy as np
import torch

from . import config as cfg
from .features import features_from_semantic
from .action_mask import mask_from_semantic
from .ppo import PPOAgent
from .rewards import compute_reward
from .state_processing import extract_semantic_state, is_confined_to_small_range


def setup_training(self):
    ppo_config = cfg.PPO_CONFIG
    override_steps = os.environ.get("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE")
    if override_steps:
        ppo_config = dataclasses.replace(ppo_config, rollout_steps=int(override_steps))
        self.logger.info(f"[debug override] rollout_steps={ppo_config.rollout_steps}")

    override_ppo = os.environ.get("PPO_AGENT_PPO_OVERRIDE")
    if override_ppo:
        ppo_config = dataclasses.replace(ppo_config, **json.loads(override_ppo))
        self.logger.info(f"[debug override] ppo_config={ppo_config}")

    reward_config = cfg.REWARD_CONFIG
    override_reward = os.environ.get("PPO_AGENT_REWARD_OVERRIDE")
    if override_reward:
        reward_config = dataclasses.replace(reward_config, **json.loads(override_reward))
        self.logger.info(f"[debug override] reward_config={reward_config}")

    steps_already_trained = int(os.environ.get("PPO_AGENT_STEPS_ALREADY_TRAINED_OVERRIDE", "0"))
    self.ppo = PPOAgent(self.model, ppo_config, steps_already_trained=steps_already_trained)

    # Warm start: restore optimizer state when the loaded checkpoint has one.
    loaded = getattr(self, "_loaded_checkpoint", None)
    if loaded is not None and "optimizer_state_dict" in loaded:
        self.ppo.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.logger.info("Restored optimizer state from checkpoint.")

    # Restore return-normalization stats; older checkpoints without them keep
    # the fresh defaults.
    if loaded is not None and "return_rms_state" in loaded:
        self.ppo.return_rms.load_state_dict(loaded["return_rms_state"])
        self.logger.info("Restored return-normalization running stats from checkpoint.")

    self.reward_config = reward_config
    self.buffer_ready_pending = False
    self.last_recorded_step = None
    self.reward_history = []
    # One dict of update() statistics per PPO update, stored verbatim.
    self.loss_history = []
    self.stall_counter = 0
    self.recent_positions = deque(maxlen=3)
    self.recent_positions_v2 = deque(maxlen=4)


def _resolve_pending_update(self, bootstrap_features):
    if getattr(self, "buffer_ready_pending", False):
        bootstrap_value = self.ppo.value_of(bootstrap_features)
        self.loss_history.append(self.ppo.update(bootstrap_value))
        self.buffer_ready_pending = False


def _update_stall_counter(self, old_semantic, mask):
    """Counts consecutive steps at a valid bombing position (BOMB legal,
    distance 0) without bombing or reaching a tile outside the recent
    position window, so short back-and-forth moves still count as stalling.
    Resets to 0 away from a bombing position and to 1 on a new position.
    """
    is_valid_bombing_pos = old_semantic.has_bombing_target and old_semantic.nearest_bombing_distance == 0
    bomb_legal = bool(mask[cfg.ACTIONS.index("BOMB")])
    is_stalled_here = old_semantic.self_pos in self.recent_positions
    self.recent_positions.append(old_semantic.self_pos)

    if is_valid_bombing_pos and bomb_legal:
        self.stall_counter = self.stall_counter + 1 if is_stalled_here else 1
    else:
        self.stall_counter = 0
    return self.stall_counter


def _update_stall_v2(self, semantic) -> bool:
    """True iff this step receives STALL_PENALTY_V2. Steps without an
    available bomb are ignored entirely; otherwise the position is appended
    to the history and is_confined_to_small_range() decides. Independent of
    stall_counter.
    """
    if not semantic.bomb_available:
        return False
    self.recent_positions_v2.append(semantic.self_pos)
    return is_confined_to_small_range(self.recent_positions_v2)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    old_semantic = extract_semantic_state(old_game_state)
    # Reads the history act() already updated for this same state.
    stall_history = (
        is_confined_to_small_range(self.recent_positions_for_feature)
        if cfg.ENABLE_STALL_HISTORY_FEATURE else None
    )
    old_features = features_from_semantic(old_semantic, stall_history=stall_history)

    # Resolve an update left pending at the previous step, when it was not
    # yet known whether that step was terminal; old_game_state is that step's
    # successor state, so its value is the correct bootstrap.
    _resolve_pending_update(self, old_features)

    mask = mask_from_semantic(old_semantic)
    stall_counter = _update_stall_counter(self, old_semantic, mask)
    stall_v2_triggered = _update_stall_v2(self, old_semantic)

    reward = compute_reward(
        old_game_state, self_action, new_game_state, events, self.reward_config,
        stall_counter, stall_v2_triggered,
    )
    self.reward_history.append(reward)

    if not np.isfinite(reward):
        self.logger.error(f"Non-finite reward computed: {reward}, events={events}")

    action_idx = cfg.ACTIONS.index(self_action)

    self.ppo.store(old_features, mask, action_idx, self.last_log_prob, self.last_value, reward, done=False)
    self.last_recorded_step = old_game_state["step"]

    if self.ppo.ready_to_update():
        self.buffer_ready_pending = True

    self.logger.debug(
        f"step={old_game_state['step']} action={self_action} reward={reward:.4f} events={events}"
    )


def end_of_round(self, last_game_state: dict, last_action: str, events):
    already_recorded = (
        last_game_state is not None and self.last_recorded_step == last_game_state.get("step")
    )

    if already_recorded:
        self.ppo.buffer.mark_last_done()
        if self.buffer_ready_pending:
            self.loss_history.append(self.ppo.update(last_value=0.0))
            self.buffer_ready_pending = False
    else:
        # Defensive fallback for a round ending without a preceding
        # game_events_occurred() call for its last step.
        if last_game_state is not None:
            semantic = extract_semantic_state(last_game_state)
            stall_history = (
                is_confined_to_small_range(self.recent_positions_for_feature)
                if cfg.ENABLE_STALL_HISTORY_FEATURE else None
            )
            features = features_from_semantic(semantic, stall_history=stall_history)
            mask = mask_from_semantic(semantic)
            stall_counter = _update_stall_counter(self, semantic, mask)
            stall_v2_triggered = _update_stall_v2(self, semantic)
            reward = compute_reward(
                last_game_state, last_action, None, events, self.reward_config,
                stall_counter, stall_v2_triggered,
            )
            self.reward_history.append(reward)
            action_idx = cfg.ACTIONS.index(last_action) if last_action in cfg.ACTIONS else cfg.ACTIONS.index("WAIT")
            self.ppo.store(features, mask, action_idx, self.last_log_prob, self.last_value, reward, done=True)
        if self.buffer_ready_pending or len(self.ppo.buffer) > 0:
            self.loss_history.append(self.ppo.update(last_value=0.0))
            self.buffer_ready_pending = False

    _save_checkpoint(self)


def _save_checkpoint(self):
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = os.environ.get("PPO_AGENT_SAVE_CHECKPOINT")
    if not save_path:
        # Refuse to fall back to the canonical checkpoint path, which would
        # silently overwrite it with throwaway weights.
        raise RuntimeError(
            "train=True but PPO_AGENT_SAVE_CHECKPOINT is not set -- refusing to save "
            f"(would otherwise silently overwrite {cfg.STAGE_A_CHECKPOINT}). Pass an "
            "explicit scratch path, e.g. common.run_episodes(..., save_checkpoint=<path>)."
        )
    torch.save(
        {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "return_rms_state": self.ppo.return_rms.state_dict(),
        },
        save_path,
    )
