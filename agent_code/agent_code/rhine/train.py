"""setup_training / game_events_occurred / end_of_round.

Framework quirk handled here: on the final step of a round, environment.py
calls game_events_occurred(old_state, action, new_state, events) and then,
within the same do_step(), also calls end_of_round(last_game_state,
last_action, events + [SURVIVED_ROUND]) with the *same* (state, action).
Appending a transition in both would double-count the last step of every
episode. Fix: game_events_occurred always appends and records which `step`
it just recorded; end_of_round only appends a new transition if
last_game_state's step doesn't match what was already recorded (defensive
path -- not reachable in Task 1 since the agent cannot die), and otherwise
just flips the already-stored transition's `done` flag to True.

Environment variable overrides (used by driver scripts, never change the
defaults in config.py):
  PPO_AGENT_ROLLOUT_STEPS_OVERRIDE  int   - smaller rollout_steps for smoke tests
  PPO_AGENT_REWARD_OVERRIDE         json  - reward-config field overrides (Stage A ablations)
  PPO_AGENT_SAVE_CHECKPOINT         path  - where end_of_round saves the checkpoint
                                             (defaults to models/task1_stage_a.pt)
"""
import dataclasses
import json
import os

import numpy as np
import torch

from . import config as cfg
from .features import features_from_semantic
from .action_mask import mask_from_semantic
from .ppo import PPOAgent
from .rewards import compute_reward
from .state_processing import extract_semantic_state


def setup_training(self):
    ppo_config = cfg.PPO_CONFIG
    override_steps = os.environ.get("PPO_AGENT_ROLLOUT_STEPS_OVERRIDE")
    if override_steps:
        ppo_config = dataclasses.replace(ppo_config, rollout_steps=int(override_steps))
        self.logger.info(f"[debug override] rollout_steps={ppo_config.rollout_steps}")

    reward_config = cfg.REWARD_CONFIG
    override_reward = os.environ.get("PPO_AGENT_REWARD_OVERRIDE")
    if override_reward:
        reward_config = dataclasses.replace(reward_config, **json.loads(override_reward))
        self.logger.info(f"[debug override] reward_config={reward_config}")

    self.ppo = PPOAgent(self.model, ppo_config)

    # Warm start: also restore optimizer momentum if setup() loaded a checkpoint
    # that has one (Stage C, and Stage A's own periodic-eval-interleaved bursts
    # reloading their own progress). Fresh Stage A runs have no checkpoint loaded,
    # so the optimizer just starts fresh, as expected.
    loaded = getattr(self, "_loaded_checkpoint", None)
    if loaded is not None and "optimizer_state_dict" in loaded:
        self.ppo.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.logger.info("Restored optimizer state from checkpoint.")

    self.reward_config = reward_config
    self.buffer_ready_pending = False
    self.last_recorded_step = None
    self.reward_history = []


def _resolve_pending_update(self, bootstrap_features):
    if getattr(self, "buffer_ready_pending", False):
        bootstrap_value = self.ppo.value_of(bootstrap_features)
        self.ppo.update(bootstrap_value)
        self.buffer_ready_pending = False


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    old_semantic = extract_semantic_state(old_game_state)
    old_features = features_from_semantic(old_semantic)

    # Resolve an update that became due at the end of the *previous* step: at that
    # point we didn't yet know whether the previous step was also the episode's
    # terminal step (end_of_round fires after game_events_occurred within the same
    # do_step). old_game_state here is exactly that previous step's new_game_state,
    # so its value is the correct bootstrap for a non-terminal truncation.
    _resolve_pending_update(self, old_features)

    reward = compute_reward(old_game_state, self_action, new_game_state, events, self.reward_config)
    self.reward_history.append(reward)

    if not np.isfinite(reward):
        self.logger.error(f"Non-finite reward computed: {reward}, events={events}")

    mask = mask_from_semantic(old_semantic)
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
            self.ppo.update(last_value=0.0)
            self.buffer_ready_pending = False
    else:
        # Defensive fallback: not reachable in Task 1 (agent cannot die, so
        # game_events_occurred always fires for the terminal step first), kept
        # for robustness in case a round ever ends without that call happening.
        if last_game_state is not None:
            semantic = extract_semantic_state(last_game_state)
            features = features_from_semantic(semantic)
            mask = mask_from_semantic(semantic)
            reward = compute_reward(last_game_state, last_action, None, events, self.reward_config)
            self.reward_history.append(reward)
            action_idx = cfg.ACTIONS.index(last_action) if last_action in cfg.ACTIONS else cfg.ACTIONS.index("WAIT")
            self.ppo.store(features, mask, action_idx, self.last_log_prob, self.last_value, reward, done=True)
        if self.buffer_ready_pending or len(self.ppo.buffer) > 0:
            self.ppo.update(last_value=0.0)
            self.buffer_ready_pending = False

    _save_checkpoint(self)


def _save_checkpoint(self):
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = os.environ.get("PPO_AGENT_SAVE_CHECKPOINT", str(cfg.STAGE_A_CHECKPOINT))
    torch.save(
        {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
        },
        save_path,
    )
