from collections import Counter
from typing import List
import logging

import os 
import matplotlib.pyplot as plt
import torch 
import numpy as np 
import events as e
import settings as s

from .agent_behavior import BASE_REWARDS, POTENTIAL_WEIGHTS
from .rl_heuristics import PPOAgent


from .callbacks import state_to_features, features_to_tensor, get_scenario, get_num_opponents, get_rounds
from . import spatial_feature_extractor as spatial
from .episode_logger import EpisodeCSVLogger, make_run_id

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']


# Note that hyperparameters will depend on the RL heuristic model we use
# Hyper parameters -- DO modify
#TRANSITION_HISTORY_SIZE = 400  # keep all transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Anchors every output path (logs/, episode CSV, checkpoints, plots) to a
# per-process directory instead of a bare relative path. parallel_run 
# exports NECKAR_RUN_DIR per process, if not performing parallel runs will use "."
BASE_DIR = os.environ.get("NECKAR_RUN_DIR", ".")

# Numbered checkpoint every CKPT_EVERY rounds + final round
CKPT_EVERY = max(1, int(os.environ.get("NECKAR_CKPT_EVERY", "500")))

# How often to refresh the latest model file, used as a safety net if the run was interrupted
SAVE_LATEST_EVERY = max(1, int(os.environ.get("NECKAR_SAVE_LATEST_EVERY", "50")))

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :model_type: model used for training, default is lookup tables (lut)
    :param behavior: behavior that agent should use (peaceful vs. aggressive), default is peaceful 
    """
    # Store model type and behavior 
    self.behavior = getattr(self, "behavior", "peaceful")
    self.model_type = getattr(self, "model_type", "mlp") 
    self.episodes_per_update = getattr(self, "episodes_per_update", 4)
    
    self.scenario = get_scenario()
    self.num_opponents = get_num_opponents()
    self.num_rounds = get_rounds() 
    
    
    self.episode_lengths = [] 
    self.current_episode_steps = 0

    # Plot / Debugging metric counters
    self.episode_bombs_dropped = 0
    self.episode_bombs_useful = 0
    self.episode_bombs_wasteful = 0
    self.episode_crates_destroyed = 0
    self.episode_coins_collected = 0
    self.episode_bomb_masked_steps = 0 # steps where BOMB is masked
    self.episode_bomb_legal_not_taken = 0 # steps where BOMB was legal but agent took something else
    self.episode_trapped_enemy = 0 # steps where enemy was trapped
    self.episode_trapped_self = 0 # steps where agent was trapped
    
    self.action_counts = Counter()  # rolling action distribution, prints every 10 episodes
    
    self.logger.info("Initialize PPO Agent with gameplay buffer")
    
    # Logger for high-volume per-step debug
    self.step_logger = logging.getLogger(f"{self.logger.name}.steps")
    self.step_logger.setLevel(logging.DEBUG)
    self.step_logger.propagate = False
    
    if not self.step_logger.handlers: # guard against duplicate handlers
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        step_handler = logging.FileHandler(os.path.join(BASE_DIR, "logs", f"{self.logger.name}-steps.log"), mode="w")
        step_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.step_logger.addHandler(step_handler)
    
    
    # Entropy values        
    entropy_start = getattr(self, "entropy_start", 0.05)
    entropy_end = getattr(self, "entropy_end", 0.01)
    entropy_decay_updates = getattr(self, "entropy_decay_updates", None) or (self.num_rounds // self.episodes_per_update)
  
    # LR values
    lr_start = getattr(self, "lr_start", 1e-4)
    lr_end   = getattr(self, "lr_end", 1e-5)   # 10x reduction by end of training
    lr_decay_updates = getattr(self, "lr_decay_updates", None) or (self.num_rounds // self.episodes_per_update)


    # Initialize CSV run and episode log path
    # The name will depend on whether or not we are training
 
    self.run_id = getattr(self, "run_id", None) or make_run_id()
 
    episode_log_path = os.path.join(
        BASE_DIR, "logs",
        f"episodes-{self.model_type}-{self.behavior}-{self.scenario}"
        f"-{self.num_rounds}-rounds-{self.num_opponents}-opponents-{self.run_id}.csv")
    self.episode_logger = EpisodeCSVLogger(episode_log_path)
 

    self.ppo_agent = PPOAgent(
        model=self.model,
        lr=lr_start, lr_end=lr_end, lr_decay_updates=lr_decay_updates,
        gamma=0.99, clip_eps=0.2,
        model_type=self.model_type, behavior=self.behavior,
        potential_weights=POTENTIAL_WEIGHTS.get(self.behavior, {}),
        base_rewards=BASE_REWARDS.get(self.behavior, {}),
        entropy_start=entropy_start, entropy_end=entropy_end,
        entropy_decay_updates=entropy_decay_updates,
        logger=self.logger, step_logger=self.step_logger)

    
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')  
   
def detect_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list) -> List[str]:
    """Called once per step to determine if a custom event occurred

    Args:
        old_game_state (dict): The state that was passed ot the last call of `act`
        self_action (str): The state that was passed to the last call of `act`
        new_game_state (dict): The state the agent is in now 
        events (List[str]): The events that occurred when going from `old_game_state` to `new_game_state`

    Returns:
        List[str]: list of custom events that were detected
    """
    
    custom_events = []
    
    if old_game_state is None or new_game_state is None: 
        return custom_events
    
    # If agent did anything, add step penalty event 
    # Done to make sure agent will finish game before max steps if possible 
    # Reasoning: Saves computation time, reduces risk of agent doing something dumb
    if self_action: 
        custom_events.append("STEP_PENALTY")
                   
    # Grab necessary data from old game state
    old_field = old_game_state['field']
    old_pos = old_game_state['self'][3]    
    old_bombs = old_game_state.get('bombs', None)
    old_opponents = [pos for _, _, _, pos in old_game_state['others']] if old_game_state['others'] else []
    old_explosion_map = old_game_state.get('explosion_map', None)
    old_entities = old_opponents + [old_pos]
    
    
    # Grab necessary data from new game state
    new_field = new_game_state['field']
    new_pos = new_game_state['self'][3]
    new_bombs = new_game_state.get('bombs', None)
    new_opponents = [pos for _,_,_, pos in new_game_state['others']] if new_game_state['others'] else []

    new_explosion_map = new_game_state.get('explosion_map', None)
    new_entities = new_opponents + [new_pos]
    
    # Add bomb visibility for debug purposes - log every bomb's position, timer and distance to agent
    # for every step
    bomb_info = [
        {"pos": bpos, "timer": btimer, "dist_to_agent": 
            abs(bpos[0] - new_pos[0]) + abs(bpos[1] - new_pos[1])}
        
        for bpos, btimer in (new_bombs or [])
    ]
    self.step_logger.debug(f"[BOMB VISIBILITY DEBUG] Step {new_game_state['step']} |"
                           f" Agent pos={new_pos} |"
                           f" Bombs={bomb_info}")
    
    
    # Get distance to safety will return 0 whenever the position isn't in danger
    old_dist_to_safety = spatial.get_distance_nearest_safety(
        old_field, old_pos, 
        old_bombs,old_opponents, 
        old_explosion_map)
    
    # If we had danger in previous state, look danger in new state 
    if old_dist_to_safety > 0: 
        new_dist_to_safety = spatial.get_distance_nearest_safety(
            new_field, new_pos, 
            new_bombs, new_opponents, 
            new_explosion_map)
        
        # If the new safety distance is smaller than the old safety distance, moving towards safety
        if new_dist_to_safety < old_dist_to_safety:
            custom_events.append("MOVED_TO_SAFETY")
        
        elif new_dist_to_safety > old_dist_to_safety: 
            custom_events.append("MOVED_FROM_SAFETY")
            
    
    # Check if a bomb was dropped
    if e.BOMB_DROPPED in events: 
        if spatial.is_bomb_useful(old_field, old_pos, old_opponents):
            custom_events.append("BOMB_USEFUL")
        
        else: 
            custom_events.append("BOMB_WASTEFUL") 
    
    # Grab opponents by name 
    old_opponents_by_name = ({name: pos for name, _, _, pos in old_game_state['others']})    
    new_opponents_by_name = ({name: pos for name, _, _, pos in new_game_state['others']})
    
    # Look through POV of each opponent in new state
    for name, new_opp_pos in new_opponents_by_name.items(): 
        
        # Get list of opponent's opponents for spatial
        # extractor calls
        new_op_opponents = {op for op in new_entities if op != new_opp_pos}
        
        # Check if the opponent is currently trapped 
        opponent_is_trapped = spatial.is_entity_trapped(
            field= new_field, entity_pos=new_opp_pos,
            bombs=new_bombs, opponents=new_op_opponents, 
            explosion_map=new_explosion_map)
          
        # Find the old position of the opponent,
        # if doesn't exist, not trapped 
        old_opp_pos = old_opponents_by_name.get(name)
        if old_opp_pos is None: 
            opponent_was_trapped = False
            
        # Check if opponent was trapped in previous state 
        else:
            old_op_opponents = {op for op in old_entities if op != old_opp_pos}
            
            opponent_was_trapped = spatial.is_entity_trapped(
                field=old_field, entity_pos=old_opp_pos, 
                bombs=old_bombs, opponents=old_op_opponents, 
                explosion_map=old_explosion_map)
    
        # If opponent wasn't trapped in old state but is now, opponent trapped event
        if not opponent_was_trapped and opponent_is_trapped:
            custom_events.append("TRAPPED_ENEMY")
            self.episode_trapped_enemy += 1    
        
        
        
    # Check if our agent is trapped in new state
    is_trapped = spatial.is_entity_trapped(
        field = new_field, entity_pos=new_pos, 
        bombs = new_bombs, opponents = set(new_opponents),
        explosion_map = new_explosion_map)
    
    
    # Check if our agent was trapped in the old state
    was_trapped = spatial.is_entity_trapped(
        field=old_field, entity_pos=old_pos, 
        bombs = old_bombs, opponents = set(old_opponents), 
        explosion_map = old_explosion_map)
    
    # If wasn't trapped and is trapped now, causes trapped self event 
    if not was_trapped and is_trapped: 
        custom_events.append("TRAPPED_SELF")
        self.episode_trapped_self += 1

    return custom_events    
        
    
def game_events_occurred(self, old_game_state: dict, self_action: str, 
                         new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    :param model_type: The type of model used for training, default is lookup tables (lut)
    :param behavior: The behavior that the agent uses, default is peaceful 
    """

    self.step_logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} '
                           f'in step {new_game_state["step"]}')

    if new_game_state is not None: 
        self.current_episode_steps = new_game_state['step']


    # Add own events to hand out rewards
    custom_events = detect_custom_events(
        self, old_game_state, self_action, 
        new_game_state, events)
  
    
    # Debug reader to check events 
    self.step_logger.debug(
        f"[EVENT DEBUG] Step: {old_game_state['step']}  |"
        f" Action: {self_action} |"
        f" Raw Events={events} |" 
        f" Custom Events={custom_events}")
    
    events.extend(custom_events)
        
    # Grab agent position 
    agent_pos = old_game_state['self'][3]    
    
    effective_next_state = new_game_state.copy()
    
    # Bomb/Crate debugging
    self.action_counts[getattr(self, 'last_action', 5)] += 1
    
    bomb_mask = getattr(self, 'last_mask', None)
    bomb_legal = bool(bomb_mask[4].item()) if bomb_mask is not None else None 
    
    # Update running counter metrics
    if self_action == 'BOMB':
        self.episode_bombs_dropped += 1
        
        if 'BOMB_USEFUL' in custom_events: 
            self.episode_bombs_useful += 1
            
        elif 'BOMB_WASTEFUL' in custom_events: 
            self.episode_bombs_wasteful += 1
            
    elif bomb_legal is True:
        # BOMB legal, but agent took something else
        self.episode_bomb_legal_not_taken += 1

    elif bomb_legal is False:
        # BOMB wasn't even offered as a choice this step 
        # (cooldown, or masked out as a self-trap)
        self.episode_bomb_masked_steps += 1


    if e.CRATE_DESTROYED in events: 
        self.episode_crates_destroyed += events.count(e.CRATE_DESTROYED)
        
    if e.COIN_COLLECTED in events:                                   
        self.episode_coins_collected += events.count(e.COIN_COLLECTED)
        
        
    self.step_logger.debug(
        f"[BOMB CHOICE DEBUG] Step {new_game_state['step']:3d} | Action: {self_action:<6s} |" 
        f" BOMB legal this step: {bomb_legal} | "
        f"Episode totals so far -> Dropped: {self.episode_bombs_dropped} "
        f"(Useful: {self.episode_bombs_useful}, Wasteful: {self.episode_bombs_wasteful}) | "
        f"Crates: {self.episode_crates_destroyed}")
        
    
    if self_action == 'BOMB':
        new_bombs = [b[0] for b in effective_next_state.get('bombs', [])]
        is_registered = agent_pos in new_bombs 
        
        self.step_logger.debug(f"[BOMB CHECK] Bombs in new_states: {new_bombs} "
                               f"| Registered: {is_registered}")
        
        
        if agent_pos not in new_bombs: 
            bombs_list = list(effective_next_state.get('bombs', []))
            bombs_list.append((agent_pos, s.BOMB_TIMER))
            effective_next_state['bombs'] = bombs_list 
    
    
    # Store scenario name and num opponents for logging purposes
    self.scenario = get_scenario()
    self.num_opponents = get_num_opponents()
    self.num_rounds = get_rounds()
    
    # Calculate reward
    shaped_reward = self.ppo_agent.compute_reward(
        game_state=old_game_state,
        next_game_state=effective_next_state,
        events=events,
        terminal=False)
        
    step = effective_next_state['step']
    action = self_action if self_action else "NONE"
    self.step_logger.debug(f"[REWARD DEBUG] Step {step:3d} | Action: {action:<10s} "
                           f"| Events: {custom_events} | Net Reward: {shaped_reward:+.4f}")


    # grab old tensor 
    state_tensor = getattr(self, 'last_state', None)
    
    # Convert old state tensor (if available) to features and then back to a tensor
    if state_tensor is None: 
        state_features = state_to_features(old_game_state, self.model_type)
        state_tensor   = features_to_tensor(state_features)
        
    # Convert next state to features then to tensors
    next_state_features = state_to_features(effective_next_state, self.model_type)
    next_state_tensor   = features_to_tensor(next_state_features)
   
    stored_action_idx = getattr(self, 'last_action', 5)
    self.step_logger.debug(
        f"[BUFFER DEBUG] storing action_idx={stored_action_idx} "
        f"(ACTIONS[{stored_action_idx}]='{ACTIONS[stored_action_idx]}') "
        f"vs actual action='{self_action}'"
    )
    
    if not hasattr(self, 'last_action'):
        self.logger.warning("[BUFFER DEBUG] self.last_action missing, using default")
    
    if not hasattr(self, 'last_log_prob'):
        self.logger.warning("[BUFFER DEBUG] self.last_log_prob missing, using default")
        
    if not hasattr(self, 'last_mask'):
        self.logger.warning("[BUFFER DEBUG] self.last_mask, using default")
        

    # Store scalar into buffer for PPO update
    self.ppo_agent.buffer.store(
        state = state_tensor,
        action = getattr(self, 'last_action', 5), # Defaults to WAIT if not set
        log_prob = getattr(self, 'last_log_prob', torch.tensor(0.0)),
        reward = shaped_reward,
        next_state = next_state_tensor, # PPO relies on state-action-reward mask 
        terminal = False,
        mask = getattr(self, 'last_mask', torch.ones(6, dtype=torch.bool)))    

    
    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    
    # Reset episode
    self.ppo_agent.reset_episode()
    
    # Keep track of terminal actions/ events
    self.action_counts[getattr(self, 'last_action', 5)] += 1
    
    bomb_mask = getattr(self, 'last_mask', None)
    bomb_legal = bool(bomb_mask[4].item()) if bomb_mask is not None else None 
        
    if last_action == 'BOMB':
        self.episode_bombs_dropped += 1
        
        old_pos = last_game_state['self'][3]
        old_field = last_game_state['field']
        old_opponents = [pos for _, _, _, pos in last_game_state['others']] if last_game_state['others'] else []
        
        if spatial.is_bomb_useful(old_field, old_pos, old_opponents):
            self.episode_bombs_useful += 1
            
        else: 
            self.episode_bombs_wasteful += 1
            
    elif bomb_legal is True: 
        # bomb legal, but agent took another action 
        self.episode_bomb_legal_not_taken += 1
        
    elif bomb_legal is False: 
        # Bomb was masked 
        self.episode_bomb_masked_steps += 1

            
    if e.CRATE_DESTROYED in events: 
        self.episode_crates_destroyed += events.count(e.CRATE_DESTROYED)
  
  
    final_steps = last_game_state['step']
    self.episode_lengths.append(final_steps)
    
    # Per-episode crate/bomb summary 
    useful_rate = (100.0 * self.episode_bombs_useful / self.episode_bombs_dropped
                   if self.episode_bombs_dropped else 0.0)
    
    self.logger.info(
        f"[EPISODE SUMMARY] Round {len(self.episode_lengths):4d} | Steps: {final_steps:3d} | "
        f"Bombs Dropped: {self.episode_bombs_dropped} "
        f"(Useful: {self.episode_bombs_useful}, Wasteful: {self.episode_bombs_wasteful}, "
        f"Useful Rate: {useful_rate:.0f}%) | Crates Destroyed: {self.episode_crates_destroyed} | "
        f"BOMB masked out: {self.episode_bomb_masked_steps} steps | "
        f"BOMB legal but not taken: {self.episode_bomb_legal_not_taken} steps")
    
    # Capture episode counters before resetting them 
    bombs_dropped = self.episode_bombs_dropped
    bombs_useful = self.episode_bombs_useful
    bombs_wasteful = self.episode_bombs_wasteful
    crates_destroyed = self.episode_crates_destroyed
    coins_collected = self.episode_coins_collected
    trapped_enemy = self.episode_trapped_enemy
    trapped_self = self.episode_trapped_self
    bomb_masked_steps = self.episode_bomb_masked_steps
    bomb_legal_not_taken = self.episode_bomb_legal_not_taken
    
    
    # Reset per-episode bomb/crate counters for next round 
    self.episode_bombs_dropped = 0
    self.episode_bombs_useful  = 0
    self.episode_bombs_wasteful = 0
    self.episode_crates_destroyed = 0
    self.episode_coins_collected = 0
    self.episode_bomb_masked_steps = 0
    self.episode_bomb_legal_not_taken = 0
    self.episode_trapped_enemy = 0
    self.episode_trapped_self = 0
    
    
    if len(self.episode_lengths) % 10 == 0:
        avg_len = np.mean(self.episode_lengths[-10:])

        self.logger.info(f"[Episode {len(self.episode_lengths)}] "
                         f"Avg Length (last 10): {avg_len:.1f} steps")

        # Rolling action distribution - nested here so it covers the last 10 episodes, matching the
        # label below, and only clears once every 10 episodes instead of every single one
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            dist_str = " | ".join(
                f"{ACTIONS[i]}: {100.0 * self.action_counts[i] / total_actions:.1f}%"
                    for i in range(len(ACTIONS)))

            self.logger.info(f"[ACTION DIST DEBUG] Last {total_actions} actions (last 10 episodes) -> {dist_str}")
            self.action_counts.clear()

    # Extract parameters from args 
    num_opponents = get_num_opponents()
    num_rounds = get_rounds()
    
    # Grab file path to save checkpoint
    filepath = os.path.join(
    BASE_DIR,
    f"actor-critic-{self.model_type}-{self.behavior}-{self.scenario}"
    f"-{num_rounds}-rounds-{num_opponents}-opponents.pt")
    
    # Calculate shaped reward for terminal state
    shaped_reward = self.ppo_agent.compute_reward(
        game_state=last_game_state,
        next_game_state=None,
        events=events,
        terminal=True)
    
    # Create state tensors 
    state_tensor = getattr(self, 'last_state', None)
    if state_tensor is None: 
        state_features = state_to_features(last_game_state, self.model_type)
        state_tensor   = features_to_tensor(state_features)
    

    # Push terminal state into PPO gameplay buffer
    self.ppo_agent.buffer.store(
        state=state_tensor,
        action=getattr(self, 'last_action', 5), # Default to WAIT action if not set 
        log_prob=getattr(self, 'last_log_prob', torch.tensor(0.0)),
        reward=shaped_reward,
        next_state=None,
        terminal=True,
        mask=getattr(self, 'last_mask', torch.ones(6, dtype=torch.bool)))

    # Add reward buffer to debug log 
    self.step_logger.debug(f"[EOR DEBUG] Reward Buffer {self.ppo_agent.buffer.rewards}")
    
    # Check the amount of episodes we should be storing before update 
    if len(self.episode_lengths) % self.episodes_per_update == 0: 
        self.ppo_agent.update() 

        self.logger.info(
            f"[EOR DEBUG] Updated ppo agent (rollout covered {self.episodes_per_update} episodes)")
    
    
    # Obtain value prediction and error 
    v_pred = None
    value_error = None
        
    # Obtain eval state for each 
    with torch.no_grad():
        # Grab state to evaluate
        eval_state = state_tensor.unsqueeze(0) if state_tensor.dim() == 1 else state_tensor
            
        if hasattr(self.model, 'get_value'):
            v_pred = self.model.get_value(eval_state).item()
            value_error = abs(v_pred - shaped_reward)
            
            self.logger.info(
                    f"[TERMINAL VALUE CHECK] Episode End | Events: {events}\n"
                    f"                        Shaped Reward (Target): {shaped_reward:8.4f}\n"
                    f"                        Critic V(s) Predicted : {v_pred:8.4f}\n"
                    f"                        Value Error           : {abs(v_pred - shaped_reward):8.4f}")
        
        
        # Run per-episode metrics 
        self.episode_logger.log_episode({
            "run_id": self.run_id, "model_type": self.model_type,
            "behavior": self.behavior, "scenario": self.scenario,
            "num_rounds": num_rounds, "opponents": num_opponents,                             
            "episodes": len(self.episode_lengths),                  
            "steps": final_steps, "bombs_dropped": bombs_dropped,
            "bombs_useful": bombs_useful,"bombs_wasteful": bombs_wasteful,
            "useful_rate": useful_rate, "crates_destroyed": crates_destroyed,
            "coins_collected": coins_collected, "killed_self": int(e.KILLED_SELF in events),
            "got_killed": int(e.GOT_KILLED in events), "killed_opponent": int(e.KILLED_OPPONENT in events),
            "opponent_killed": int(e.OPPONENT_ELIMINATED in events), # was opponent_eliminated
            "survived_round": int(e.SURVIVED_ROUND in events), "terminal_action": last_action,
            "shaped_reward": shaped_reward, "critic_value_pred": v_pred, "value_error": value_error,
            "total_loss": self.ppo_agent.loss_history[-1] if self.ppo_agent.loss_history else None, 
            "critic_loss": self.ppo_agent.critic_loss_history[-1] if self.ppo_agent.critic_loss_history else None, 
            "mean_reward": self.ppo_agent.mean_reward_history[-1] if self.ppo_agent.mean_reward_history else None, 
            "entropy": self.ppo_agent.entropy_history[-1] if self.ppo_agent.entropy_history else None,
            "bomb_masked": bomb_masked_steps,
            "bomb_legal_not_taken": bomb_legal_not_taken,
            "trapped_self": trapped_self, 
            "trapped_enemy": trapped_enemy
            
        })
            
    
        # Create file names for the metrics 
        #common_filepath = (f"{self.run_id}-{self.model_type}_{self.behavior}_{self.scenario}_{num_rounds}_rounds_{num_opponents}_opponents.png")

        #loss_filepath    = "total_loss_" + common_filepath
        #critic_filepath  = "critic_loss_" + common_filepath
        #mr_filepath      = "mean_reward_" + common_filepath 
        #entropy_filepath = "entropy_" + common_filepath
        #episode_filepath = "episode_steps_" + common_filepath
        
    
        # Create title names for the metrics 
        
        #common_title = (f" - Model: {self.model_type}, Behavior: {self.behavior}, Scenario: {self.scenario}")
        #model_name = "PPO"
        
        
        #loss_title = model_name + " Total MSE Loss" + common_title
        #critic_title = model_name + " Critic MSE Loss" + common_title 
        #mr_title = model_name + " Mean Reward" + common_title
        #entropy_title = model_name + " Entropy" + common_title
        #episode_title = model_name + " Episode Steps" + common_title 

        
        #plot_metric(self.ppo_agent.loss_history, loss_filepath, "Training Loss", loss_title)
        
        #if hasattr(self.ppo_agent, "critic_loss_history"):
        #    plot_metric(self.ppo_agent.critic_loss_history, critic_filepath, 
         #               "Critic Loss", critic_title)
        
        #plot_metric(self.ppo_agent.mean_reward_history, mr_filepath, "Mean Reward", mr_title)
        #plot_metric(self.ppo_agent.entropy_history, entropy_filepath, "Entropy", entropy_title)
        #plot_metric(self.episode_lengths, episode_filepath, "Episode Steps", episode_title)

    episode = len(self.episode_lengths)
    is_final_round = num_rounds > 0 and episode == num_rounds

    # Keep a rolling "latest" copy (no -ep suffix, so run_eval ignores it)
    if episode % SAVE_LATEST_EVERY == 0 or is_final_round:
        save_model(self.model, filepath)
        self.logger.info(f"Saved trained model to {filepath}")

    # Numbered checkpoints for evaluation: every CKPT_EVERY rounds, plus the final round
    if episode % CKPT_EVERY == 0 or is_final_round:
        ckpt_path = filepath[:-len(".pt")] + f"-ep{episode}.pt"
        save_model(self.model, ckpt_path)
        self.logger.info(f"Saved checkpoint to {ckpt_path}")


    # Store a checkpoint every 500 rounds
    if len(self.episode_lengths) % 500 == 0:
       ckpt_path = filepath.replace(".pt", f"-ep{len(self.episode_lengths)}.pt")
       torch.save(self.model.state_dict(), ckpt_path)
       self.logger.info(f"Saved checkpoint to {ckpt_path}")
       #print(f"Saved checkpoint to {ckpt_path}")
        
        
    self.logger.info(f"Saved trained model to {filepath}")

def save_model(model, path: str):
    """Write to a temp file, then swap it in, so a crash mid-save can't leave a corrupt .pt."""
    tmp_path = path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, path)   

def plot_metric(history: list, filename: str, ylabel: str, title: str, color: str = "blue", window: int = 10):
    
    """Plot a metric over training
    
    Params: 
    history (list): history of changes for a metric 
    filename (str): name of file to store plot in 
    ylabel (str): y label
    title (str): Title for the plot 
    color (str): plot color, defaults to blue
    window (int): Window to start tracking standardizations for 
    """
    
    # Create plot output directory if it doesn't exist
    output_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, filename)
            
    plt.figure(figsize=(8, 5))
    plt.plot(history, label=ylabel, color=color)
    
    
    # Calculate moving average 
    if len(history) >= window:     
        moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
        x_range = range(window - 1, len(history))
        
        plt.plot(x_range, moving_avg, label=f"{window}-step Moving average", color='orange')
        
    plt.xlabel('PPO Updates')
    plt.ylabel(ylabel)
    plt.title(title)  
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()