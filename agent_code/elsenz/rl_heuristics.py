
""" All functions and information regarding RL heurisitics for Tabular Q-Learning 

"""
import os
import torch 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

from torch.distributions import Categorical

from .callbacks import state_to_features
from . import spatial_feature_extractor as spatial

ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

class RunningMeanStd: 
    """Running  mean/ variance - Welford online algorithm"""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

        
    def update(self, x: torch.Tensor): 
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        batch_count = x.numel()
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count 
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        
        self.mean, self.var, self.count = new_mean, M2 / tot_count, tot_count
        
    @property
    def std(self):
        return (self.var ** 0.5) + 1e-8


class GameplayBuffer: 
    """ Stores step transitions during gameplay
        Elements stored: 
        states: the state for each played step 
        actions: actions taken for each step 
        log_probs: policy values of taking actions
        rewards: rewards given for taking actions
        next_states: states transitioned to after taking actions
        terminals: if states are terminal 
        masks: Masks to ignore invalid moves
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.masks = []

    def store(self, state, action, log_prob, reward, next_state, terminal, mask):
        
        # Need to make sure that items are detached to prevent memory leaks
        self.states.append(state.detach().cpu() if torch.is_tensor(state) else state)
        self.next_states.append(next_state.detach().cpu() if torch.is_tensor(next_state) else next_state)
            
        # Convert tensor/arrays
        self.actions.append(int(action.item() if torch.is_tensor(action) else action))
        self.log_probs.append(float(log_prob.item() if torch.is_tensor(log_prob) else log_prob))
        self.rewards.append(float(reward.item() if torch.is_tensor(reward) else reward))
        self.terminals.append(bool(terminal.item() if torch.is_tensor(terminal) else terminal))
        
        # Store action mask as detached CPU tensor 
        if torch.is_tensor(mask):
            self.masks.append(mask.detach().cpu().to(dtype=torch.bool))
            
        elif isinstance(mask, np.ndarray):
            self.masks.append(mask.astype(bool))
        
        else: 
            self.masks.append(mask)
        

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()
        self.masks.clear()



def compute_potential(game_state: dict, potential_weights: dict, visited_tiles: set, step_logger: logging.Logger) -> float: 
    
    """ Compute the total state potential for potential based reward shaping (PBRS)
    
    Args: 
        game_state (dict): State of the game environment 
        potential_weights (dict): dictionary of potential weights given a particular model     
        
        returns total state potential 
    """


    # If there's no game state, just return potential of 0 
    if game_state is None: 
        return 0.0
    
    # Grab scene data
    agent_pos = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    crates = [(x,y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x,y] == 1]

    # Find total walkable tiles
    total_walkable_tiles = np.sum(field == 0)
    
    
    # Grab properties from game state
    bombs = game_state['bombs']
    bomb_positions = spatial.get_bomb_positions(bombs)
    opponents = [other[3] for other in game_state['others']]
    entities = opponents + [agent_pos]
    explosion_map = game_state.get('explosion_map', None)
    
        
    # Overall potential = w_survive * Phi_danger + w_explore * Phi_explore + w_coin * Phi_coin(s) + w_crate * Phi_crate(s) + w_kill * Phi_kill(s) + w_trap * Phi_trap(s)
    # We split these up into distinct portions, for example, if there's no bombs in the game, no danger = Phi_danger = 0.0
    
    # Pre-initialize all potentials to 0.0
    phi_danger  = 0.0
    phi_explore = 0.0
    phi_coin    = 0.0
    phi_crate   = 0.0
    phi_kill    = 0.0
    phi_trap    = 0.0

    # Get potential weights 
    # Note that w_danger is set to 1.0 because if danger is 0, should promote the action 
    w_danger  = potential_weights.get('SURVIVE', 1.0) 
    w_explore = potential_weights.get('EXPLORE', 0.0)
    w_coin    = potential_weights.get('COIN', 0.0)
    w_crate   = potential_weights.get('CRATE', 0.0)
    w_kill    = potential_weights.get('KILL', 0.0)
    w_trap    = potential_weights.get('TRAP', 0.0)
    
    # Danger potential:  w_survive * (1 - (1 / (distance_to_danger + 1))
    safety_dist = spatial.get_distance_nearest_safety(
        field=field, agent_pos=agent_pos, bombs=bombs, 
        opponents=opponents, explosion_map=explosion_map)
    
    if safety_dist > 0 and safety_dist != float('inf'): 
        timer_map  = spatial.build_bomb_timer_map(field, bombs)
        tile_timer = float(timer_map[agent_pos[0], agent_pos[1]])
        
        # Active explosion or unresolvable timer = urgent
        if not np.isfinite(tile_timer):
            tile_timer = 1.0
        
        
        # Urgency ~0 if there's a lot of time left before explosion 
        # Urgency ~1 if escape is as long/longer than time left before explosion
        urgency = safety_dist / max(tile_timer, 1.0)
        phi_danger = -w_danger * min(urgency, 1.0)
        
        step_logger.debug(
            f"[DANGER DEBUG] pos={agent_pos} |"
            f" safety_dist={safety_dist} |" 
            f" tile_timer={tile_timer:.1f} |"
            f" urgency={urgency:.3f} |" 
            f" phi_danger={phi_danger:.4f}")
        
    elif safety_dist == float('inf'):
        phi_danger = -w_danger
        step_logger.debug(
            f"[DANGER DEBUG] pos={agent_pos} |"
            f" NO ESCAPE ROUTE (safety_dist=inf) |"
            f" phi_danger={phi_danger:.4f}") 
        
    # Calculate exploration potential based on visited tile set size 
    current_visited_count = len(visited_tiles | {agent_pos})
    
    if total_walkable_tiles > 0: 
        phi_explore = w_explore * (current_visited_count / float(total_walkable_tiles))
    

    def safe_get_dist(targets):
        
        if not targets: 
            return None 
        
        res = spatial.get_nearest_target(
            field=field, agent_pos=agent_pos,
            targets=targets, bomb_positions=bomb_positions, 
            opponents=opponents, explosion_map=explosion_map)
        
        if res is None or res[0] is None: 
            return None
        
        return res[0] 
        
    
    # Coin potential: w_coin + (1/(dist_to_nearest_coin + 1.0))

    if coins: 
        d_nearest_coin = safe_get_dist(coins)
        
        if d_nearest_coin is not None: 
            phi_coin = w_coin * (1.0 / (d_nearest_coin + 1.0))
        
    # Create potential: w_crate + (1/(dist_to_nearest_crate + 1.0))
    if crates: 
        d_nearest_crate = safe_get_dist(crates)
        
        # Found that agent wants to go after crates instead of coins, so we want to make sure 
        # that the coins are more appetizing (collect coin first before bombing)

        if d_nearest_crate is not None: 
            phi_crate = w_crate * (1.0 / (d_nearest_crate + 1.0))
    
    
    # Kill Potential: w_kill * sum_{e in enemies}(1/(distance_to_e + 1)) 
    # Trap Potential: w_trap * sum{e in enemies}I_trap(e)

    if opponents:
        
        sum_opponents = 0.0 
        sum_trapped = 0.0 
        nearest_opponent_dist = float('inf')
        
        for opponent in opponents: 
            
            # Grab all opponents of the current opponent
            op_opponents = {op for op in entities if op != opponent}   
            
            # Find the distance to the opponent 
            res = spatial.get_nearest_target(
                field=field, agent_pos=agent_pos, 
                targets=[opponent], bomb_positions=bomb_positions, 
                opponents = op_opponents, explosion_map=explosion_map)
            
            d_opponent = res[0] if (res is not None and res[0] != float('inf')) else None # it might be res[1] - where does it come from?
            

            if d_opponent is not None: 
                sum_opponents += 1.0 / (d_opponent + 1.0)
                nearest_opponent_dist = min(nearest_opponent_dist, d_opponent)
        
            # Determine value of indicator (I_trap(e))
            is_trapped = spatial.is_entity_trapped(
                field=field, entity_pos=opponent, 
                bombs=bombs, opponents=op_opponents, 
                explosion_map=explosion_map)
            
            sum_trapped += int(is_trapped) 
            

        phi_kill = w_kill * sum_opponents 
        phi_trap = w_trap * sum_trapped
        
                        
    step_logger.debug(f"[PHI DEBUG] phi_danger: {phi_danger: .4f} |" 
                        f" phi_explore: {phi_explore: .4f} |"
                        f" phi_coin: {phi_coin: .4f} |" 
                        f" phi_crate: {phi_crate: .4f} |"
                        f" phi_kill: {phi_kill: .4f} |"
                        f" phi_trap: {phi_trap: .4f} |")    

    # Return the the sum of each of the parts 
    return float(phi_danger + phi_explore + phi_coin + phi_crate + phi_kill + phi_trap)


def compute_reward(game_state: dict, next_game_state: dict, events: list, terminal: bool,
                   event_rewards: dict, potential_weights: dict, gamma: float, 
                   visited_tiles: set, step_logger: logging.Logger) -> float: 
        """Calculate total reward by combining the raw event rewards and custom state potentials. 

        R'(s, a, s') = R_base + gamma * Phi(s') - Phi(s)

        game_state: dict containing current game state
        next_game_state: dict containing game state after taking an action 
        events: list of events supplied by game engine
        terminal: whether or not this is the terminal state
        return: float reward for taking an action and transitioning from 
                one state to the next state
        """

        reward = 0.0 

        # Loop through our found raw rewards and add value into our reward computation 
        for event in events: 
            if event in event_rewards: 
                
                step_logger.debug(f"event: {event} |" 
                                       f" reward: {event_rewards[event]}")
                
                reward += event_rewards[event]
   
                    
        # Calculate potential calculation for the current game state (phi(s))
        phi_state = compute_potential(game_state, potential_weights, visited_tiles, step_logger)

        if terminal or next_game_state is None: 
            phi_next = 0.0
            
        else: 
            phi_next = compute_potential(next_game_state, potential_weights, visited_tiles, step_logger)
        
 
        potential_reward = np.clip((gamma * phi_next) - phi_state, -2.0, 2.0)
     
        # Compute net reward 
        # Since we are working with changing potential rewards
        # We will not use the DQN clipping convention [-1.0, 1.0] and will instead 
        # only clip extreme changes in variance. 
        # The largest possible change is 
        # killed self + got_killed + potential term:
        
        #  -5 - 5 -2 = -12 (peaceful)
        #  -8 - 3 - 2 = -13 (aggressive)
        
        # So we will use a bound of +/-15
        
        reward_net = float(reward + potential_reward)
        reward_net = np.clip(reward_net, -15.0, 15.0)
        
        # Record visited tiles, this will prevent back tracking as well as 
        # Making sure that that the agent tries something new 
        if game_state is not None: 
            visited_tiles.add(game_state['self'][3])
        
        step_logger.debug(
            f"[REWARD DEBUG] Base: {reward:.4f} |"
            f" Phi_s: {phi_state:.5f} | "
            f" Phi_s': {phi_next:.5f} |"
            f" dPhi: {potential_reward:.5f} |"
            f" Net: {reward_net:.4f}")
        
        return reward_net
    
class TabularQAgent: 
    """Tabular Q-learning handler"""
    
    def __init__(self, model, gamma: float=0.99, alpha: float=0.1, model_type: str="lut", 
                 behavior: str="peaceful", base_rewards: dict = None, 
                 potential_weights: dict = None, logger: logging.Logger = None, 
                 step_logger: logging.Logger = None):
        
                
        # Create environment variables to allow unmasked vs masked run 
        self.use_mask = os.environ.get('_MASK', 'True').strip().lower() == 'true' # Default to true if not set 

        
        self.model = model 
        self.model_type = model_type 
        self.gamma = gamma 
        self.alpha = alpha 
        self.behavior = behavior
        self.potential_weights = potential_weights or {}
        self.event_rewards = base_rewards or {} 
        
        self.logger = logger or logging.getLogger(__name__)
        self.step_logger = step_logger or self.logger
        
        self.visited_tiles = set() 
        
        # Track history of metrics
        self.loss_history = [] 
        self.critic_loss_history = []
        self.mean_reward_history = [] 
        self.entropy_history = []
        
        self.update_count = 0
        self._episode_td_errors = [] 
        self._episode_rewards = [] 
        

    def reset_episode(self):
        self.visited_tiles.clear()
        
    def compute_reward(self, game_state: dict, next_game_state: dict, events: list, terminal: bool) ->float: 
        """ Delegate for computing the reward

        Args:
            game_state (dict): Current state of the game
            next_game_state (dict): State after choosing an action 
            events (list): List of current events
            terminal (bool): If the current state is the terminal state

        Returns:
            float: Computed reward
        """
    
        return compute_reward(game_state, next_game_state, events, terminal, 
                              self.event_rewards, self.potential_weights, self.gamma, 
                              self.visited_tiles, self.step_logger)
        
    @torch.no_grad()
    def step_update(self, state_idx: int, action: int, reward: float,
                 next_state_idx: "int | None", terminal: bool,
                 epsilon: float, next_game_state: dict = None) -> float:
        """Apply one TD update to the Q-table and track per-episode stats."""
        
        # Apply mask 
        next_legal_mask = None
        if not terminal and next_game_state is not None:
            from .callbacks import action_mask
            next_legal_mask = torch.tensor(action_mask(next_game_state), dtype=torch.bool)

        # Calculate TD error 
        td_error = self.model.td_update(
            state_idx=state_idx, action=action, reward=reward,
            next_state_idx=None if terminal else next_state_idx,
            alpha=self.alpha, gamma=self.gamma,
            next_legal_mask=next_legal_mask)

        self._episode_td_errors.append(abs(td_error))
        self._episode_rewards.append(reward)

        self.step_logger.debug(
            f"[Q-UPDATE] state={state_idx} action={action} reward={reward:+.4f}"
            f"next_state={next_state_idx} termimal={terminal} td_error={td_error:+.4f}"
            f"episilon={epsilon:.4f}")
        
        # If terminal make sure to flush the episode 
        if terminal:
            self.flush_episode(epsilon)

        return td_error
        
    
    def flush_episode(self, epsilon: float):
        
        # Find the mean temporal difference and reward
        mean_td = float(np.mean(np.abs(self._episode_td_errors))) if self._episode_td_errors else 0.0
        mean_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        
        # Store values into history 
        self.loss_history.append(mean_td)
        self.mean_reward_history.append(mean_reward)
        self.entropy_history.append(epsilon)
        
        self.update_count += 1
        self.logger.info(f"[Q-LEARNING UPDATE #{self.update_count}] |"
                         f" mean TD error per episode = {mean_td:.4f} |"
                         f" mean reward = {mean_reward:+.4f} |"
                         f" epsilon={epsilon:.4f}")
        
        self._episode_td_errors.clear()
        self._episode_rewards.clear()
   