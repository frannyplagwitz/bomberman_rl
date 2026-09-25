
""" All functions and information regarding RL heurisitics for PPO Actor-Critic

"""
import os
import torch 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

from torch.distributions import Categorical

from .callbacks import state_to_features, action_mask
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
            
            d_opponent = res[0] if (res is not None and res[0] != float('inf')) else None
            

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
    

class PPOAgent:
    """ PPO Actor-Critic handler"""
        
        
    def __init__(self, model: torch.nn.Module, lr: float = 3e-4, lr_end: float = None,
                    lr_decay_updates: int = None, gamma: float=0.99, clip_eps: float = 0.2, 
                    model_type: str = "mlp", behavior: str = "peaceful", 
                    potential_weights: dict = None, base_rewards: dict = None, 
                    entropy_start: float = 0.05, entropy_end: float = 0.01, 
                    entropy_decay_updates: int = 150, logger: logging.Logger = None, 
                    step_logger: logging.Logger = None):
                 
                 

        """ Setup values necessary for a PPO agent
        
        Args: 
            model (torch.nn.Module): Model that is being used (lookup table, MLP, CNN)
            lr (float): learning rate for updates, le-4 by default 
            gamma (float): discount factor for reinforcement learning. 0.99 by default 
            clip_eps (float): value updates are clipped down to. 0.20 by default
            model_type (str): type of model used, mlp by default 
            behavior (str): behavior mode of the agent. peaceful by default 
            potential_weights (dict): weights for potential function calculations
            base_rewards (dict): base rewards for different events
            
            entropy_start (float): entropy start range value used when calculating total loss 
            entropy_end (float): entropy end range value used for calculating total loss
            entropy_decay_updates: number of times entropy should be updated
            
            logger (logging.Logger): logger for low-volume per-episode/per-update summaries.
                Defaults to a module logger (so this class still works if used outside train.py).
                
            step_logger (logging.Logger): logger for high-volume per-step traces (reward/danger
                breakdowns - one of these fires on every single step). Defaults to `logger` if not
                given, so passing nothing still works, it just won't be split into its own file.
        
        """ 
        
        # Logging values 
        self.loss_history = [] 
        self.critic_loss_history = [] 
        self.mean_reward_history = [] 
        self.entropy_history = []
        self.update_count = 0

        # Get loggers
        self.logger = logger or logging.getLogger(__name__)
        self.step_logger = step_logger or self.logger


        # Initialize the PPO elements
        self.model = model 
        self.model_type = model_type
        self.gamma = gamma 
        self.clip_eps = clip_eps
        self.buffer = GameplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Add learning rate (LR) decay from lr start to lr end so that updates become smaller as training 
        # progresses, hopefully resulting in faster convergence 
        self.lr_start = lr
        self.lr_end = lr_end if lr_end is not None else lr   # None => no decay, backward compatible
        self.lr_decay_updates = lr_decay_updates or 1
        
        # Load behavior settings and potential weights for reward shaping 
        self.behavior = behavior
        self.potential_weights = potential_weights or {}
        self.event_rewards = base_rewards or {} 
        
        #  Decay the entropy linearly from start to end
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.entropy_decay_updates = entropy_decay_updates
        
        
        # Episodic episode tracker 
        self.visited_tiles = set()
        
        # Reward normalizer
        self.return_rms = RunningMeanStd()
        self.running_return = 0.0
        
        
    def reset_episode(self):
        """Reset visited tiles per episode"""
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
    
    def update(self, ppo_epochs: int = 4, batch_size: int = 64): 
    
        """ 
        Reviews batch of past transitions (S,A,R,S') from a memory buffer.
        Calculates how good/bad past actions were by comparing the actor 
         policy with critic state-value results
        Adjusts network weights to improve agent decision making 
        Occurs after gameplay.
        """
        
        # Note that we need to update using batches
        if len(self.buffer.states) == 0: 
            return 
        
        # Linearly decay learning rate
        lr_progress = min(1.0, self.update_count / max(1, self.lr_decay_updates))
        current_lr  = self.lr_start + (self.lr_end - self.lr_start) * lr_progress
        for pg in self.optimizer.param_groups:
            pg['lr'] = current_lr
            
    
        # 1. Updates agent's policy using policy gradients and adjusts in the direction that maximizes the expected cumulative reward
        # 2.  Maximizes a surrogate objective that measures improvements over old policy: 
        #   L(theta) = E_t[\frac{policy}{old policy} * A_t]
        #. 3. Uses clipping to limit the probability ratio between new and old policies to prevent excessively large policy updates
        #  4. Computes advantage A_t to determine how much better or worse an action was compared to expected value of the state
        #  5. Guides policy update by increasing probability of actions

        device = next(self.model.parameters()).device
        
        # Obtain properties from buffer
        states    = torch.stack(self.buffer.states).to(device)
        actions   = torch.tensor(self.buffer.actions, dtype=torch.long, device=device)
        log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=device)
        rewards   = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        terminals = torch.tensor(self.buffer.terminals, dtype=torch.float32, device=device)
        
        # Read in masks based on type
        if isinstance(self.buffer.masks[0], torch.Tensor):
            masks = torch.stack(self.buffer.masks).to(device=device, dtype=torch.bool)
            
        elif isinstance(self.buffer.masks[0], np.ndarray):
            masks = torch.from_numpy(np.array(self.buffer.masks)).to(device=device, dtype=torch.bool)
        
        else: 
            masks = torch.tensor(self.buffer.masks, dtype=torch.bool, device=device)
        
        
        # Process next states
        next_states_list = [] 
        for s_next in self.buffer.next_states: 
            if s_next is None: 
                # Terminal state
                next_states_list.append(torch.zeros_like(states[0]))
                
            elif isinstance(s_next, dict):
                features = state_to_features(s_next, self.model_type)
                next_states_list.append(features.detach().to(dtype=torch.float32))
            
            elif torch.is_tensor(s_next):
                next_states_list.append(s_next.detach().to(dtype=torch.float32))

            elif isinstance(s_next, np.ndarray):
                next_states_list.append(torch.from_numpy(s_next).to(dtype=torch.float32))
            else:
                next_states_list.append(torch.as_tensor(s_next, dtype=torch.float32))
            
        
        next_states = torch.stack(next_states_list).to(device)

    
        # Normalize the reward
        raw_rewards = rewards.clone()
        
        # Walks through the buffer step by step and calculates a running discounted accumulator
        # running return = running return * discount + reward 
        # Used to cheaply check how large the discounted returns become instead of doing a 
        # backward computation 
        
        running_returns = []
        for r, done in zip(rewards.tolist(), terminals.tolist()):
            self.running_return = self.running_return * self.gamma + r
            running_returns.append(self.running_return)
            
            if done: 
                self.running_return = 0.0
        
        # Feed batch into RunningMeanStd and normalize the rewards using the found
        # standard deviation estimate 
        self.return_rms.update(torch.tensor(running_returns, device=device))
        rewards = rewards / self.return_rms.std
    
        # Compute value estimates using GAE
        self.model.eval()
        with torch.no_grad(): 
            _, current_values = self.model(states)
            _, next_values = self.model(next_states)
            
            current_values = current_values.reshape(-1)
            next_values = next_values.reshape(-1)
        
            
            # Calculate TD error via Generalized Advantage Estimation (GAE)
            # Apparently PPO works better with GAE instead of 1-step TD

            # A_t = target - V(s)
            # target = R_{t+1} + \gamma V_w(S_{t+1})
            
            gae = 0
            gae_lambda = 0.95
            advantages = torch.zeros_like(rewards)
            
            
            for t in reversed(range(len(rewards))):

                if terminals[t]: 
                    delta = rewards[t] - current_values[t]
                    gae = delta 
                    
                else: 
                    delta = rewards[t] + self.gamma * next_values[t] - current_values[t]
                    gae = delta + self.gamma * gae_lambda * gae
                    
                advantages[t] = gae
                
            
            targets = advantages + current_values
            
            # Normalize the advantages
            if len(advantages) > 1: 
                std = advantages.std()
                
                if not torch.isnan(std) and std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (std + 1e-8)       
                         
               
        self.model.train()
        dataset_size = len(states)
        
        # loop over multiple epochs
        for _ in range(ppo_epochs):
            permutation = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = permutation[start_idx:start_idx + batch_size]
                
                # Obtain the properties for each batch 
                b_states     = states[batch_indices]
                b_actions    = actions[batch_indices]
                b_log_probs  = log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_targets    = targets[batch_indices]
                b_masks      = masks[batch_indices]
                
                # Get actor and critic results from model 
                logits, current_values = self.model(b_states)
                current_values = current_values.reshape(-1)
                
                if logits.dim() == 3:
                    logits = logits.squeeze(1) # need to make sure that shape is (N, 6)
                    

                # Make masked tiles unappealing 
                masked_logits = logits.masked_fill(~b_masks, -1e9)
                dist = Categorical(logits=masked_logits)
                
                # Find new policy value and the loss ratio between it and the old policy
                new_log_probs = dist.log_prob(b_actions)
                ratios = torch.exp(new_log_probs - b_log_probs)
                
                # We know that for advantage: 
                # At > 0 is good, so maximizing the value will increase the ratio
                # At < 0 is bad, so maximizing the value will decrease the ratio 
                
                
                # Find the clipped surrogate objective
                unclipped = ratios * b_advantages
                clipped   = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                
                 # Find actor and critic loss
                actor_loss  = -torch.min(unclipped, clipped).mean()
                critic_loss =  F.mse_loss(current_values, b_targets)
 
                # Calculate entropy based on our coefficient
                progress     = min(1.0, self.update_count / max(1, self.entropy_decay_updates))
                entropy_coef = self.entropy_start + (self.entropy_end - self.entropy_start) * progress
                entropy      = dist.entropy().mean()
                total_loss   = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
                
                # Perform optimization 
                self.optimizer.zero_grad()
                total_loss.backward()
        
                # Capture raw gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)
                self.optimizer.step()
                
                # Cache loss items for logging purposes 
                last_total_loss  = total_loss.item() 
                last_actor_loss  = actor_loss.item() 
                last_critic_loss = critic_loss.item()
                last_entropy     = entropy.item() 
                last_grad_norm   = grad_norm.item() 
                
            
        # Logging + memory cleanup
        self.loss_history.append(last_total_loss)
        self.critic_loss_history.append(last_critic_loss)
        self.mean_reward_history.append(raw_rewards.mean().item())
        self.entropy_history.append(last_entropy)
    
        self.update_count += 1
        
        self.logger.info(f"[PPO UPDATE #{self.update_count}] | LR: {current_lr:.6f}")
        self.logger.info(f"Buffer Size: {dataset_size}")
        self.logger.info(f"| Rewards:     "
                         f" Total = {raw_rewards.sum().item():.2f} |"
                         f" Mean Step Reward = {rewards.mean().item():.3f} |"
                         f" Normalizer Std = {self.return_rms.std:.3f} |")
        
        self.logger.info(f" Loss Summary:"
                         f" Total = {last_total_loss:.4f} |"
                         f" Actor = {last_actor_loss:.4f} |"
                         f" Critic = {last_critic_loss:.4f} |")
        
        self.logger.info(f" Metrics: Entropy = {last_entropy:.4f} |"
                         f" Grad Norm = {last_grad_norm:.4f} |")
        
        
        # Per-action advantage / reward 
        # Displays what the update actually is teaching the policy after each action 
        self.logger.info(" Per-Action Advantage / Raw Reward ")
        
        for a_idx, a_name in enumerate(ACTION_NAMES):
            a_mask = (actions == a_idx)
            count = int(a_mask.sum().item())
            
            if count == 0:
                continue
            
            mean_adv = advantages[a_mask].mean().item()
            mean_rewards = raw_rewards[a_mask].mean().item()
            self.logger.info(f"|   {a_name:<6s} (n={count:3d}) : "
                             f"Mean Advantage = {mean_adv:+.4f} |" 
                             f" Mean Raw Reward = {mean_rewards:+.4f}")
        
        
        self.buffer.clear()

            
    
