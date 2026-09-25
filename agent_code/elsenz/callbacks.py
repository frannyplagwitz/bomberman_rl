import os
import sys

import torch
import numpy as np

import settings as s
from . import spatial_feature_extractor as spatial

from .LUT import TabularQAgent 



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

# For parallel runs, keep track of the base directory 
BASE_DIR = os.environ.get("ELSENZ_RUN_DIR", ".")
 

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.num_actions = len(ACTIONS)
    self.behavior = os.environ.get('ELSENZ_BEHAVIOR', getattr(self, 'behavior', 'peaceful')) # Default to peaceful behavior if not specified
    self.episodes_per_update = int(os.environ.get('ELSENZ_EPISODES_PER_UPDATE', '4')) # Episodes to store before update, defaults to 4
        
    
    self.last_log_prob = torch.tensor(0.0) # iniitalize last log prob to 0.0
    self.last_mask = torch.ones(self.num_actions, dtype=torch.bool) # initialize last action mask 
    
    # Scenario 
    self.scenario = get_scenario()
    self.num_rounds = get_rounds() 
    print(f"[AGENT SETUP] Running in scenario: {self.scenario}")
    
    # Get opponents number
    self.num_opponents = get_num_opponents()
        
    
    # Instantiate the network architecture and optimizer
    self.device = torch.device("cpu") # Only going to do stuff with CPU 
    
    self.model = build_model(self)
    self.model.to(self.device)
    
    # Choose file to read in from depending on the model and behavior
    #Ex. if using defaults, the file will be "actor-critic-mlp-peaceful.pt"
        
    
    file_name = os.path.join(BASE_DIR, f"elsenz-lut-{self.behavior}-{self.scenario}.pt")

    # Let training run pick up weights from a checkpoint saved
    if os.path.isfile(file_name) and not getattr(self, 'train', False):
        self.logger.info(f"Loading model from saved state from file {file_name}.")
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))
        self.model.eval()    # Set to evalution mode 

    else: 
        self.logger.info("Setting up model from scratch")
        if getattr(self, 'train', False):
            self.model.train()
            
        else:
            self.model.eval()
            
def build_model(self):
        
    """ Initializes model structure and optimizer for the specified model type
        Required because each model will function better with 
        different hyperparameter specifications
            
        Note: Something we can try is adjusting the learning rates for our models    
        """
    model = TabularQAgent(num_states=250, action_dim=self.num_actions)      
    return model


def action_mask(game_state: dict) -> np.ndarray: 
    """
    Return a binary mask for 
    [UP, RIGHT, DOWN, LEFT, BOMB, WAIT]

    1 = valid action
    0 = invalid action
    """

    mask = np.ones(6, dtype=np.float32)

    if game_state is None:
        return mask

    field = game_state["field"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    explosion_map = game_state.get("explosion_map", np.zeros_like(field))

    _, _, bomb_available, (x, y) = game_state["self"]
    
    # one step safety margin that is used to make sure that a tile is actually safe
    SAFETY_MARGIN = 1
    
    # Get positions occupied by bombs, opponents and tiles that are dangerous 
    # (avoid them if about to explode)
    bomb_positions = spatial.get_bomb_positions(bombs)
    opponents = {other[3] for other in others}
    directions = spatial.get_directions() # Get directions (UP, RIGHT, DOWN, LEFT)
    

    # Get danger map and also map of timers     
    danger_tiles = spatial.build_danger_map(field, bombs, explosion_map)
    timer_map    = spatial.build_bomb_timer_map(field, bombs)
    

    # Check if each movement action is valid [UP, RIGHT, DOWN, LEFT]
    for i, (dx, dy) in enumerate(directions):
        nx, ny = x + dx, y + dy
        
        # If neighbor not open, set mask to 0 and look at next neighbor 
        if not spatial.is_tile_open(field, (nx, ny), bomb_positions, opponents, explosion_map):
            mask[i] = 0.0
            continue
        
        # Check if destination tile will put agent in immediate danger
        if (nx, ny) in danger_tiles: 
            steps_to_safety, _, is_trapped = spatial.evaluate_safety(
                field, (nx, ny), danger_tiles, 
                bomb_positions, opponents, 
                explosion_map, bombs)
            
            # If the destination tile has a bomb that's about to go off, don't want to go there
            # If we don't add SAFETY_MARGIN, WAIT will be a legal action until the agent needs 
            # every remaining move to escape , which means there's no ability for the agent 
            # to explore
            
            min_timer = timer_map[nx, ny]
            
            if is_trapped or (steps_to_safety + SAFETY_MARGIN) >= min_timer: 
                mask[i] = 0.0
                
                
    # Check if BOMB action is valid
    if not bomb_available or (x, y) in bomb_positions: 
        mask[4] = 0.0
    
    else: 
        # Don't want to commit suicide 
        sim_bombs = list(bombs) + [((x, y), s.BOMB_TIMER)]
        sim_danger = spatial.build_danger_map(field, sim_bombs, explosion_map)
        sim_positions = bomb_positions.union({(x, y)})
        
        # Check if agent can escape if it places its bomb 
        sim_steps, _, sim_trapped = spatial.evaluate_safety(
            field = field, 
            entity_pos=(x, y), 
            danger_tiles = sim_danger, 
            bomb_positions=sim_positions,
            opponents=opponents,
            explosion_map=explosion_map,
            bombs=sim_bombs)
        
        sim_timer_map = spatial.build_bomb_timer_map(field, sim_bombs)
        min_sim_timer = sim_timer_map[x, y]

        if sim_trapped or (sim_steps + SAFETY_MARGIN) >= min_sim_timer:
            mask[4] = 0.0
        
        
    # Check if WAIT action is valid 
    # (do not want to wait in an area where agent is in immediate danger)
    
    if (x, y) in danger_tiles:
        steps_to_safety, _, is_trapped = spatial.evaluate_safety(
            field, (x, y), danger_tiles, bomb_positions,
            opponents, explosion_map, bombs)

        min_timer = timer_map[x, y]

        # Add safety margin to make sure the agent can actually escape and 
        # its not a false positive
        if is_trapped or (steps_to_safety + SAFETY_MARGIN) >= min_timer:
            mask[5] = 0.0
    
    return mask 
                    
        
def act(self, game_state: dict) -> str:
    """
    Called by the game engine during every turn to select an action 
    Maximum execution time is 0.5s per turn
    
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # If game state is empty, just perform wait action 
    if game_state is None: 
        return ACTIONS[5]    

    # Extract features and make sure that we have the correct tensor type and device
    state_features = state_to_features(game_state)


    # Get features based on the model type 

    if not isinstance(state_features, torch.Tensor):
        state_features = torch.tensor(state_features, dtype=torch.long)
    
    state_tensor = features_to_tensor(state_features).view(1).to(self.device)


    # Extract action mask 
    mask_np = action_mask(game_state)
    mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
    
    # If all actions are masked, unmask the wait action 
    if not mask_tensor.any(): 
        mask_tensor[5] = True
    
    with torch.no_grad():
        
        # Grab q_values from tensor, check if it blew up 
        q_values = self.model(state_tensor)
        if torch.isnan(q_values).any():
            print(f"\n[NaN DETECTED AT STEP {game_state['step']}] Q-table blew up!")

    masked_q = q_values.masked_fill(~mask_tensor.unsqueeze(0), -1e9)

    if getattr(self, 'train', False):
        # If training, mask out invalid actions and choose action based on epsilon 
        
        epsilon = get_curr_epsilon(self)
        legal_actions = mask_tensor.nonzero(as_tuple=True)[0].tolist()
        
        if np.random.rand() < epsilon:
            action = int(np.random.choice(legal_actions))
        else:
            action = int(torch.argmax(masked_q, dim=-1).item())

        # Store old state, action and mask 
        self.last_state = state_tensor.squeeze(0)
        self.last_action = action
        self.last_mask = mask_tensor.cpu()
    else:
        action = int(torch.argmax(masked_q, dim=-1).item())

    return ACTIONS[action]
    
    
# Functions to retrieve values from arguments
def get_scenario() -> str: 
    """Get current scenario based on arg passed to main.
    Returns: scenario sent, defaults to classic
    """
    if "--scenario" in sys.argv: 
        idx = sys.argv.index("--scenario")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        
    return "classic"
    
def get_num_opponents() -> int: 
    """Get the number of opponents based on arg passe to main.
    Returns: number of opponents, defaults to 0
    """
    
    agents_list = [] 

    if "--agents" in sys.argv: 
        idx = sys.argv.index("--agents")
        
        # Collect positional arguments following --agents
        for arg in sys.argv[idx + 1:]:
            if arg.startswith("-"):
                break
            agents_list.append(arg)
    
    # Count all opponents 
    opponents = [a for a in agents_list if a != "elsenz"]
    return len(opponents)


def get_rounds() -> int: 
    """Get the number of rounds training.
    Returns: number of rounds training, defaults to 0
    """
    
    if "--n-rounds" in sys.argv: 
        idx = sys.argv.index('--n-rounds')
        
        if idx + 1 < len(sys.argv):
            return int(sys.argv[idx + 1])
        
    return 0 

def get_curr_epsilon(self, eps_start: float = 1.0, eps_end: float = 0.05, 
                     eps_decay_episodes: int = 300) -> float:
    """Linearly decay epsilon for epsilon-greedy exploration"""
    
    if not getattr(self, 'train', False):
        return 0.0
    
    episode = len(getattr(self, 'episode_lengths', []))
    progress = min(1.0, episode / max(1, eps_decay_episodes))
    return eps_start + (eps_end - eps_start) * progress 

   
def state_to_features(game_state: dict) -> torch.Tensor:
    """Extract features from game state based on agent's model type 
        :param game_state: A dictionary describing the current game board 
        :return tensor containing extracted features
    """
    
    if game_state is None: 
        return torch.zeros(1)
    
    
    # Determine hash key for location in the table
    # Extract features into a 1D vector
    feat_key = get_table_key(game_state)
        
    # Convert the state tuple hash into an integer index
    hash_idx = int(abs(feat_key))
    return torch.tensor(hash_idx, dtype=torch.long)
    
    
def features_to_tensor(features): 
    """Return table index tensors for LUT and float32 """

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    
    return features.to(dtype=torch.long)

        
def extract_features_1D (game_state: dict) -> np.ndarray: 
    """
    Extract features into a 1D array
    
    Channels included (17 x 17) = 289 features per channel 
    0: Walls    ( 1 for wall, 0 otherwise)
    1: Crates   ( 1 for crate, 0 otherwise)
    2: Coins    ( 1 for coin, 0 otherwise)
    3: Entities (+1 for self, -1 for enemy, 0 otherwise)
    4: Bomb timers  (normalized timer )
    5: Danger tiles
    
    :param game_state: A dictionary describing the current game board 
    :returns 1D array containing extracted features 
    """
    if game_state is None: 
        return np.zeros(1742, dtype=np.float32)
    
    
    # Just flatten 3D grid into 1D -> (1742, )
    features_3d = extract_features_3D(game_state).flatten().astype(np.float32)
    
    # Extract scalar features (4, )
    spatial_features = spatial.extract_spatial_features(game_state)
    
    # Concatenate the features into a unified array (1742, )
    return np.concatenate([features_3d, spatial_features])
    
                    
def get_table_key(game_state: dict) -> tuple: 
    """ Generates hashable key for the state representation for actor-critic lookup tables
    Args:
        game_state (dict): the current game board 

    Returns:
        tuple: hashable state key 
    """
    
    if game_state is None: 
        return None 

    
    # Parse raw game state into important parts
    field = game_state['field']
    agent_pos = game_state['self'][3]
    coins     = game_state['coins']
    crates = [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x, y] == 1]


    bombs = game_state['bombs']
    opponents = [other[3] for other in game_state['others']]
    explosions = game_state['explosion_map']

    bomb_positions = spatial.get_bomb_positions(bombs)
    opponents_set  = set(opponents) if opponents else set()

    coins_crates = coins + crates # combine coins and crates



    # Extract danger info 
    in_danger, (escape_dx, escape_dy) = spatial.find_escape_route(
        field=field, 
        agent_pos=agent_pos,
        bombs=bombs,
        opponents=opponents_set, 
        explosion_map=explosions
    )

    # Extract coin crate info 
    coin_crate_info = spatial.get_nearest_target(
        field=field, 
        agent_pos=agent_pos, 
        targets=coins_crates, 
        bomb_positions=bomb_positions, 
        opponents=opponents_set, 
        explosion_map=explosions)


    if coin_crate_info is None or coin_crate_info[1] is None: 
        coin_crate_dx, coin_crate_dy = 0, 0
    
    else:
        _, (coin_crate_dx, coin_crate_dy) = coin_crate_info
        
        
        
    # Extract opponent info, note that here we don't want to treat opponents as obstacles
    # because we are actively looking for them  
    opponent_info = spatial.get_nearest_target(
        field=field, 
        agent_pos=agent_pos,
        targets=opponents,
        bomb_positions=bomb_positions,
        opponents=None,
        explosion_map=explosions)
    
    if opponent_info is None or opponent_info[1] is None: 
        opponent_dx, opponent_dy = 0, 0
    
    else:
        _, (opponent_dx, opponent_dy) = opponent_info
        
    
    

    # Spatial direction map: 
    dir_map = {(0, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3, (-1, 0): 4}
    
    f_danger = 1 if in_danger else 0                    # 0 or 1, 2 total states
    f_escape = dir_map.get((escape_dx, escape_dy), 0)   # cardinal directions + wait = 5 states
    f_coin_crate = dir_map.get((coin_crate_dx, coin_crate_dy), 0) # cardinal directions + wait = 5 
    f_opponent = dir_map.get((opponent_dx, opponent_dy), 0) # cardinal directions + wait = 5
    
    
    
    
    # Convert multidimensional features into one single key
    return int(f_danger + (f_escape * 2) + (f_coin_crate  * 2 * 5) + (f_opponent * 2 * 5 * 5) )
    
  
def extract_features_3D (game_state: dict) -> np.ndarray: 
    """
    Extract features into a 3D array
    :param game_state: A dictionary describing the current game board
    :returns 3D Pytorch feature tensor of size (6,17,17) containing extracted features from game state 
    """
    # Return 3D array of shape (height, width, channels) where channels = 4, and height and width are both 17
    # Channel 0: Walls (1 if wall, 0 otherwise)
    # Channel 1: Crates (1 if crate, 0 otherwise)
    # Channel 2: Coins (1 if coin, 0 otherwise)
    # Channel 3: Entities (+1 for self, -1 for enemies, 0 otherwise)
    # Channel 4: Bomb positions and timers (timer/max_timer, range from 0.0 to 1.0)
    # Channel 5: Danger tiles (danger tiles normalized by explosion time for each (1.0 == exploding now) )
    
    # Important note: normally the structure is (height, width, channels)
    #   With pytorch it is (channels, height, width)
    #  Pytorch is apparently better than numpy for processing channels for grids (figure out why)
    
    if game_state is None: 
        return np.zeros((6, 17, 17), dtype=np.float32)
        
        
    field = game_state['field']
    width, height = field.shape
    max_timer = float(s.BOMB_TIMER)
    
    
    # Initialize tensor (channels, height, width)
    tensor = np.zeros((6, height, width), dtype=np.float32)
    
    # Channel 0 (Walls)
    # If value in field is -1 = 1, else 0
    tensor[0, field == -1] = 1.0
    
    # Channel 1 (Crates)
    # If value in field is 1 = 1, else 0 
    tensor[1, field == 1] = 1.0
    
    # Channel 2 (Coins)
    # Find coin positions and set to 1, else 0
    for cx, cy in game_state.get('coins', []):
        tensor[2, cx, cy] = 1.0
        
    # Channel 3 (Entities)
    # If agent position, set to +1, if enemy, set to -1 else 0
    if 'self' in game_state and game_state['self']:
        ax, ay = game_state['self'][3]
        tensor[3, ax, ay] = 1.0
    
    for other in game_state.get('others', []):
        ex, ey = other[3]
        tensor[3, ex, ey] = -1.0
    
        
    # Channel 4 (Bomb Postions & Timers)
    # Map bomb coordinates with normalized remaining time 
    
    bombs = game_state.get('bombs', [])
    for (bx, by), timer in bombs: 
        tensor[4, bx, by] = (float(timer) + 1.0) / (max_timer + 1.0)
    

    # Channel 5 (Danger Tiles)
    # Get danger tiles and normalize them based on imminence (1.0 = exploding now, lower = explodes later)
    explosion_map = game_state.get('explosion_map', None)
    if explosion_map is not None: 
        tensor[5, explosion_map > 0] = 1.0 # Active explosion 
    
    # Project blast radii across all directions for active bombs 
    for (bx, by), timer in bombs:
        imminence = (max_timer - float(timer) + 1.0) / (max_timer + 1.0)
        
        # Get all tiles in blast path 
        danger_tiles = spatial.get_danger_tiles(field, (bx, by))
        
        # Take the maximum imminence score if a tile is covered by multiple bombs 
        for tx, ty in danger_tiles: 
            tensor[5, tx, ty] = max(tensor[5, tx, ty], imminence)
    
    return tensor     