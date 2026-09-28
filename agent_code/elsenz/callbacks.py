import os
import sys

import torch
from torch.distributions import Categorical 
import numpy as np

import settings as s
from . import spatial_feature_extractor as spatial

from . import LUT as lut_models
from .LUT import TabularQAgent
from . import state_symmetry


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
    :param model_type: Type of model we are building, default is lookup table 
    :param behavior: Agent behavior, default is peaceful
    """

    self.num_actions = len(ACTIONS)
    self.behavior = os.environ.get('ELSENZ_BEHAVIOR', getattr(self, 'behavior', 'peaceful')) # Default to peaceful behavior if not specified
    self.episodes_per_update = int(os.environ.get('ELSENZ_EPISODES_PER_UPDATE', '4')) # Episodes to store before update, defaults to 4
    self.model_type = os.environ.get('ELSENZ_MODEL_TYPE', getattr(self, 'model_type', 'lut')) # Default to lookup table if not specified
    
    
    self.last_log_prob = torch.tensor(0.0) # iniitalize last log prob to 0.0
    self.last_mask = torch.ones(self.num_actions, dtype=torch.bool) # initialize last action mask 
    
    # Scenario 
    self.scenario = get_scenario()
    self.num_rounds = get_rounds() 
    print(f"[AGENT SETUP] Running in scenario: {self.scenario}")
    
    # Get opponents number
    self.num_opponents = get_num_opponents()
        
    
    # Instantiate the network architecture and optimizer
    # self.device = torch.device("cpu") # Only going to do stuff with CPU, Can discuss if we want to train using GPU though 
    self.device = 'cpu'
    # if torch.cuda.is_available():
    #     self.logger.info("CUDA is available, using GPU for training")
    #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    


    self.model = build_model(self)
    self.model.to(self.device)

    # Choose file to read in from depending on the model and behavior
    #Ex. if using defaults, the file will be "actor-critic-lut-peaceful.pt"
    file_name = (f"actor-critic-{self.model_type}-{self.behavior}-{self.scenario}"
                 f"-{self.num_rounds}-rounds-{self.num_opponents}-opponents.pt")
    
    # improved version is above, might break
    # file_name = os.path.join(BASE_DIR, f"elsenz-lut-{self.behavior}-{self.scenario}.pt")
    
    # Override to let training run pick up weights from another warmup checkpoint
    warm_start = getattr(self, 'warm_start', None) or os.environ.get('ELSENZ_WARM_START')

    # Let training run pick up weights from a checkpoint saved
    if os.path.isfile(file_name) and not getattr(self, 'train', False):
        self.logger.info(f"Loading model from saved state from file {file_name}.")
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))
        self.model.eval()    # Set to evalution mode 

    elif getattr(self, 'train', False) and warm_start: 
        if os.path.isfile(warm_start): 
            self.logger.info(f"Warm-start training from checkpoint {warm_start}")
            self.model.load_state_dict(torch.load(warm_start, map_location=self.device))
    
        else: 
            self.logger.warning(f"warm_start={warm_start} was set but file doesn't exist."
                                f"using freshly-initialized model instead")
        
        self.model.train()
    
    else: 
        self.logger.info("Setting up model from scratch")
        if getattr(self, 'train', False):
            self.model.train()
            
        else:
            self.model.eval()
            
def get_shared_table_path(self) -> str:
    """Path of the on-disk canonical Q-table for every bot trained with this
    behavior/scenario/opponent-count combination.

    Keying by (behavior, scenario, num_opponents) rather than by run/process means that
    launching several qfiac_agent processes with the same combo (e.g. via
    train_lut_parallel.sh, or several qfiac_agent instances in one --agents line) all warm-
    start from - and, at the end of training, merge their experience back into - the same
    file, pooling experience instead of each learning an isolated copy that's thrown away.
    A different behavior (different reward shaping) gets its own file so their conflicting
    Q-values never mix. See models/LUT.py's merge_tables()/load_canonical_table() and
    LUT_AGENT.md ("Train independently, merge afterwards") for how this file gets written.

    Set ELSENZ_TABLE_NAME to pin a bot to one named table (e.g. "peaceful_v1") instead,
    overriding the derived name above. This is how a single table gets trained and
    revisited across runs that would otherwise hash to different files - for example
    curriculum training the same table through several --scenario values in a row.

    A "-bombtimer" suffix is appended whenever ELSENZ_BOMB_TIMER_KEY is enabled (see
    bomb_timer_key_enabled()), on top of either the derived name or ELSENZ_TABLE_NAME. That
    option changes what a given integer key means (a different, larger
    get_num_table_states()), so a table built with it and one built without it can never be
    the same file - loading one into the other would silently misinterpret every row. This
    is what lets the option be introduced without touching any already-trained table: an
    existing "peaceful_v1.table" is simply never read or written by a process running with
    ELSENZ_BOMB_TIMER_KEY=1, which reads/writes "peaceful_v1-bombtimer.table" instead.
    """

    shared_dir = os.path.join(os.path.dirname(__file__), "shared_tables")
    suffix = "-bombtimer" if bomb_timer_key_enabled() else ""

    table_name = os.environ.get('ELSENZ_TABLE_NAME')
    if table_name:
        return os.path.join(shared_dir, f"lut-{table_name}{suffix}.table")

    return os.path.join(
        shared_dir,
        f"lut-{self.behavior}-{self.scenario}-{self.num_opponents}-opponents{suffix}.table")


def get_contributions_dir(self) -> str:
    """Directory each parallel LUT worker drops its end-of-run (weight, visit_counts)
    contribution into (models.LUT.save_contribution), for the training script to fold into
    the canonical table (models.LUT.merge_tables) once every worker has exited. Same keying
    as get_shared_table_path(), so it always sits next to the table it feeds into.
    """

    return get_shared_table_path(self)[: -len(".table")] + "-contributions"


def build_model(self):

    """ Initializes model structure and optimizer for the specified model type
        Required because each model will function better with
        different hyperparameter specifications

        Note: Something we can try is adjusting the learning rates for our models
        """

    if self.model_type == "lut":
        # Each process trains its own private table - no cross-process
        # It only touches the canonical table twice: once here, to warm-start from whatever's
        # already been learned, and once more at the end of training (train.py:end_of_round)
        # to contribute what it learned this run back.
        
        model = TabularQAgent(num_states=get_num_table_states(), action_dim=self.num_actions)
        lut_models.load_canonical_table(model, get_shared_table_path(self))

    else: 
        raise ValueError(f"Unknown model_type: {self.model_type}")            
        
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

    # one step safety margin that is used to make sure that a tile is actually safe
    # Action only counts as escapable if there's a full step after taking it 
    SAFETY_MARGIN = 1

    field = game_state["field"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    explosion_map = game_state.get("explosion_map", np.zeros_like(field))

    _, _, bomb_available, (x, y) = game_state["self"]
    
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
                    
        
def get_rotation_k(self, game_state: dict) -> int:
    """Quarter-turns to canonicalize this episode's game states for the LUT path
    (see state_symmetry.py). Fixed once per round from the agent's spawn corner
    at step 1, then held constant for the rest of the round - the corner is only
    meaningful at spawn, the agent walks away from it immediately after.
    """

    if game_state["step"] == 1 or not hasattr(self, "rotation_k"):
        field = game_state["field"]
        corner = state_symmetry.get_start_corner(
            game_state["self"][3], rows=field.shape[1], cols=field.shape[0])
        self.rotation_k = state_symmetry.ROTATION_BY_CORNER[corner]

    return self.rotation_k


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

    # For the LUT path, canonicalize the state (rotate it so the agent's spawn
    # corner is always top-left, see state_symmetry.py) before anything looks at
    # it, so get_table_key/action_mask both see - and stay consistent with - the
    # same canonical view the Q-table was trained on. The chosen action is
    # rotated back to real-world directions just before it's returned, at the
    # very end of the "lut" branch below.
    if self.model_type == "lut":
        rotation_k = get_rotation_k(self, game_state)
        model_state = state_symmetry.canonicalize(game_state, rotation_k)
    else:
        model_state = game_state

    # Extract features and make sure that we have the correct tensor type and device
    state_features = state_to_features(model_state, self.model_type)


    # Get features based on the model type
    if self.model_type == "lut":
        if not isinstance(state_features, torch.Tensor):
            state_features = torch.tensor(state_features, dtype=torch.long)
        state_tensor = features_to_tensor(state_features, self.model_type).view(1).to(self.device)

    # state_tensor = features_to_tensor(state_features).view(1).to(self.device)


    # Extract action mask. action_mask() itself is unchanged - it's a pure
    # function of whatever field/bombs/positions it's handed, so passing it the
    # canonicalized state (for "lut") is enough to get a mask that lines up with
    # the canonical-frame Q-values below; no separate mask-rotation step needed.
    mask_np = action_mask(model_state)
    mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
    
    # If all actions are masked, unmask the wait action 
    if not mask_tensor.any(): 
        mask_tensor[5] = True
    

    # Perform Tabular forward pass if lut

    if self.model_type == "lut":

        # A no-op for the default (train-independently-then-merge) TabularQAgent - it has
        # no refresh(). Only SharedTabularQAgent (opt-in, for a handful of live-synced
        # bots rather than many independent ones - see models/LUT.py) defines this, to
        # pull in whatever other bots have written to the shared table since we last looked.
        if hasattr(self.model, "refresh"):
            self.model.refresh()

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

        # `action` (and self.last_action, above) stay in canonical frame - that's
        # the frame the Q-table itself is indexed in, and train.py's TD update
        # needs to stay consistent with it. Only the string we hand back to the
        # game engine gets rotated back to a real-world direction.
        return state_symmetry.uncanonicalize_action(ACTIONS[action], rotation_k)

    # PPO actor-critic (MLP / CNN)
    with torch.set_grad_enabled(getattr(self, 'train', False)):
        logits, value = self.model(state_tensor) 
        
        # Check if there are any insane weight blow ups
        if torch.isnan(logits).any() or torch.isnan(value).any():
            print(f"\n[NaN DETECTED AT STEP {game_state['step']}] Model weights blew up!")
    
    # Make masked tiles super unappealing 
    # -1e9 for pytorch is the highest value 
    masked_logits = logits.masked_fill(~mask_tensor.unsqueeze(0), -1e9) 
    
    # Select action after forward pass
    if getattr(self, 'train', False):
        dist = Categorical(logits=masked_logits)
        action_idx = dist.sample() 
        log_prob = dist.log_prob(action_idx)
        
        # Store attributes for batch collection
        self.last_state = state_tensor.squeeze(0)
        self.last_action = action_idx.item()
        self.last_log_prob = log_prob.detach()
        self.last_value = value.detach().squeeze().cpu()
        self.last_mask = mask_tensor.cpu()
        
    else: 
        # Greedily select action 
        action_idx = torch.argmax(masked_logits, dim=-1)
    
    return ACTIONS[action_idx.item()]
    
    
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

   
def state_to_features(game_state: dict, model_type: str) -> torch.Tensor:
    """Extract features from game state based on agent's model type 
        :param game_state: A dictionary describing the current game board 
        :param model_type: Model type we need to extract features for 
        :return tensor containing extracted features
    """
    
    if game_state is None: 
        return torch.zeros(1)
    
    if model_type == "lut":
        # Determine hash key for location in the table
        # Extract features into a 1D vector
        feat_key = get_table_key(game_state)
        
        # Convert the state tuple hash into an integer index
        hash_idx = int(abs(feat_key))
        return torch.tensor(hash_idx, dtype=torch.long)
    
    
    elif model_type == "mlp":
        # Extract features into a 1D tensor
        features = extract_features_1D(game_state)
        return torch.tensor(features, dtype=torch.float32)
    

    elif model_type == "cnn":
        # Extract features into a 3D tensor
        features =  extract_features_3D(game_state)
        return torch.tensor(features, dtype=torch.float32)
    
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")
    

def features_to_tensor(features, model_type): 
    """Return table index tensors for LUT and float32 
        feature vectors for MLP/CNN"""

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    
    if model_type == "lut":
        return features.to(dtype=torch.long)

    else: 
        return features.to(dtype=torch.float32)    
    
    
        
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
    
                    
# Spatial direction map, shared by every direction-valued sub-feature below:
# (dx, dy) of the step towards a target/escape tile -> a small discrete index
DIR_MAP = {(0, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3, (-1, 0): 4}

# (feature name, radix) for every sub-feature packed into the LUT state key, in the
# order _pack_mixed_radix expects them. Keeping this list as the single source of
# truth means the table's required size (NUM_TABLE_STATES, below) can never drift
# out of sync with what get_table_key() actually produces.
#
# "danger" is 2-valued (safe / in-a-blast-radius) by default. Setting
# ELSENZ_BOMB_TIMER_KEY=1 swaps it for a 4-valued "danger_timer" bucket (see
# _bomb_timer_bucket) that also distinguishes *how soon* - e.g. "clears in 2 turns"
# vs "just landed" - which action_mask already uses to guarantee survival, but the
# table itself couldn't previously use to make better use of the safe options (LUT_AGENT.md
# "Design discussion" has the full rationale). This is opt-in and changes what a given
# integer key means, so get_shared_table_path() routes it to a different file
# (an "-bombtimer" suffix) automatically - an existing table trained without it is
# never read or written by a process running with it enabled, and vice versa.
STATE_FEATURE_RADIXES_BASE = [
    ("danger", 2),        # 0 = safe, 1 = standing in a blast radius
    ("escape_dir", 5),    # step direction towards safety (or "wait" if already safe/none found)
    ("target_dir", 5),    # step direction towards the nearest coin/crate/opponent (any kind)
    ("coin_dir", 5),      # step direction towards the nearest coin specifically
    ("coin_dist", 4),     # bucketed BFS distance to that coin: 0=none, 1=near, 2=mid, 3=far
    ("enemy_dir", 5),     # step direction towards the nearest opponent specifically
    ("enemy_dist", 4),    # bucketed BFS distance to that opponent: 0=none, 1=near, 2=mid, 3=far
]

STATE_FEATURE_RADIXES_BOMB_TIMER = [
    ("danger_timer", 4),  # 0 = safe, 1 = explodes next turn, 2 = 2-3 turns, 3 = 4+ turns
    ("escape_dir", 5),
    ("target_dir", 5),
    ("coin_dir", 5),
    ("coin_dist", 4),
    ("enemy_dir", 5),
    ("enemy_dist", 4),
]


def bomb_timer_key_enabled() -> bool:
    """Whether ELSENZ_BOMB_TIMER_KEY is set, i.e. get_table_key() should use the 4-valued
    danger_timer bucket instead of the default 2-valued danger flag. Read directly from the
    environment (same pattern as ELSENZ_MODEL_TYPE/ELSENZ_BEHAVIOR/ELSENZ_TABLE_NAME elsewhere
    in this file) so every call site - act(), train.py's canonical_state(), build_model() -
    agrees without having to thread an extra parameter through all of them.
    """
    return os.environ.get('ELSENZ_BOMB_TIMER_KEY', '0').strip().lower() in ('1', 'true', 'yes')


def get_state_feature_radixes() -> list:
    return STATE_FEATURE_RADIXES_BOMB_TIMER if bomb_timer_key_enabled() else STATE_FEATURE_RADIXES_BASE


def get_num_table_states() -> int:
    """Total number of distinct LUT states = product of every active sub-feature's radix.
    build_model() sizes the Q-table using this, so it always matches whatever
    get_table_key() can actually produce for the current ELSENZ_BOMB_TIMER_KEY setting.
    """
    n = 1
    for _, radix in get_state_feature_radixes():
        n *= radix
    return n


# Kept as the base-key size for any code/docs that still reference the constant directly;
# get_num_table_states() is what build_model() actually sizes the table with.
NUM_TABLE_STATES = 1
for _, _radix in STATE_FEATURE_RADIXES_BASE:
    NUM_TABLE_STATES *= _radix


def _pack_mixed_radix(values: list) -> int:
    """Combine [(value, radix), ...] sub-features into a single non-negative int key
    (Horner's method / mixed-radix encoding), so each distinct combination of
    sub-feature values maps to exactly one table row and every row is used."""

    key = 0
    for value, radix in values:
        key = key * radix + value
    return key


def _distance_bucket(distance) -> int:
    """Coarsely bucket a BFS distance so the table stays small: exact distance doesn't
    matter much for decision-making once a target is more than a few tiles away."""

    if distance is None or not np.isfinite(distance):
        return 0  # no reachable target of this kind
    if distance <= 2:
        return 1  # near: right next to it
    if distance <= 5:
        return 2  # mid: worth a detour
    return 3      # far: background awareness only


def _bomb_timer_bucket(timer) -> int:
    """Bucket the minimum bomb timer threatening the agent's own tile (0 = not
    threatened at all) into the 4-valued danger_timer feature used when
    ELSENZ_BOMB_TIMER_KEY is enabled. `timer` is spatial.build_bomb_timer_map()'s value at
    the agent's position: np.inf if no bomb/explosion threatens this tile, otherwise the
    smallest number of steps until the soonest one goes off.
    """

    if timer is None or not np.isfinite(timer):
        return 0  # safe
    if timer <= 1:
        return 1  # explodes next turn - most urgent
    if timer <= 3:
        return 2  # explodes in 2-3 turns
    return 3      # explodes in 4+ turns (still inside a blast radius, but time to work with)


def get_table_key(game_state: dict) -> int:
    """ Generates hashable key for the state representation for actor-critic lookup tables

    Packs danger/escape (as before) together with the direction+distance to the
    nearest coin and the direction+distance to the nearest opponent, tracked as
    separate sub-features (rather than folded into the single combined "target"
    feature) so the table can tell "a coin is 1 tile north" apart from "an enemy is
    1 tile north" instead of treating every kind of target the same way.

    Args:
        game_state (dict): the current game board

    Returns:
        int: hashable state key in [0, NUM_TABLE_STATES) # important difference in lut it is int, in elsenz-main it was tuple
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

    # coins_crates = coins + crates # combine coins and crates # source elsenz

    targets = coins + crates + opponents # combine all targets on the field # source lut

    # Extract danger info
    in_danger, (escape_dx, escape_dy) = spatial.find_escape_route(
        field=field,
        agent_pos=agent_pos,
        bombs=bombs,
        opponents=opponents_set,
        explosion_map=explosions
    )

    # # Extract coin crate info 
    # coin_crate_info = spatial.get_nearest_target(
    #     field=field, 
    #     agent_pos=agent_pos, 
    #     targets=coins_crates, 
    #     bomb_positions=bomb_positions, 
    #     opponents=opponents_set, 
    #     explosion_map=explosions)

    # Extract target info (any kind - coin, crate or opponent), same as before
    target_info = spatial.get_nearest_target(
        field=field,
        agent_pos=agent_pos,
        targets=targets,
        bomb_positions=bomb_positions,
        opponents=opponents_set,
        explosion_map=explosions)



    # if coin_crate_info is None or coin_crate_info[1] is None: 
    #     coin_crate_dx, coin_crate_dy = 0, 0
    
    # else:
    #     _, (coin_crate_dx, coin_crate_dy) = coin_crate_info
        

    if target_info is None or target_info[1] is None:
        target_dx, target_dy = 0, 0

    else:
        _, (target_dx, target_dy) = target_info

    # Coin-specific direction + distance
    coin_info = spatial.get_nearest_target(
        field=field,
        agent_pos=agent_pos,
        targets=coins,
        bomb_positions=bomb_positions,
        opponents=opponents_set,
        explosion_map=explosions) if coins else None

    if coin_info is None or coin_info[1] is None:
        coin_dx, coin_dy, coin_dist = 0, 0, None
    else:
        coin_dist, (coin_dx, coin_dy) = coin_info

    # Extract opponent info, note that here we don't want to treat opponents as obstacles
    # because we are actively looking for them  
    enemy_info = spatial.get_nearest_target(
        field=field,
        agent_pos=agent_pos,
        targets=opponents,
        bomb_positions=bomb_positions,
        opponents=opponents_set,
        explosion_map=explosions) if opponents else None

    if enemy_info is None or enemy_info[1] is None:
        enemy_dx, enemy_dy, enemy_dist = 0, 0, None
    else:
        enemy_dist, (enemy_dx, enemy_dy) = enemy_info

    f_escape = DIR_MAP.get((escape_dx, escape_dy), 0)         # cardinal directions + wait = 5 states
    f_target = DIR_MAP.get((target_dx, target_dy), 0)         # cardinal directions + wait = 5 states
    f_coin_dir = DIR_MAP.get((coin_dx, coin_dy), 0)           # cardinal directions + wait = 5 states
    f_coin_dist = _distance_bucket(coin_dist)                 # none/near/mid/far = 4 states
    f_enemy_dir = DIR_MAP.get((enemy_dx, enemy_dy), 0)        # cardinal directions + wait = 5 states
    f_enemy_dist = _distance_bucket(enemy_dist)               # none/near/mid/far = 4 states

    if bomb_timer_key_enabled():
        # How soon the bomb/explosion currently threatening this tile goes off, not just
        # whether one does - lets the table learn e.g. "this clears in 2 turns, hold
        # position" instead of only ever seeing a flat "in danger" bit. Reuses the same
        # timer map action_mask() already computes for exact escape-safety checks -
        # no new BFS needed.
        timer_map = spatial.build_bomb_timer_map(field, bombs)
        f_danger_feature = (_bomb_timer_bucket(timer_map[agent_pos[0], agent_pos[1]]), 4)
    else:
        f_danger_feature = (1 if in_danger else 0, 2)

    # Convert multidimensional features into one single key
    # return int(f_danger + (f_escape * 2) + (f_coin_crate  * 2 * 5) + (f_opponent * 2 * 5 * 5) )
    return _pack_mixed_radix([
        f_danger_feature,
        (f_escape, 5),
        (f_target, 5),
        (f_coin_dir, 5),
        (f_coin_dist, 4),
        (f_enemy_dir, 5),
        (f_enemy_dist, 4),
    ])
    
  
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