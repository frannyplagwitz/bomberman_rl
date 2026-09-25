from collections import deque
import numpy as np 
import settings as s


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def get_directions():
    # Get directions based on instruction actions: 
    # ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    return DIRECTIONS


def get_bomb_positions(bombs: list) -> set[tuple[int, int]]:
    """Get the bomb positions from bomb list 

    Args:
        bombs (list): list of bombs on the field 
            bomb_pos: bomb positions on field 
            timer: steps till bomb explodes
    Returns:
        set[tuple[int, int]]: positions of bombs on the field
    """
    return {bomb[0] for bomb in bombs} if bombs else set()
 
 
def is_tile_open(
    field: np.ndarray, 
    tile_pos: tuple[int, int], 
    bomb_positions: set[tuple[int, int]] = None, 
    opponents: set[tuple[int, int]] = None, 
    explosion_map: np.ndarray = None) -> bool: 
    
    """Check if the agent can step onto a tile"""

    tx, ty = tile_pos 
    
    # out of bounds 
    if not (0 <= tx < field.shape[0] and 0 <= ty < field.shape[1]):
        return False 
    
    # tile is a crate or a wall 
    if field[tx, ty] != 0: 
        return False
    
    # tile contains a bomb 
    if bomb_positions and tile_pos in bomb_positions: 
        return False
    
    # tile contains an opponent
    if opponents and tile_pos in opponents: 
        return False 
    
    # If tile contains an explosion 
    if explosion_map is not None and explosion_map[tx, ty] > 0: 
        return False 
    
    return True 


def get_danger_tiles(field: np.ndarray, bomb_pos: tuple[int, int]) -> set[tuple[int, int]]:
                    
    """ Obtains the tiles that a bomb endangers
        
    Args:
    
        field (np.ndarray): 2D array of the scene, each tuple contains a value corresponding to the scene layout
            -1: An indestructable/inaccessible tile (wall, pillar)
            0: A free tile
            1: A tile with a crate
            
        bomb_positions (set[tuple[int,int]): Set of bomb positions on the field 
                  
    Returns:
        set[tuple[int, int]]: Set of danger tile positions on the field that a particular bomb affects
    """
    danger_tiles = {bomb_pos}
    bx, by = bomb_pos
    
    for dx, dy in DIRECTIONS:
        for step in range(1, s.BOMB_POWER + 1):
            nx, ny = bx + (dx * step), by + (dy * step)


            # Check if inside field boundary
            if not (0 <= nx < field.shape[0] and 
                    0 <= ny < field.shape[1]):
                break 
            
            # Check the value of the tile 
            tile_value = field[nx, ny]
        
            # If wall, will block explosions so ignore rest of direction 
            if tile_value == -1:
                    break
        
            # Will be either free or a crate, so will be marked as dangerous 
            danger_tiles.add((nx, ny))
            
            # If crate, mark as dangarous but ignore rest of direction 
            if tile_value == 1:
                break
    
    return danger_tiles
    

def build_danger_map(field: np.ndarray, bombs: list, explosion_map: np.ndarray = None) -> set[tuple[int, int]]:
                    
    """ Builds map of total danger tiles consisting of projected explosion tiles 
         from active bombs and active explosions
        
    Args:
        field (np.ndarray): 2D array of the scene, each tuple contains a value corresponding to the scene layout
            -1: An indestructable/inaccessible tile (wall, pillar)
            0: A free tile
            1: A tile with a crate
            
        bombs (list): list of bombs on the field 
            bomb_pos: bomb positions on field 
            timer: steps till bomb explodes
                
        explosion_map (np.ndarray): 2D array of the scene where each 
                                    tuple contains a value corresponding to explosion. Defaults to None
            0: Tile is not exploding 
            > 0: Tile is exploding 
             
    Returns:
        set[tuple[int, int]]: Set of danger tile positions on the field
    """
    danger_tiles = set()
    
    if explosion_map is not None and np.any(explosion_map > 0): 
        danger_tiles.update(map(tuple, np.argwhere(explosion_map > 0)))
            
    if bombs: 
        for b in bombs:
            
            # Unpack b to get position 
            curr = b 
            while (isinstance(curr, (tuple, list, np.ndarray)) and 
                   len(curr) > 0 and isinstance(curr[0], (tuple, list, np.ndarray))):
                curr = curr[0]
                
            # Extract (x, y) from coordinate pairs 
            if (isinstance(curr, (tuple, list, np.ndarray)) and 
                len(curr) >= 2): 
                
                x, y = int(curr[0]), int(curr[1])
                danger_tiles.update(get_danger_tiles(field, (x, y)))
            
    return danger_tiles

def build_bomb_timer_map(field: np.ndarray, bombs: list )  -> np.ndarray:
    """ Creates 2D map the size of the field where each tile contains the 
        timer nearest to finishing of the bombs affecting it
    
    Args: 
        field (np.ndarray): 2D array of the scene, each tuple contains a value corresponding to the scene layout
            -1: An indestructable/inaccessible tile (wall, pillar)
            0: A free tile
            1: A tile with a crate
            
        bombs (list): list of bombs on the field 
            bomb_pos: bomb positions on field 
            timer: steps till bomb explodes
                        
    Returns:
        np.ndarray: 2D map the size of the field described above
     
    """
    # Build map the size of timer where each field is filled with inf 
    timer_map = np.full(field.shape, fill_value=np.inf, dtype=np.float32)
    
    # If no bombs, just return the map as is 
    if not bombs: 
        return timer_map
    
    # Look through bombs and find each tile they affect
    for b in bombs: 
        if isinstance(b, (tuple, list)) and len(b) == 2: 
            (bx, by), timer = b[0], b[1]
            for tx, ty in get_danger_tiles(field, (bx, by)):
                timer_map[tx, ty] = min(timer_map[tx, ty], float(timer))

    return timer_map


# Entity Tracking Functions (Agent / Opponent)

def evaluate_safety(field: np.ndarray, entity_pos: tuple[int, int], 
                    danger_tiles: set[tuple[int, int]], bomb_positions: set[tuple[int, int]],
                    opponents: set[tuple[int, int]], explosion_map: np.ndarray = None, 
                    bombs: list = None) -> tuple[float, tuple[int, int], bool]:
    
    """ Evaluates the path to safety for any moving entity (agent or enemy)
    
    Args: 
        field (np.ndarray): 2D array of the scene, each tuple contains a 
                            value corresponding to the scene layout
                            
            -1: An indestructable/inaccessible tile (wall, pillar)
             0: A free tile
             1: A tile with a crate
            
        entity_pos (tuple[int, int]): position of a moving entity on the field (agent or enemy)
        danger_tiles (set[tuple[int, int]]): set of dangerous tiles on the field                 
        
        bomb_positions (list[tuple[int,int]): list of bomb positions on the field 
        opponents (list[tuple[int, int]]): list of opponents on the field
        
        explosion_map (np.ndarray): 2D array of the scene where each tuple contains a 
                                    value corresponding to explosion. Defaults to None 
            0: Tile is not exploding 
            > 0: Tile is exploding 
    
    Returns:
        tuple[float, tuple[int, int], bool]: tuple containing information about the entity's safety 
                                             (steps, dir, is_trapped)
                                             
            steps (float): steps to nearest safe tile 
            dir (tuple[int, int]): direction the entity needs to move 
            is_trapped (bool): is the entity trapped and can't escape 
    
    """
    
    # If the entity is not in a blast zone, it is currently safe, doesn't need to escape 
    if entity_pos not in danger_tiles: 
        # Check if there's open neighbors
        open_neighbors = sum(
            1 for dx, dy in DIRECTIONS
            if is_tile_open(
                field=field, 
                tile_pos=(entity_pos[0] + dx, entity_pos[1] + dy), 
                bomb_positions=bomb_positions, 
                opponents=opponents, 
                explosion_map=explosion_map)
            
            and not tile_blocked_by_opponent(
                tile=(entity_pos[0] + dx, entity_pos[1] + dy), opponents=opponents, steps=1))
        
        return 0.0, (0, 0), (open_neighbors == 0) # if no open neighbors, trapped but safe

    # Map lowest bomb timer per tile 
    timer_map = build_bomb_timer_map(field, bombs) if bombs else None
    
    # Find the nearest non-danger tile via BFS
    queue = deque([(entity_pos, 0, (0,0))])
    visited = {entity_pos}
    
    while queue: 
        (x, y), dist, first_step = queue.popleft()
        
        # If the steps to reach  the tile exceed the time left on a bomb, path bad 
        if timer_map is not None and timer_map[x, y] != np.inf: 
            if dist >= timer_map[x, y]:
                continue 
            
        # Look at the neighbors in the danger zone
        for dx, dy in DIRECTIONS: 
            neighbor = (x + dx, y + dy)
            
            # Determine if neighbor tile is currently open, not blocked by opponent and not visited
            if (is_tile_open(field=field, tile_pos=neighbor, 
                             bomb_positions=bomb_positions, 
                             opponents=opponents, 
                             explosion_map=explosion_map) 
                
                and not tile_blocked_by_opponent(tile=neighbor, opponents=opponents, 
                                                 steps=(dist + 1))
                and neighbor not in visited):
                
                step = (dx, dy) if dist == 0 else first_step # find step direction 
                    
                # If the neighbor is not in danger, its the nearest point of escape + not trapped
                if neighbor not in danger_tiles: 
                    return float(dist + 1), step, False
            
                # otherwise, continuing looking 
                visited.add(neighbor)
                queue.append((neighbor, dist + 1, step))
            
    # Reached the end of the queue and no viable escape route found, trapped 
    return float('inf'), (0, 0), True


def is_bomb_useful(field: np.ndarray, agent_pos: tuple[int, int], 
                   opponents: list[tuple[int, int]] = None) -> bool: 
    
    """ Check if dropping a bomb is a useful action (destroys crate/ endangers opponent)

    Args:
        field (np.ndarray): field (np.ndarray): 2D array of the scene where each tuple 
                            contains a value corresponding to scene layout: 
                            
            -1: An indestructable/inaccessible obstacle (wall, pillar)
             0: An open tile 
             1: A destructable obstacle (crate)
             
        agent_pos (tuple[int, int]): Position of the agent on the field
        opponents (set[tuple[int, int]]): Set of opponents on the field 
        
    Returns:
        bool: True if the bomb was useful, false otherwise 
    """
    bx, by = agent_pos
    opponents_set = opponents or set()
    
    if (bx, by) in opponents_set: 
        return True
    
    for dx, dy in DIRECTIONS: 
        for step in range(1, s.BOMB_POWER + 1):
            nx, ny = bx + (dx * step), by + (dy * step)
            
            # Check if within field bounds
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            
            tile_value = field[nx, ny]
            
            if tile_value == -1: 
                break
            
            # Will destroy a crate or there's an opponent in the blast radius 
            if tile_value == 1 or (nx, ny) in opponents_set:
                return True
            
    return False


def get_nearest_target(field: np.ndarray,  agent_pos: tuple[int, int], 
                       targets: list[tuple[int, int]],  
                       bomb_positions: set[tuple[int, int]] = None, 
                       opponents: set[tuple[int, int]] = None, 
                       explosion_map: np.ndarray = None) -> tuple[float, tuple[int, int]]:
    
    """ Finds the nearest target using BFS
    
    Args: 
    
        field (np.ndarray): 2D array of the scene where each tuple 
                            contains a value corresponding to scene layout: 
                            
            -1: An indestructable/inaccessible obstacle (wall, pillar)
             0: An open tile 
             1: A destructable obstacle (crate)
             
        agent_pos (tuple[int, int]): Position of the agent on the field
        targets (list[tuple[int, int]]): List of targets on the field (coins, crates, enemies)
        
        opponents (set[tuple[int, int]]): Set of opponents on the field 
        bomb_positions (set[tuple[int, int]]): Set of bomb positions
       
        explosion_map (np.ndarray): 2D array of the scene where each tuple 
                                    contains a value corresponding to explosion
            0: Tile is not exploding 
            > 0: Tile is exploding 
            
    Returns:
        tuple[float, tuple[int, int]]: Information considering the nearest target (steps, direction)
            steps (float): steps to the target 
            direction: tuple[int, int]: direction needed to move in 
    """
    
 
    # If we currenly don't have any targets, will return the max distance 
    # Walking distance width = 17 - (2 wall tiles) = 15
    # Walking distance height = 17 - (2 wall tiles) = 15
    # Theoretical total distance = 15 
    if not targets:
        return float('inf'), None
        
        
    # If the agent is already on top of the objective, return distance of 0
    target_set = set(targets)
    if agent_pos in target_set: 
        return 0.0, (0, 0)
               
    # Store position, distance and direction of first step into queue 
    queue = deque([(agent_pos, 0, (0, 0))])
    visited = {agent_pos}
        
    # Move through queue, pop out contents
    while queue: 
        (x, y), tot_steps, first_step = queue.popleft()
            
        # If the position is in the target set, 
        # immediately return the distance and direction step
        if (x, y) in target_set:
            return float(tot_steps), first_step
            
        # Look at each surrounding tile 
        for dx, dy in DIRECTIONS: 
            neighbor = (x + dx, y + dy)
            
            if neighbor in visited: 
                continue 
            
            step = (dx, dy) if tot_steps == 0 else first_step
            
            # If neighbor is a crate, being next to it is the same as reaching it
            if ( 0 <= neighbor[0] < field.shape[0] and 0 <= neighbor[1] < field.shape[1] and
                field[neighbor] == 1):
                if neighbor in target_set: 
                    return float(tot_steps + 1), step
                
                continue # blocked by crate
            
            # Add neighbor to visited tiles  
            if is_tile_open(field, neighbor, bomb_positions, opponents, explosion_map):

                visited.add(neighbor)
                step = (dx, dy) if tot_steps == 0 else first_step 
                queue.append((neighbor, tot_steps + 1, step))
                        
    return float('inf'), None


def find_escape_route(field: np.ndarray, agent_pos: tuple[int, int], 
                      bombs: list, opponents: list,
                      explosion_map: np.ndarray = None) -> tuple[bool, tuple[int, int]]:
    
    """ Based on spatial data evaluates if the agent is in danger and calculates immediate step to safety
    
    Args:
     
        field (np.ndarray): 2D array of the current environment, 
                            each tuple contains value of element on the tile: 
                            
                -1: Indestructable/inaccessible tile (wall/pillar)
                 0: Free tile 
                 1: Tile with a crate
                 
        agent_pos (tuple[int, int]): position of the agent in the field 
                   
        bombs (list): Bombs on the field
            Tuple contents: 
                bomb_pos: coordinates of the bomb on the field 
                timer: Steps to explosion, max is determined by settings.MAX_TIMER
  
        opponents (list): list of opponents on the field 
                   
        explosion_map (np.ndarray): 2D array of the scene where each 
                                    tuple contains a value corresponding to explosion.
            Defaults to None
            
            0: Tile is not exploding 
            > 0: Tile is exploding 
            
    Returns:
        tuple[bool, tuple[int, int]]: tuple containing escape route information (in_danger, steps)
            in_danger (bool): is the agent currently in danger 
            steps (tuple[int, int]): steps required to reach safety 
    """
    
    # Find danger zones across the map 
    bomb_positions = get_bomb_positions(bombs)
    danger_tiles   = build_danger_map(field, bomb_positions, explosion_map)
    
    # If the agent isn't in danger tiles, not in danger
    if agent_pos not in danger_tiles: 
        return False, (0, 0)
    
    opponents_set = set(opponents) if opponents else set()
        
    # Evaluate safety route and return results 
    _, safe_step, _ = evaluate_safety(
        field=field, 
        entity_pos=agent_pos,
        danger_tiles=danger_tiles, 
        bomb_positions=bomb_positions,
        opponents=opponents_set,
        explosion_map=explosion_map,
        bombs=bombs)
                         
    return True, safe_step


# Tile specific functions

def extract_spatial_features(game_state) -> np.ndarray: 
    
    """Extracts the following spatial scalars: 
        1. in_danger (if the agnet is currently in danger or not. 1.0 if in danger, 0.0 otherwise)
        2. escape_dir (Categorical float) 0=UP 1=RIGHT 2=DOWN 3=LEFT 4=WAIT
        3. can_place_bomb_safe: 1.0 if possible, 0.0 otherwise
        4. steps_to_safety: Normalized float (0.0 to 1.0)
        
    Args: 
           game_state: current state of the game
    
    Returns: Extracted spatial features
    """
    
    # Extract values from game state
    field = game_state['field']
    agent_pos = game_state['self'][3]
    
    bombs = game_state['bombs']
    opponents = [other[3] for other in game_state['others']]
    explosion_map = game_state.get('explosion_map', None) 
    
    danger_tiles = build_danger_map(field, bombs, explosion_map)
    bomb_positions = get_bomb_positions(bombs)
    initial_bomb_positions = bomb_positions - {agent_pos}
    
    opponents_set = set(opponents) if opponents else set() 
    
    # 1. Check danger 
    in_danger = 1.0 if agent_pos in danger_tiles else 0.0
    
    # 2. Check steps to safety and escape direction 
    steps_to_safety_norm = 0.0
    escape_dir_idx = 4 # WAIT by default 
     
     
    steps, safe_step, _ = evaluate_safety(field, agent_pos, danger_tiles, 
                                          initial_bomb_positions, 
                                          opponents_set, 
                                          explosion_map, bombs)

    # If danger, normalize the steps to safety (max is is bomb range + 1)
    if in_danger == 1.0: 
        steps_to_safety_norm = min(steps, 5.0) / 5.0 if steps != float('inf') else 1.0
        
        # Map (dx, dy) to direction index 
        directions = DIRECTIONS
        if safe_step in directions: 
            escape_dir_idx = directions.index(safe_step)


    # Convert escape dir to one-hot encoding
    escape_one_hot = np.zeros(5, dtype=np.float32)
    escape_one_hot[escape_dir_idx] = 1.0

    # Check if we can place a safe bomb (useful + doesn't trap agent)
    can_place_safe_bomb = 0.0
    has_bomb = game_state['self'][2]
    
    if not in_danger and has_bomb and  is_bomb_useful(field, agent_pos, opponents):
        # Simulate dropping a bomb to check if it will do more harm than good
        sim_bombs = list(bombs) + [(agent_pos, s.BOMB_TIMER)]
        sim_danger = build_danger_map(field, sim_bombs, explosion_map)
        sim_positions = bomb_positions.union({agent_pos}) # Block out current tile


        _, _, sim_trapped = evaluate_safety(
            field, agent_pos, sim_danger, sim_positions,
            opponents_set, explosion_map, sim_bombs)

        if not sim_trapped: 
            can_place_safe_bomb = 1.0
                
    # Return as 1D feature array
    return np.concatenate([
        np.array([in_danger, can_place_safe_bomb, steps_to_safety_norm], 
                 dtype=np.float32), escape_one_hot])
  

def is_entity_trapped(field: np.ndarray, 
                      entity_pos: tuple[int, int], 
                      bombs: list, opponents: list,
                      explosion_map: np.ndarray = None) -> bool:
    """
    Checks if a given entity (agent or opponent) is currently trapped in the field
    Args: 
        field (np.ndarray): 2D array of the current environment, 
                            each tuple contains value of element on the tile: 
                            
            -1: Indestructable/inaccessible tile (wall/pillar)
             0: Free tile 
             1: Tile with a crate
        
        entity_pos (tuple[int, int]): position of an entity (agent or opponent) on the field
                
        bombs (list): list of bombs on the field 
            Tuple Contents: 
                bomb_pos: bomb position 
                timer: Steps in timer, max is determined by settings.MAX_TIMER
            
        opponents(list): list of opponents (to the entity) on the field 
             
        explosion_map (np.ndarray): 2D array of the scene where each tuple 
                                    contains a value corresponding to explosion. 
            Defaults to None
            
            0: Tile is not exploding 
            > 0: Tile is exploding 
            
            
    Returns:
        bool: True if given enemy is trapped, False otherwise
    """
    # Find danger tiles and positions for the active bombs
    danger_tiles = build_danger_map(field, bombs, explosion_map)
    set_opponents = set(opponents)
    
    bomb_positions = get_bomb_positions(bombs)
    
    # Check that no bombs adn that the entity isn't on a danger tile 
    if entity_pos not in danger_tiles and not bomb_positions: 
        return False
    
    
    # Evaluate safety 
    _, _, is_trapped = evaluate_safety(field, entity_pos, danger_tiles, 
                                       bomb_positions, 
                                       set_opponents, 
                                       explosion_map, bombs)

    return is_trapped


# Agent Specific Functions 
# --------------------------
        
def get_distance_nearest_safety(field: np.ndarray, agent_pos: tuple[int, int], 
                                bombs: list, opponents: list = None, 
                                explosion_map: np.ndarray = None) -> float: 
    
    """Get distance to nearest safe tile 
    
    Args:
        
        field (np.ndarray): 2D array of the current environment, each tuple contains value of
                            element on the tile: 
                            
            -1: Indestructable/inaccessible tile (wall/pillar)
             0: Free tile 
             1: Tile with a crate
        
        agent_pos (tuple[int, int]): position of the agent on the field
        
        bombs (list): list of bombs on the field 
            Tuple Contents: 
                bomb_pos: bomb position 
                timer: Steps in timer, max is determined by settings.MAX_TIMER
            
        opponents(list): list of opponents on the field 
             
        explosion_map (np.ndarray): 2D array of the scene where each tuple contains a 
                                    value corresponding to explosion. 
            Defaults to None
            
            0: Tile is not exploding 
            > 0: Tile is exploding 

    Returns:
        distance to nearest safe tile
    """
    
    bomb_positions = get_bomb_positions(bombs)
    danger_tiles   = build_danger_map(field, bombs, explosion_map)
    opponents_set  = set(opponents) if opponents else set() 
    
    steps, _, _ = evaluate_safety(
        field, agent_pos, danger_tiles, 
        bomb_positions, opponents_set,
        explosion_map, bombs)
    
    return steps 
 
 
def tile_blocked_by_opponent(tile: tuple[int, int], opponents: set[tuple[int, int]], steps: int) -> bool: 
    """Checks if tile is blocked/can be blocked by an opponent
    Will be blocked if opponent could walk there within defined steps
    
    Args: 
           tile (tuple[int, int]): tile on field checking
           opponents(set[tuple[int, int]]): list of opponents (to the entity) on the field 
           steps: total steps it takes for agent to get to that tile 
           
    Returns: true if tile is blocked/can be blocked by an opponent, false otherwise
    """
    
    if not opponents: 
        return False
    
    tx, ty = tile 
    return any(abs(tx - ox) + abs(ty - oy) <= steps for ox, oy in opponents)
