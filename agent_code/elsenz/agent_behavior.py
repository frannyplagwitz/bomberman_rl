import os 
import sys

BASE_REWARDS = {
    "peaceful": {
        'WAITED': -0.005,        # cost of 2 step penalties
        'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
        'MOVED_FROM_SAFETY': -0.15,
        'CRATE_DESTROYED': 0.5,  # 5 crates = 1 coin, interesting but not enough to avoid collecting coin
        'COIN_COLLECTED': 1.0,	 # given in instructions
        'KILLED_OPPONENT': 5.0,	 # given in instructions
        'BOMB_USEFUL': 0.1,      # guidance step, not as good as collecting a coin but better than destroying a crate, but want to make sure that agent places bombs near crates
        'BOMB_WASTEFUL': -0.3,    # guidance step, penalize the agent for using a bomb that doesn't destroy crate or kill opponent, but don't want to punish exploration t00 much
        'KILLED_SELF': -5.0,     # same penalty as being killed (will be added on top of being killed)
        'GOT_KILLED': -5.0,	     # opposite of killing opponent 
        'SURVIVED_ROUND': 0.0,	 # don't want to set this too high or else the agent won't take any risks, set to same value as killing opponent 
        'TRAPPED_SELF':  -5.0,   # same as being killed (will probably get killed)
        'TRAPPED_ENEMY': 1.0,	 # guidance step, as good as collecting a coin
        'INVALID_ACTION': -0.05, # guidance step, worse than moving/waiting, but want to make sure that agent isn't afraid of exploration 
        'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },
    
    "aggressive": {
        'WAITED': 	-0.005, 	    # cost of 2 step penalties
        'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
        'MOVED_FROM_SAFETY': -0.15,
        'CRATE_DESTROYED': 0.1,	# 0.2 / 3
        'COIN_COLLECTED': 0.5,	    # 1/2 coin collected reward because should be less interesting
        'KILLED_OPPONENT': 10.0,    # triple the peaceful agent reward
        'BOMB_USEFUL': 0.2,         # guidance step, not as good as collecting a coin but better than destroying a crate
        'BOMB_WASTEFUL': -0.5,    # guidance step, penalize the agent for using a bomb that doesn't destroy crate or kill opponent
        'KILLED_SELF': -8.0,	    # -8.0  = agent avoids killing itself but will do so if agent can take out an enemy
        'GOT_KILLED':	-3.0,	    # -5.0 - 3: aggressive requires risker behavior
        'SURVIVED_ROUND': 0.0,	    # Don't care about surviving the longest, just want to take out as many opponents as possible 
        'TRAPPED_SELF':  -5.0,      # Remains the same, as peaceful, if agent gets trapped can't kill more 
        'TRAPPED_ENEMY': 3.0,
        'INVALID_ACTION': -0.05,    # guidance step, worse than moving/waiting
        'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    }    
}


# Some notes: original weights: 
    #    "SURVIVE": 2.0,    # killed + kill self / 100 (want to make sure peaceful agent doesn't flee from everything)
     #   "EXPLORE": 0.01,    # collect coin / 100   (need to make sure that we don't explore forever)
     #   "COIN": 0.20,       # collect coin / 5
     #   "CRATE": 0.04,      # destroy crate / 5
      #  "KILL": 0.05 ,       # kill opponent / 100
       # "TRAP": 0.02,         # trap opponent / 100

# Dividing by 20 because currently escaping a bomb costs almost as much as collecting a coin, so it causes the agent to drop a 
# bomb just so it can escape it 


POTENTIAL_WEIGHTS = {
    "peaceful": {
        "SURVIVE": 1.0,     # killed + kill self / 100 (want to make sure peaceful agent doesn't flee from everything)
        "EXPLORE": 0.005,   # collect coin / 10  (need to make sure that we don't explore forever)
        "COIN": 0.08,       # collect coin / 5
        "CRATE": 0.03,      # destroy crate / 5
        "KILL": 0.01 ,      # kill opponent / 100
        "TRAP": 0.005,      # trap opponent / 100
    },
    
    "aggressive": {
        "SURVIVE": 0.1,    # killed + kill self / 300 (survival isn't as important to aggressive agent)
        "EXPLORE": 0.05,     # coin / 100 (need to make sure what we don't explore forever)
        "COIN": 0.3,       # coin / 20 (want to weight killing over coin collecting)
        "CRATE": 0.15,     # crate / 20
        "KILL": 0.8,       # kill opponent / 50 (prioritize killing)
        "TRAP": 0.3,        # trap opponent / 50 (prioritize trapping but not as much as killing )
    }
}


def apply_env_overrides(base_rewards, potential_weights):
    """Applies environment overrides calculated from Bayesian Optimization (BO). 
    
    Sets ELSENZ_<BEHAVIOR>_<KEY> environment variables before launching each trial's 
    training subprocess (ex. ELSENZ_PEACEFUL_BOMB_WASTEFUL = -0.42)
    
    Scans for any variables matching a real entry in POTENTIAL_WEIGHTS or BASE_REWARDS
    and overrides the value. 
    
    If run without Bayesian Optimization, the entries will remain unchanged

    """
    prefix = "ELSENZ_"
    
    # Keep list of keys that do not have behavior in the name
    NON_REWARD_KEYS = frozenset({
        "BEHAVIOR",            # ELSENZ_BEHAVIOR
        "EPISODES_PER_UPDATE", # ELSENZ_EPISODES_PER_UPDATE
        "MASK",                # ELSENZ_MASK
    })
    
    
    unmatched = []
    
    for env_key, raw_value in os.environ.items(): 
        
        # Ignore environment keys that don't start with agent prefix
        if not env_key.startswith(prefix):
            continue
        
        # Get the rest environment key, make sure it's not just the prefix
        rest_key = env_key[len(prefix):]
        
        # If key is in NON_REWARD_KEYS, don't care about it not having a behavior, ignore 
        if rest_key in NON_REWARD_KEYS: 
            continue 
        
        
        if "_" not in rest_key: 
            unmatched.append((env_key, raw_value, "expected ELSENZ_<BEHAVIOR>_<KEY> but no '_' found"))    
            continue 
        
        # Take out the behavior and key from the rest of the env key
        behavior, key = rest_key.split("_", 1)
        behavior = behavior.lower() 
        
        try: 
            value = float(raw_value)
        except ValueError:
            unmatched.append((env_key, raw_value, "value could not be parsed as a float"))
            continue 
        
        # Check if the passed behavior contains base rewards or potential weights
        # If so, feed in new optimized value
        if behavior in base_rewards and key in base_rewards[behavior]: 
            base_rewards[behavior][key] = value 
            
        elif behavior in potential_weights and key in potential_weights[behavior]:
            potential_weights[behavior][key] = value
        
        else: 
            unmatched.append((env_key, raw_value, f"Parsed as KEY={key!r}, BEHAVIOR={behavior!r}, not a real key+behavior pair"))    
    
    if unmatched: 
        known_behaviors = sorted(set(base_rewards) | set(potential_weights))
        known_keys = sorted(set().union(*[d.keys() for d in base_rewards.values()],
                                        *[d.keys() for d in potential_weights.values()]))
    
    
        print(f"\n[agent_behavior] WARNING: {len(unmatched)} ELSENZ_* environment "
              f"variable(s) did not match the expected ELSENZ_<BEHAVIOR>_<KEY> structure "
              f"and were IGNORED - training proceeded using the DEFAULT weight for each "
              f"of these instead of the intended override:", file=sys.stderr)
        
        
        for env_key, raw_value, reason in unmatched:
            print(f"  - {env_key}={raw_value!r}: {reason}", file=sys.stderr)
 
        print(f"  Known behaviors: {known_behaviors}", file=sys.stderr)
        print(f"  Known keys: {known_keys}", file=sys.stderr)
        print(f"  Expected format: ELSENZ_<BEHAVIOR>_<KEY>\n", file=sys.stderr)
 
    
    return base_rewards, potential_weights


BASE_REWARDS, POTENTIAL_WEIGHTS = apply_env_overrides(
    BASE_REWARDS, POTENTIAL_WEIGHTS)
            
        
        
        





