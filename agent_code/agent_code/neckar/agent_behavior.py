import os 
import sys

BASE_REWARDS = {
    "peaceful": {
        'WAITED': -0.005,        # cost of 2 step penalties
        'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
        'MOVED_FROM_SAFETY': -0.15,
        'CRATE_DESTROYED': 0.5,  # 5 crates = 1 coin, interesting but not enough to avoid collecting coin # for aggr it was 0.2
        'COIN_COLLECTED': 1.0,	 # given in instructions
        'KILLED_OPPONENT': 5.0,	 # given in instructions
        'WON_ROUND': 0.0,        # peaceful doesn't chase highest score - see train.py's WON_ROUND synthesis
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

    # Basic values
    "aggressive_1": {
        'WAITED': 	-0.005, 	    # cost of 2 step penalties
        'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
        'MOVED_FROM_SAFETY': -0.15,
        'CRATE_DESTROYED': 0.1,	# 0.2 / 3
        'COIN_COLLECTED': 0.5,	    # 1/2 coin collected reward because should be less interesting
        'KILLED_OPPONENT': 10.0,    # triple the peaceful agent reward
        'WON_ROUND': 4.0,          # synthetic terminal bonus - see train.py's WON_ROUND synthesis
        'BOMB_USEFUL': 0.2,         # guidance step, not as good as collecting a coin but better than destroying a crate
        'BOMB_WASTEFUL': -0.15,    # lower than peaceful (-0.3): bomb spam must be cheaper than peaceful or the agent
                                    # never bombs enough to stumble into a kill (observed 0 kills / ~0 bombs in logs)
        'KILLED_SELF': -3.0,	    # lower than peaceful (-5.0): aggressive requires riskier behavior, so dying to your
                                    # own bomb should sting less here, not more (was -8.0, worse than peaceful - bug)
        'GOT_KILLED':	-3.0,	    # -5.0 - 3: aggressive requires risker behavior
        'SURVIVED_ROUND': 0.0,	    # Don't care about surviving the longest, just want to take out as many opponents as possible 
        'TRAPPED_SELF':  -5.0,      # Remains the same, as peaceful, if agent gets trapped can't kill more 
        'TRAPPED_ENEMY': 3.0,
        'INVALID_ACTION': -0.05,    # guidance step, worse than moving/waiting
        'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },

    # More impact on opponents
    "aggressive_2": {
            'WAITED': 	-0.005, 	    # cost of 2 step penalties
            'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
            'MOVED_FROM_SAFETY': -0.15,
            'CRATE_DESTROYED': 0.1,	# 0.2 / 3
            'COIN_COLLECTED': 0.5,	    # 1/2 coin collected reward because should be less interesting
            'KILLED_OPPONENT': 15.0,    # triple the peaceful agent reward
            'WON_ROUND': 5.0,          # synthetic terminal bonus - see train.py's WON_ROUND synthesis
            'BOMB_USEFUL': 0.4,         # guidance step, not as good as collecting a coin but better than destroying a crate
            'BOMB_WASTEFUL': -0.1,    # guidance step, penalize the agent for using a bomb that doesn't destroy crate or kill opponent
            'KILLED_SELF': -4.0,	    # -8.0  = agent avoids killing itself but will do so if agent can take out an enemy
            'GOT_KILLED':	-6.0,	    # -5.0 - 3: aggressive requires risker behavior
            'SURVIVED_ROUND': 0.05,	    # Don't care about surviving the longest, just want to take out as many opponents as possible 
            'TRAPPED_SELF':  -7.0,      # Remains the same, as peaceful, if agent gets trapped can't kill more 
            'TRAPPED_ENEMY': 5.0,
            'INVALID_ACTION': -0.05,    # guidance step, worse than moving/waiting
            'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },

    # Focus only on opponents
    "aggressive_3": {
            'WAITED': 	-0.005, 	    # cost of 2 step penalties
            'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
            'MOVED_FROM_SAFETY': -0.15,
            'CRATE_DESTROYED': 0.05,	# 0.2 / 3
            'COIN_COLLECTED': 0.5,	    # 1/2 coin collected reward because should be less interesting
            'KILLED_OPPONENT': 30.0,    # triple the peaceful agent reward
            'WON_ROUND': 6.0,          # synthetic terminal bonus - see train.py's WON_ROUND synthesis
            'BOMB_USEFUL': 0.4,         # guidance step, not as good as collecting a coin but better than destroying a crate
            'BOMB_WASTEFUL': -0.75,    # guidance step, penalize the agent for using a bomb that doesn't destroy crate or kill opponent
            'KILLED_SELF': -4.0,	    # -8.0  = agent avoids killing itself but will do so if agent can take out an enemy
            'GOT_KILLED':	-2.0,	    # -5.0 - 3: aggressive requires risker behavior
            'SURVIVED_ROUND': 0.50,	    # Don't care about surviving the longest, just want to take out as many opponents as possible 
            'TRAPPED_SELF':  -5.0,      # Remains the same, as peaceful, if agent gets trapped can't kill more 
            'TRAPPED_ENEMY': 10.0,
            'INVALID_ACTION': -0.05,    # guidance step, worse than moving/waiting
            'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },
    # aggressive_4: repurposed (see POTENTIAL_WEIGHTS below) into an explicit "explore more,
    # care less about surviving" variant, per the finding that aggressive_1/2/3 all converged
    # to corner-camping - EXPLORE's per-tile shaping bonus (see rl_heuristics.py's
    # compute_potential, w_explore * visited_fraction) was too small (0.03-0.05) to compete
    # with the risk of leaving a safe area, and the death penalties (KILLED_SELF/GOT_KILLED)
    # were harsh enough relative to a near-zero real kill rate that risk-aversion was the
    # rational optimum. Softened both here, and added a WAITED penalty double the others'
    # (-0.01 vs -0.005) as a direct anti-idling lever, since idling-in-place is exactly what
    # "camping" looks like at the per-step level.
    # Enemy-focus adjustment: KILL/TRAP/KILLED_OPPONENT are already at the family's max and
    # the reward clip (+/-15, rl_heuristics.py's compute_reward) is NOT being raised - raising
    # it risks reward-scale instability in PPO (huge spikes make the value function harder to
    # fit), and KILLED_OPPONENT already saturates it for aggressive_3 (30.0 alone > 15). So
    # instead every *other* reward here is cut ~20% (WON_ROUND, CRATE_DESTROYED,
    # COIN_COLLECTED, BOMB_USEFUL, EXPLORE, COIN, CRATE potential weights) - this doesn't
    # change the absolute size of the kill incentive, but makes it a proportionally larger
    # share of the reward the agent actually receives, without touching the clip or the
    # kill-related terms themselves. Penalties (WAITED, BOMB_WASTEFUL, death penalties,
    # MOVED_FROM_SAFETY) are left alone - they discourage bad behavior and don't compete
    # with the kill incentive the way positive non-kill rewards do.
    "aggressive_4": {
            'WAITED': 	-0.01, 	    # doubled vs the rest of the family - direct anti-camping/anti-idling lever
            'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee (only fires while actively in a bomb's blast radius - see train.py, not a general "stay safe" bias)
            'MOVED_FROM_SAFETY': -0.15,
            'CRATE_DESTROYED': 0.24,	# down from 0.3 (~20%) - enemy-focus: shrink non-kill rewards, not the kill ones
            'COIN_COLLECTED': 0.64,	    # down from 0.8 (~20%), same reasoning
            'KILLED_OPPONENT': 10.0,    # unchanged - kill-related, not touched by the enemy-focus reduction
            'WON_ROUND': 4.0,          # down from 5.0 (~20%) - most observed wins are by elimination, not combat (see below), so leave room for KILL to dominate
            'BOMB_USEFUL': 0.24,         # down from 0.3 (~20%)
            'BOMB_WASTEFUL': -0.2,    # softened from -0.5 - don't discourage bombing while exploring new crate-rich areas
            'KILLED_SELF': -3.0,	    # softened from -8.0 - less risk-averse, matching "less survival"
            'GOT_KILLED':	-2.0,	    # softened from -3.0, same reasoning
            'SURVIVED_ROUND': 0.0,	    # Don't care about surviving the longest, just want to take out as many opponents as possible
            'TRAPPED_SELF':  -3.0,      # softened from -5.0, consistent with lower risk-aversion
            'TRAPPED_ENEMY': 3.0,
            'INVALID_ACTION': -0.05,    # guidance step, worse than moving/waiting
            'STEP_PENALTY': -0.0025,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },

    # aggressive_5: radical experiment, per the enemy-focus curriculum result above (zero
    # kills for aggressive_1/3/4 even against a stationary opponent that never fights back -
    # the agent simply had no reason to engage, since every other reward source was already
    # enough to "win"). Every reward here is zero EXCEPT KILLED_OPPONENT/TRAPPED_ENEMY (below)
    # and the KILL/TRAP potential weights (POTENTIAL_WEIGHTS) - literally nothing else this
    # agent does is worth anything, not surviving, not coins, not crates, not even avoiding
    # its own death. If this doesn't produce a kill-seeking policy, the reward function isn't
    # the bottleneck and something more structural (state representation, training exposure,
    # network capacity) is.
    "aggressive_5": {
            'WAITED': 0.0,
            'MOVED_TO_SAFETY': 0.0,
            'MOVED_FROM_SAFETY': 0.0,
            'CRATE_DESTROYED': 0.0,
            'COIN_COLLECTED': 0.0,
            'KILLED_OPPONENT': 15.0,    # exactly the reward clip ceiling (rl_heuristics.py's +/-15 clip) - higher would be wasted, see the clip finding above
            'WON_ROUND': 0.0,          # not about outscoring anymore - purely combat
            'BOMB_USEFUL': 0.0,
            'BOMB_WASTEFUL': 0.0,
            'KILLED_SELF': 0.0,        # not punished, but also not rewarded - ending the episode early already costs it future chances at KILL/TRAP potential and the kill bonus itself, so an explicit penalty isn't needed to disincentivize it
            'GOT_KILLED': 0.0,         # same reasoning
            'SURVIVED_ROUND': 0.0,
            'TRAPPED_SELF': 0.0,
            'TRAPPED_ENEMY': 8.0,       # meaningful but below KILLED_OPPONENT - a trap is a means to a kill, not equivalent to one
            'INVALID_ACTION': 0.0,
            'STEP_PENALTY': 0.0,
    },

    # aggressive_6: less radical than aggressive_5, targeting the specific risk flagged there -
    # bombing itself carried zero incentive, only proximity did, so "press BOMB near the
    # opponent" had nothing teaching it directly. Keeps coin/crate rewards alive but very low
    # (just enough that bombing crates is still occasionally worth doing, which doubles as
    # practice for bomb placement/timing near a target) and gives BOMB_USEFUL a genuine
    # mid-strength incentive - the direct fix for aggressive_5's known gap. Also reintroduces
    # a real KILLED_SELF punishment (aggressive_5 left it at 0.0 deliberately) to push back
    # against reckless self-destructive bombing once bombing itself has an upside again.
    "aggressive_6": {
            'WAITED': -0.005,        # standard family penalty
            'MOVED_TO_SAFETY': 0.15,  # standard guidance step
            'MOVED_FROM_SAFETY': -0.15,
            'CRATE_DESTROYED': 0.03,   # very low - present so bombing crates still happens occasionally (bomb-placement practice), but not a competing goal
            'COIN_COLLECTED': 0.1,     # very low - lowest in the family, present but clearly not the point
            'KILLED_OPPONENT': 20.0,    # high, kill-focused - already past the reward clip ceiling either way (see clip finding)
            'WON_ROUND': 3.0,          # moderate - with coin/crate this low, outscoring is much harder to stumble into by accident than it was for aggressive_1-4, so this doesn't reopen the "win without fighting" loophole as easily
            'BOMB_USEFUL': 0.3,        # middle strength - the direct fix for aggressive_5's "bombing has no incentive" gap
            'BOMB_WASTEFUL': -0.3,     # moderate penalty, discourages random bomb-spam without discouraging bombing near a target
            'KILLED_SELF': -6.0,       # a real punishment (aggressive_5 left this at 0.0) - meaningful now that bombing has an upside worth being reckless about
            'GOT_KILLED': -3.0,        # standard family penalty
            'SURVIVED_ROUND': 0.0,     # still don't care about mere survival - unchanged theme from the rest of the family
            'TRAPPED_SELF': -5.0,      # standard family penalty
            'TRAPPED_ENEMY': 8.0,      # same as aggressive_5 - trap focus unchanged
            'INVALID_ACTION': -0.05,   # standard guidance step
            'STEP_PENALTY': -0.0025,   # standard family penalty
    },

    # aggressive_7: same kill-focus as aggressive_6, softening three specific penalties
    # identified as likely discouraging the exploratory/predictive bombing a real kill
    # strategy needs, rather than changing what's rewarded:
    # - BOMB_WASTEFUL halved: is_bomb_useful() (spatial_feature_extractor.py) only checks
    #   whether an opponent is in the blast radius AT THE MOMENT the bomb is dropped, not
    #   whether they're still there when it detonates several steps later. A bomb placed to
    #   cut off an escape route where the opponent WILL be (not where they currently are -
    #   i.e. genuine trap-setting) registers as wasteful under this check, even though it's
    #   exactly the kind of predictive placement a real kill needs. Softening this reduces
    #   the cost of that specific, currently-mislabeled "wasteful" behavior.
    # - MOVED_FROM_SAFETY less than halved: discourages staying near danger even to press an
    #   advantage (e.g. lingering after bombing to block an escape route instead of
    #   retreating immediately) - too strong a penalty here works directly against follow-
    #   through on an attack.
    # - STEP_PENALTY cut more aggressively (already the smallest magnitude term in the whole
    #   reward function, so expected impact is more marginal than the other two, but cheap
    #   to test): removes a small bias toward grabbing an easy nearby coin/finishing quickly
    #   over patient positioning near an opponent.
    # NOT included here (deferred, not confirmed): a graduated "cornering" potential
    # (continuous reward for progressively restricting an opponent's escape routes, vs the
    # current TRAP potential's all-or-nothing is_entity_trapped() check) - would require
    # changing evaluate_safety()'s BFS logic in spatial_feature_extractor.py, a shared
    # function also used for the agent's own danger-detection, so a bigger and riskier change
    # than adjusting reward magnitudes. Left as a future option, not attempted this round.
    "aggressive_7": {
            'WAITED': -0.005,          # unchanged from aggressive_6
            'MOVED_TO_SAFETY': 0.15,   # unchanged
            'MOVED_FROM_SAFETY': -0.07, # down from -0.15 (~55%) - less than halved, still a real penalty, just softer
            'CRATE_DESTROYED': 0.03,   # unchanged from aggressive_6
            'COIN_COLLECTED': 0.1,     # unchanged from aggressive_6
            'KILLED_OPPONENT': 20.0,   # unchanged
            'WON_ROUND': 3.0,          # unchanged
            'BOMB_USEFUL': 0.3,        # unchanged from aggressive_6
            'BOMB_WASTEFUL': -0.12,    # down from -0.3 (60%) - the main targeted change, see above
            'KILLED_SELF': -6.0,       # unchanged from aggressive_6
            'GOT_KILLED': -3.0,        # unchanged
            'SURVIVED_ROUND': 0.0,     # unchanged
            'TRAPPED_SELF': -5.0,      # unchanged
            'TRAPPED_ENEMY': 8.0,      # unchanged
            'INVALID_ACTION': -0.05,   # unchanged
            'STEP_PENALTY': -0.001,    # down from -0.0025 (60%)
    },

    "trapper": {
            'WAITED': -0.005,        # cost of 2 step penalties
            'MOVED_TO_SAFETY': 0.15,  # guidance step to reinforce flee
            'MOVED_FROM_SAFETY': -0.1,
            'CRATE_DESTROYED': 0.2,  # 5 crates = 1 coin, interesting but not enough to avoid collecting coin
            'COIN_COLLECTED': 0.80,	 # given in instructions
            'KILLED_OPPONENT': 25.0,	 # given in instructions
            'WON_ROUND': 4.0,          # synthetic terminal bonus - see train.py's WON_ROUND synthesis
            'BOMB_USEFUL': 0.5,      # guidance step, not as good as collecting a coin but better than destroying a crate, but want to make sure that agent places bombs near crates
            'BOMB_WASTEFUL': -0.1,    # guidance step, penalize the agent for using a bomb that doesn't destroy crate or kill opponent, but don't want to punish exploration t00 much
            'KILLED_SELF': -10.0,     # same penalty as being killed (will be added on top of being killed)
            'GOT_KILLED': -7.0,	     # opposite of killing opponent 
            'SURVIVED_ROUND': 0.20,	 # don't want to set this too high or else the agent won't take any risks, set to same value as killing opponent 
            'TRAPPED_SELF':  -7.0,   # same as being killed (will probably get killed)
            'TRAPPED_ENEMY': 1.2,	 # guidance step, as good as collecting a coin
            'INVALID_ACTION': -0.05, # guidance step, worse than moving/waiting, but want to make sure that agent isn't afraid of exploration 
            'STEP_PENALTY': -0.0015,  # want to play long but make sure we're around the end in ~400 steps, avoids panic playing and making crazy moves )
    },

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
        'CONFINE': 0.0     # discourage entering areas where agent can be confined

    },
    
    "aggressive": {
        "SURVIVE": 0.1,    # killed + kill self / 300 (survival isn't as important to aggressive agent)
        "EXPLORE": 0.05,     # coin / 100 (need to make sure what we don't explore forever)
        "COIN": 0.3,       # coin / 20 (want to weight killing over coin collecting)
        "CRATE": 0.15,     # crate / 20
        "KILL": 0.8,       # kill opponent / 50 (prioritize killing)
        "TRAP": 0.3,        # trap opponent / 50 (prioritize trapping but not as much as killing )
        "CONFINE": 0.0
    },
    "aggressive_1": {
            "SURVIVE": 0.1,    # killed + kill self / 300 (survival isn't as important to aggressive agent)
            "EXPLORE": 0.03,     # coin / 100 (need to make sure what we don't explore forever)
            "COIN": 0.2,       # was 0.5 - the highest COIN weight of *any* aggressive variant (even above
                               # aggressive_2/3's 0.15/0.05), which contradicted this dict's own "weight killing
                               # over coin collecting" comment. 0.2 keeps aggressive_1 as the mildest variant
                               # (still clearly above aggressive_2/3, since it's meant to be less extreme than
                               # them) while actually sitting below KILL (0.8) and TRAP (0.3) by a wide margin,
                               # matching the comment's stated intent instead of contradicting it.
            "CRATE": 0.15,     # crate / 20
            "KILL": 0.8,       # kill opponent / 50 (prioritize killing)
            "TRAP": 0.3,        # trap opponent / 50 (prioritize trapping but not as much as killing )
            "CONFINE": 0.0
    },
    "aggressive_2": {
            "SURVIVE": 0.1,    # killed + kill self / 300 (survival isn't as important to aggressive agent)
            "EXPLORE": 0.05,     # coin / 100 (need to make sure what we don't explore forever)
            "COIN": 0.15,       # coin / 20 (want to weight killing over coin collecting)
            "CRATE": 0.15,     # crate / 20
            "KILL": 0.8,       # kill opponent / 50 (prioritize killing)
            "TRAP": 0.4,        # trap opponent / 50 (prioritize trapping but not as much as killing )
            "CONFINE": 0.0
    },
    "aggressive_3": {
            "SURVIVE": 0.1,    # killed + kill self / 300 (survival isn't as important to aggressive agent)
            "EXPLORE": 0.05,     # coin / 100 (need to make sure what we don't explore forever)
            "COIN": 0.05,       # coin / 20 (want to weight killing over coin collecting)
            "CRATE": 0.05,     # crate / 20
            "KILL": 0.99,       # kill opponent / 50 (prioritize killing)
            "TRAP": 0.5,        # trap opponent / 50 (prioritize trapping but not as much as killing )
            "CONFINE": 0.0
    },
    # aggressive_4: "explore more, survive less" - EXPLORE raised ~12x (0.05 -> 0.6, now
    # comparable magnitude to KILL/TRAP instead of a rounding error next to them: at full-map
    # coverage this potential alone contributes +0.6, vs the old 0.05 max) and SURVIVE cut
    # 5x (0.1 -> 0.02) so the urgency-avoidance penalty near bombs matters less. COIN/CRATE
    # both raised too, since finding either requires actually moving through the map - ties
    # the exploration incentive to concrete instrumental goals instead of leaving it as pure
    # novelty-seeking. KILL/TRAP left at their original 0.99/0.5 (NOT reduced, despite the
    # near-zero real kill rate seen in the curriculum results) - the point of this family is
    # to train an agent that finds strategies to actually kill, so the incentive to try
    # should stay maximal even though it hasn't paid off yet; explore/survive are the levers
    # being changed here, not kill-seeking.
    # Enemy-focus adjustment (see BASE_REWARDS aggressive_4 above for the full reasoning):
    # EXPLORE/COIN/CRATE cut ~20% from the original "explore more" redesign so KILL/TRAP -
    # unchanged - make up proportionally more of the potential sum. SURVIVE left as-is since
    # it's already near-zero and only ever negative (a danger penalty, not a competing
    # positive incentive).
    "aggressive_4": {
                "SURVIVE": 0.02,   # unchanged - already near-zero, and only ever a danger penalty (never a competing positive reward)
                "EXPLORE": 0.48,     # down from 0.6 (~20%) - enemy-focus: shrink non-kill rewards, not the kill ones
                "COIN": 0.16,       # down from 0.2 (~20%)
                "CRATE": 0.16,      # down from 0.2 (~20%)
                "KILL": 0.99,       # unchanged - keep the incentive to find a kill strategy at full strength
                "TRAP": 0.5,        # unchanged, same reasoning
                "CONFINE": 0.0
    },

    # aggressive_5: only nonzero potentials, matching BASE_REWARDS' all-zero-except-kill/trap
    # design above. These two are the ONLY reward signal this agent gets before an actual
    # kill/trap happens - the continuous proximity shaping (w_kill/(distance+1), see
    # rl_heuristics.py's compute_potential) is what makes this trainable at all despite the
    # base rewards being otherwise fully sparse; without it there'd be no gradient guiding the
    # agent toward an opponent across the many steps before a kill/trap ever occurs.
    "aggressive_5": {
                "SURVIVE": 0.0,
                "EXPLORE": 0.0,
                "COIN": 0.0,
                "CRATE": 0.0,
                "KILL": 1.0,        # highest in the family - the only positive signal available before an actual kill lands
                "TRAP": 0.7,        # highest in the family, below KILL - trapping is a means to a kill, not equivalent
                "CONFINE": 0.0
    },

    # aggressive_6: KILL/TRAP unchanged from aggressive_5 (still the priority), COIN/CRATE
    # very low but nonzero (matching BASE_REWARDS' reasoning above), SURVIVE/EXPLORE at
    # modest family-standard levels rather than aggressive_5's zero - this variant isn't
    # trying to be maximally radical, just to fix the specific "no bombing incentive" gap.
    "aggressive_6": {
                "SURVIVE": 0.05,
                "EXPLORE": 0.05,
                "COIN": 0.02,      # very low - present but clearly not the point
                "CRATE": 0.02,     # very low, same reasoning
                "KILL": 1.0,       # unchanged from aggressive_5 - still the priority
                "TRAP": 0.7,       # unchanged from aggressive_5
                "CONFINE": 0.0
    },

    # aggressive_7: potentials unchanged from aggressive_6 - this variant only touches the
    # three BASE_REWARDS penalties above (BOMB_WASTEFUL, MOVED_FROM_SAFETY, STEP_PENALTY),
    # not the potential-shaping weights.
    "aggressive_7": {
                "SURVIVE": 0.05,
                "EXPLORE": 0.05,
                "COIN": 0.02,
                "CRATE": 0.02,
                "KILL": 1.0,
                "TRAP": 0.7,
                "CONFINE": 0.0
    },

    "trapper": {
            "SURVIVE": 1.0,     # killed + kill self / 100 (want to make sure peaceful agent doesn't flee from everything)
            "EXPLORE": 0.01,   # collect coin / 10  (need to make sure that we don't explore forever)
            "COIN": 0.1,       # collect coin / 5
            "CRATE": 0.02,      # destroy crate / 5
            "KILL": 0.03 ,      # kill opponent / 100
            "TRAP": 0.01,      # trap opponent / 100
            'CONFINE': 0.0     # discourage entering areas where agent can be confined
    
    },
}


def apply_env_overrides(base_rewards, potential_weights):
    """Applies environment overrides calculated from Bayesian Optimization (BO). 
    
    Sets NECKAR_<BEHAVIOR>_<KEY> environment variables before launching each trial's 
    training subprocess (ex. NECKAR_PEACEFUL_BOMB_WASTEFUL = -0.42)
    
    Scans for any variables matching a real entry in POTENTIAL_WEIGHTS or BASE_REWARDS
    and overrides the value. 
    
    If run without Bayesian Optimization, the entries will remain unchanged

    """
    prefix = "NECKAR_"
    
    # Keep list of keys that do not have behavior in the name
    NON_REWARD_KEYS = frozenset({
        "MODEL_TYPE",          # NECKAR_MODEL_TYPE
        "BEHAVIOR",            # NECKAR_BEHAVIOR
        "EPISODES_PER_UPDATE", # NECKAR_EPISODES_PER_UPDATE
        "MASK",                # NECKAR_MASK
        # Control knobs, not reward weights. Each contains a '_', so without listing them
        # here they get split into a bogus <behavior>_<key> pair and reported as unmatched
        # on every run that sets one (bo_search.py, run_eval.py, curriculum_run.py).
        "RUN_DIR",             # NECKAR_RUN_DIR      (run_eval.py, curriculum_run.py)
        "RUN_ID",              # NECKAR_RUN_ID       (bo_search.py, curriculum_run.py)
        "WARM_START",          # NECKAR_WARM_START   (bo_search.py, curriculum_run.py)
        "SEED",                # NECKAR_SEED         (train.py)
        "SCENARIO",            # NECKAR_SCENARIO     (callbacks.get_scenario)
        "EVAL_POLICY",         # NECKAR_EVAL_POLICY  (run_eval.py)
    })
    
    

    # Match against the actual known behavior names (peaceful, aggressive_1..4, trapper)
    # rather than naively splitting rest_key on its first underscore. Behavior names with
    # an underscore in them - every aggressive_N variant - break a first-underscore split:
    # "NECKAR_AGGRESSIVE_1_KILL" would split into behavior="aggressive", key="1_KILL", and
    # since bare "aggressive" is no longer a key in either dict (renamed to aggressive_1..4
    # when the variants were added), that override was silently dropped - every BO trial
    # would have run with aggressive_1's untouched defaults regardless of what was
    # suggested, making the search meaningless without ever raising an error. Sorting
    # candidates longest-first means a multi-word behavior name is always matched whole,
    # not accidentally truncated to a shorter prefix.
    known_behaviors = sorted(set(base_rewards) | set(potential_weights), key=len, reverse=True)

    # Collected here and reported once at the end, so a typo'd override surfaces as a
    # warning instead of silently training with the default weight.
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

        # Resolve <BEHAVIOR>_<KEY> by matching against the real behavior names, longest
        # first, rather than splitting on the first '_': behaviors like "aggressive_1"
        # contain an underscore themselves, so a naive split yields ("aggressive",
        # "1_KILL") and the override is silently dropped.
        behavior, key = None, None
        for candidate in known_behaviors:
            candidate_prefix = candidate.upper() + "_"
            if rest_key.upper().startswith(candidate_prefix):
                behavior = candidate
                key = rest_key[len(candidate_prefix):]
                break

        if behavior is None or not key:
            unmatched.append((env_key, raw_value, "expected NECKAR_<BEHAVIOR>_<KEY> but no '_' found"))    
            continue

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
    
    
        print(f"\n[agent_behavior] WARNING: {len(unmatched)} NECKAR_* environment "
              f"variable(s) did not match the expected NECKAR_<BEHAVIOR>_<KEY> structure "
              f"and were IGNORED - training proceeded using the DEFAULT weight for each "
              f"of these instead of the intended override:", file=sys.stderr)
        
        
        for env_key, raw_value, reason in unmatched:
            print(f"  - {env_key}={raw_value!r}: {reason}", file=sys.stderr)
 
        print(f"  Known behaviors: {known_behaviors}", file=sys.stderr)
        print(f"  Known keys: {known_keys}", file=sys.stderr)
        print(f"  Expected format: NECKAR_<BEHAVIOR>_<KEY>\n", file=sys.stderr)
 
    
    return base_rewards, potential_weights


BASE_REWARDS, POTENTIAL_WEIGHTS = apply_env_overrides(
    BASE_REWARDS, POTENTIAL_WEIGHTS)
            
        
        
        





