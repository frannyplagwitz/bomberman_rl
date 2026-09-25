import torch
import torch.nn as nn 
import torch.nn.functional as F


class TabularQAgent(nn.Module): 
    """Lookup table (LUT) for Tabular Q-Learning reinforcement learning 
       Manages table of action-state values (Q(s,a))
       Q: state_id -> Q(s,a) for each action 
       
       Action selection is epsilon-greedy and is handled by callbacks
       Learning is done via TD update 
    """
    

    # Actions will be ["UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"]
        
    def __init__(self, num_states: int, action_dim: int):
        
        super().__init__()
        
        self.num_states = num_states
        self.action_dim = action_dim 
        
        # Initialize q_table and embed with weights
        self.q_table = nn.Embedding(num_states, action_dim) # (state idx -> action-state value)
        nn.init.zeros_(self.q_table.weight)


    def forward(self, state: torch.Tensor):
        """Returns Q(s, a) for the given state index/indices. shape = [batch, action_dim]"""
        return self.q_table(state)
        
              
    def get_value(self, state: torch.Tensor) -> torch.Tensor: 
        """Get state value from the best result stored at the passed state"""
        return self.q_table(state).max(dim=-1).values

        
    @torch.no_grad()
    def td_update(self, state_idx: int, action: int, reward: float, next_state_idx: "int | None", 
                  alpha: float = 0.1, gamma: float = 0.99, next_legal_mask: "torch.Tensor | None" = None) -> float:
        
        """One-step TD tabular Q-learning
        
        Value-based update for Q-learning is defined as Q(s,a) <-- Q(s,a) + α[Target - Q(s,a)]
        We are using TD(0) as the target, which is defined as r + γ max_a' Q(s', a')
        
        How it works: 
        
        1. Grab the action-state value using the passed state index and current action 
        2. Grab the next action-state value using the next state index and the current action
        3. Calculate the update target (if no next state, will just be the reward)
        4. Calculate the difference between the target and the current action state value 
        5. Multiply by learning rate and add to current action state value 
        
        Since this is Q learning, and Q-learning behaves greedily, the next action will the action that results in 
        the max score
        
        Params: 
            state_idx: state index that values are stored at 
            action: action that was taken 
            next_state_idx: next state index where values are stored at 
            alpha: learning rate, 0.1 by default
            gamma: discount factor, 0.99 by default
            next_legal_mask: optional bool tensor of shape (action_dim, ), 
                True where action is legal in the next state
        """
        
        # Grab the current action-state value 
        state_action = self.q_table.weight[state_idx, action]
        
        if next_state_idx is None:
            target = reward
        
        else: 
            next_q = self.q_table.weight[next_state_idx] # Grab the q-value tensor stored at next state idx
            
        
            # Need to check if that action is masked though before committing
            
            # If mask, choose the unmasked action that results in the best Q-Value 
            if next_legal_mask is not None:
                if next_legal_mask.any():
                    next_state_action = next_q[next_legal_mask].max().item()
                else:
                    next_state_action = next_q[5].item()   # WAIT, same fallback as act()
                    
            else:
                # No mask given: plain Q-learning, max over all actions
                next_state_action = next_q.max().item()
 
            target = reward + gamma * next_state_action
            
        # Find the td error and add value * learning rate to value in table 
        td_error = target - state_action.item() 
        self.q_table.weight[state_idx, action] += alpha * td_error
        
        return td_error 
        
        
    
