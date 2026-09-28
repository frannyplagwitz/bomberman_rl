import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal 

class MLPActorCritic(nn.Module):
    """ Need two separate networks, one for actor (policy) one for critic (value function)
        Will support discrete and continuous spaces
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        
        """Initialize the MLP networks 
        Params: 
        input_dim: The input size
        """
        
        super().__init__()
        
        
        # Instantiate separate actor critic networks 
        self.actor_base = self.build_mlp_base(input_dim, hidden_dim)
        self.critic_base = self.build_mlp_base(input_dim, hidden_dim)
        
        # Output heads 
        self.actor_head = nn.Linear(hidden_dim, action_dim) # outputs discrete logits (policy)
        self.critic_head = nn.Linear(hidden_dim, 1)         # state value (V(s))

    
    def build_mlp_base(self, input_dim: int, hidden_dim: int) -> nn.Sequential: 
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())
        
        
    def forward(self, state: torch.Tensor): 
        """Move through layers and return action distribution + values"""
        
        # Actor forward pass, returns logits
        actor_features = self.actor_base(state)
        logits = self.actor_head(actor_features)

        # Critic forward pass, returns state value (V(s))
        critic_features = self.critic_base(state)
        value = self.critic_head(critic_features).squeeze(-1)
  
        return logits, value
    
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value V(s) without computing logits, used for rollouts """
        critic_features = self.critic_base(state)
        return self.critic_head(critic_features).squeeze(-1)
        

    def get_action_value(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """Helper function that is used to obtain action-value (Q(s,a)) for rollout and also loss calculation 
           Returns the action, probability of choosing the action (log_prob), entropy and the state value """
        
        logits, value = self(state)
        dist = Categorical(logits=logits)
        
    
        # If no action, sample action from probabilities
        if action is None: 
            action = dist.sample()
            
        # Find probability and entropy 
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
        