import torch
import torch.nn as nn
from torch.distributions import Categorical

class CNNActorCritic(nn.Module):
    """ Need two separate networks, one for actor (policy) one for critic (value function)
    """
    
    GRID_SHAPE = (6, 17, 17) # 6 x 17 x 17
    
    def __init__(self, action_dim: int, n_scalars: int = 8, hidden_dim: int = 64):
        
        super().__init__()

        # Set the grid size, feature scalars as well as total output
        self.grid_size = 6 * 17 * 17
        self.n_scalars = n_scalars
        conv_out = 64 * 17 * 17  # 18496
        
        # Instantiate separate actor critic networks 
        # Create convolution layers as well as the fully connected layer
        
        
        self.actor_conv = self.build_conv()
        self.critic_conv = self.build_conv()
        self.actor_fc = nn.Sequential(nn.Linear(conv_out + n_scalars, hidden_dim), nn.ReLU())
        self.critic_fc = nn.Sequential(nn.Linear(conv_out + n_scalars, hidden_dim), nn.ReLU())

        # Output heads 
        self.actor_head = nn.Linear(hidden_dim, action_dim) # outputs discrete logits (policy )
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        
    @staticmethod
    def build_conv() -> nn.Sequential: 
        # 3 x 3x3 convolutions, with padding of 1 to allow feature maps to keep 17 x 17 size
        # Conv 1: 6 input channels and outputs 32 feature maps of size 17 x 17
        # Conv 2: 32 input channels and outputs 64 feature maps of size 17 x 17
        # Conv 3: 64 input channels and outputs 64 feature maps of size 17 x 17
        
        return nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten())     
        
    def split(self, x: torch.Tensor):
        """Split flattened tensor x [batch, 1742] into grid [batch, 6, 17, 17] and one-hot scalars [batch, 8]"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        grid = x[:, :self.grid_size].view(-1, *self.GRID_SHAPE)
        scalars = x[:, self.grid_size:self.grid_size + self.n_scalars]
        return grid, scalars
    
    def encode(self, conv: nn.Module, fc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        "Encode one-hot feature vector into convolutional board features"
        grid, scalars = self.split(x)
        return fc(torch.cat([conv(grid), scalars], dim=1))
    
      
    def forward(self, state: torch.Tensor): 
        """Move through layers and return action distribution + values"""
        
        # Actor (compute probability logits)
        # Critic (conpute state value V(s))
        logits = self.actor_head(self.encode(self.actor_conv, self.actor_fc, state))
        value = self.critic_head(self.encode(self.critic_conv, self.critic_fc, state)).squeeze(-1)

        return logits, value 
    
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value V(s) without computing logits, used for rollouts """
        return self.critic_head(self.encode(self.critic_conv, self.critic_fc, state)).squeeze(-1)
    

def warm_start_from_old_cnn(new_model: CNNActorCritic, old_path: str):
    """Load a checkpoint of CNN from BEFORE the 8 spatial scalars were added.
    
    This makes it so that every layer that still fits is copied, and old weights will be kept.
    The main difference is that it will add weights for the 8 spatial scalars which will start at 0

    Params:
        new_model: freshly built CNNActorCritic (with n_scalars > 0)
        old_path: path to the old CNN checkpoint (.pt state_dict)
    """
    old = torch.load(old_path, map_location="cpu")
    new = new_model.state_dict()
 
    # Old Sequential indices: 0/2/4 = convs, 7 = Linear (1,3,5,8 = ReLU, 6 = Flatten)
    rename = {"_base.0.": "_conv.0.", "_base.2.": "_conv.2.",
              "_base.4.": "_conv.4.", "_base.7.": "_fc.0."}
 
    copied, enlarged = [], []
    for old_key, value in old.items():
        key = old_key
        for a, b in rename.items():
            key = key.replace(a, b)
        if key not in new:
            continue
        if new[key].shape == value.shape:
            new[key] = value.clone()
            copied.append(key)
        elif new[key].dim() == 2 and new[key].shape[0] == value.shape[0]:
            # enlarged FC weight [64, 18496] -> [64, 18504]: old inputs first, new scalars at 0
            new[key].zero_()
            new[key][:, :value.shape[1]] = value
            enlarged.append(key)
 
    # Every parameter of the new model must have come from the old checkpoint
    missing = [k for k in new if k not in copied and k not in enlarged]
    if missing:
        raise ValueError(f"Warm start from {old_path} left parameters uninitialised: {missing}")
 
    new_model.load_state_dict(new)
    print(f"[WARM START] loaded {old_path}: {len(copied)} tensors copied, "
          f"{len(enlarged)} enlarged ({enlarged})", flush=True)
 


