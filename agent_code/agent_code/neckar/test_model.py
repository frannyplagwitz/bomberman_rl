"""Quick checks for the neckar models 

1. Shapes: both models accept the flat 1742-dim input, batched and single.
2. Warm start: a CNN warm-started from an old checkpoint gives the SAME outputs
   as the old CNN (the new scalar weights start at 0), so nothing was lost.
 
Run from agent_code/neckar/ :
    python test_models.py                         # warm-start test on a random old model
    python test_models.py --old-ckpt path/to/old_cnn_checkpoint.pt
"""
import argparse
import os
import tempfile
 
import torch
import torch.nn as nn
 
from models.CNN import CNNActorCritic, warm_start_from_old_cnn
from models.MLP import MLPActorCritic
 
 
class OldCNN(nn.Module):
    """The CNN architecture BEFORE the 8 scalars were added (same layer names/indices)."""
 
    def __init__(self, action_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
 
        def base():
            return nn.Sequential(
                nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 17 * 17, hidden_dim), nn.ReLU())
 
        self.actor_base = base()
        self.critic_base = base()
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
 
    def forward(self, grid):
        return (self.actor_head(self.actor_base(grid)),
                self.critic_head(self.critic_base(grid)).squeeze(-1))
 
 
def check_shapes():
    print("== Shape check ==")
    models = [("CNN", CNNActorCritic(action_dim=6, n_scalars=8)),
              ("MLP", MLPActorCritic(input_dim=1742, action_dim=6))]
    for name, model in models:
        model.eval()
        with torch.no_grad():
            logits_b, value_b = model(torch.zeros(4, 1742))  # batch of 4, as in the PPO update
            logits_1, value_1 = model(torch.zeros(1, 1742))  # batch of 1, as in act()
            
        print(f"  {name}: batch -> logits {tuple(logits_b.shape)}, value {tuple(value_b.shape)} | "
              f"single -> logits {tuple(logits_1.shape)}, value {tuple(value_1.shape)}")
        assert logits_b.shape == (4, 6), f"{name}: expected logits (4, 6)"
        assert logits_1.shape == (1, 6), f"{name}: expected logits (1, 6)"
        if name == "CNN":
            assert value_b.shape == (4,), "CNN: expected value (4,)"
    print("  OK")
 
 
def check_warm_start(old_ckpt: str = None):
    print("== Warm-start check ==")
    old = OldCNN()
    if old_ckpt:
        old.load_state_dict(torch.load(old_ckpt, map_location="cpu"))
        path = old_ckpt
        print(f"  using old checkpoint: {old_ckpt}")
    else:
        path = os.path.join(tempfile.mkdtemp(), "old_cnn_random.pt")
        torch.save(old.state_dict(), path)
        print("  no checkpoint given, using a randomly initialised old CNN")
 
    new = CNNActorCritic(action_dim=6, n_scalars=8)
    warm_start_from_old_cnn(new, path)
 
    old.eval()
    new.eval()
    x = torch.randn(5, 1742)                      # random grid AND random scalars
    grid = x[:, :6 * 17 * 17].reshape(-1, 6, 17, 17)
    with torch.no_grad():
        old_logits, old_value = old(grid)
        new_logits, new_value = new(x)
 
    diff_logits = (old_logits - new_logits).abs().max().item()
    diff_value = (old_value - new_value).abs().max().item()
    print(f"  max |difference|: logits {diff_logits:.2e}, value {diff_value:.2e}")
    assert diff_logits < 1e-4 and diff_value < 1e-4, "warm-started CNN differs from the old CNN"
    print("  OK: the warm-started model behaves exactly like the old one")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ckpt", default=None, help="old CNN checkpoint (.pt) to warm-start from")
    args = parser.parse_args()
 
    check_shapes()
    check_warm_start(args.old_ckpt)
 