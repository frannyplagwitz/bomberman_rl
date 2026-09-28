"""Actor-critic MLP with a shared trunk and separate policy/value heads."""
import torch
import torch.nn as nn

# Finite instead of -inf: -inf can produce NaN gradients through
# softmax/log_prob, while a large finite value is numerically equivalent.
MASK_NEG_VALUE = -1e8


class ActorCriticMLP(nn.Module):
    def __init__(self, n_features: int, n_actions: int, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(in_dim, n_actions)
        self.critic_head = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor):
        z = self.trunk(features)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask: bool tensor, True = legal action. Illegal actions get near-zero probability."""
    return logits.masked_fill(~mask, MASK_NEG_VALUE)
