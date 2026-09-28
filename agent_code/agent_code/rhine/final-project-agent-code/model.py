"""Actor-Critic MLP: 2 hidden layers x 64 units, shared trunk, two output
heads (policy logits, value). Feature-extraction code upstream feeds this;
the action mask is applied to the logits before sampling/argmax.
"""
import torch
import torch.nn as nn

# Large finite negative value instead of -inf: masked_fill(..., -inf) can produce
# NaN gradients through softmax/log_prob in edge cases; a large finite value is
# numerically equivalent (probability ~0 after softmax) without that risk.
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
