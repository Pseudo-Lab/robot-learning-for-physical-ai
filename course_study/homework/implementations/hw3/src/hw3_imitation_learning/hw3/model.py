from torch import nn
import torch
from torch.nn import functional as F
from ethz_starter.hw3_imitation_learning.hw3.model import BasePolicy


class ObstaclePolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = [nn.Linear(state_dim, d_model), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(d_model, d_model), nn.ReLU()))
        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        flat_actions = self.net(state)
        return flat_actions.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class MultiTaskPolicy(ObstaclePolicy):
    """The input includes the selected cube, so the action head is shared across goals."""


def build_policy(
    policy_type, *, state_dim, action_dim, chunk_size, d_model=128, depth=2
):
    classes = {"obstacle": ObstaclePolicy, "multitask": MultiTaskPolicy}
    if policy_type not in classes:
        raise ValueError(f"Unknown policy type: {policy_type}")
    return classes[policy_type](state_dim, action_dim, chunk_size, d_model, depth)
