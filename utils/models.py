from typing import Callable

import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self


class MLP(nn.Module):
    """
    Basic MLP class that lets you pass dimensions in the same
    way as a KAN and the activation function
    """
    def __init__(
        self: Self,
        dimensions: list[int],
        activation_fn: Callable = F.tanh,
    ) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.activation_fn = activation_fn

        self.layers = nn.ModuleList()
        for dim_1, dim_2 in zip(dimensions, dimensions[1:]):
            self.layers.append(nn.Linear(dim_1, dim_2))
    
    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x
