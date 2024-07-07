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
        activation_function: Callable = F.relu,
    ) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.activation_function = activation_function

        self.layers = nn.ModuleList()
        for dim_1, dim_2 in zip(dimensions, dimensions[1:]):
            self.layers.append(nn.Linear(dim_1, dim_2))
    
    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation_function(x)
        return x
