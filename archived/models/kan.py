from typing import Any

import torch as T
from torch import nn
from typing_extensions import Self

from .kan_layers import KanLayer


class KanModel(nn.Module):
    def __init__(
        self: Self,
        dims: list[int],
        layer_type: KanLayer,
        grid_size: int,
        layer_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            layer_type.value(d_1, d_2, grid_size, **layer_kwargs)
            for d_1, d_2 in zip(dims, dims[1:])
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
