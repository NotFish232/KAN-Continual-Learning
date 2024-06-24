import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self


class FourierSeriesKanLayer(nn.Module):
    def __init__(self: Self, in_dim: int, out_dim: int, grid_size: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        self.grid: T.Tensor
        self.register_buffer("grid", T.arange(1, self.grid_size + 1).reshape(1, 1, -1))

        self.fourier_coeffs = nn.Parameter(T.randn(2, out_dim, in_dim, grid_size))

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        # x.shape (batch_size, in_dim)
        grid_x = (x.unsqueeze(-1) * self.grid).unsqueeze(1)
        # grid_x.shape (batch_size, 1, in_dim, grid_size)
        series = F.silu(self.fourier_coeffs[0] * T.sin(grid_x))
        series += F.silu(self.fourier_coeffs[1] * T.cos(grid_x))
        # series.shape (batch_size, out_dim, in_dim, grid_size)
        summed_series = T.sum(series, dim=(-1, -2))
        # summed_series.shape (batch_size, out_dim)
        return summed_series
