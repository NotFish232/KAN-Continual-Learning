import torch as T
from torch import nn
from typing_extensions import Self
from torch import optim
from matplotlib import pyplot as plt


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
        series = nn.functional.silu(self.fourier_coeffs[0] * T.sin(grid_x))
        series += nn.functional.silu(self.fourier_coeffs[1] * T.cos(grid_x))
        # series.shape (batch_size, out_dim, in_dim, grid_size)
        summed_series = T.sum(series, dim=(-1, -2))
        # summed_series.shape (batch_size, out_dim)
        return summed_series


if __name__ == "__main__":
    model = FourierSeriesKanLayer(1, 1, 25)
    optimizer = optim.Adam(model.parameters(), 1e-2)
    criterion = nn.MSELoss()

    x = T.linspace(-1, 1, 500).unsqueeze(1)
    y = (x[:, :1]) ** 2 - T.sin((2 * x) ** 3) + T.cos((3 * x) ** 2)

    while True:
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        plt.clf()
        plt.title(f"Loss: {loss:.5f}")
        for i in range(y.shape[1]):
            plt.plot(x, y[:, i])
            plt.plot(x, y_hat.detach()[:, i])
        plt.pause(0.01)
