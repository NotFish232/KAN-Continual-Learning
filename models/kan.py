import torch as T
from kan_layers import BSplineKanLayer
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from typing_extensions import Self


class KanModel(nn.Module):
    def __init__(
        self: Self, dims: list[int], spline_order: int = 3, num_knots: int = 7
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            BSplineKanLayer(d_1, d_2, spline_order, num_knots)
            for d_1, d_2 in zip(dims, dims[1:])
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    model = BSplineKanLayer(1, 1, 25, 3)
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

if __name__ == "__main__":
    main()
