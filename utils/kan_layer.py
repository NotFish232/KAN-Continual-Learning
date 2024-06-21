import torch as T
from torch import nn
from typing_extensions import Self
from matplotlib import pyplot as plt


class BSplineBasisFunctions(nn.Module):
    def __init__(self: Self, spline_order: int, num_knots: int) -> None:
        super().__init__()

        self.spline_order = spline_order
        self.num_knots = num_knots

        self.knot_vector: T.Tensor
        self.register_buffer(
            "knot_vector",
            T.linspace(-1, 1, num_knots),
            persistent=False,
        )

    def forward(self: Self, t: T.Tensor) -> T.Tensor:
        batched_t = t.unsqueeze(1).repeat(1, self.num_knots - 1)

        basis_functions = (self.knot_vector[:-1] <= batched_t) & (
            self.knot_vector[1:] > batched_t
        )

        for k in range(1, self.spline_order):
            basis_functions = (batched_t[:, :-k] - self.knot_vector[: -k - 1]) / (
                self.knot_vector[k:-1] - self.knot_vector[: -k - 1]
            ) * basis_functions[:, :-1] + (
                self.knot_vector[k + 1 :] - batched_t[:, :-k]
            ) / (
                self.knot_vector[k + 1 :] - self.knot_vector[1:-k]
            ) * basis_functions[
                :, 1:
            ]

        return basis_functions


class KanLayer(nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        out_dim: int,
        spline_order: int = 3,
        num_knots: int = 7,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.spline_order = spline_order
        self.num_knots = num_knots

        self.w_b = nn.Parameter(T.FloatTensor(1))
        self.w_s = nn.Parameter(T.FloatTensor(1))

        self.spline_parameters = nn.Parameter(
            T.randn((in_dim, out_dim, num_knots - spline_order))
        )

        self.basis_functions = BSplineBasisFunctions(spline_order, num_knots)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        bases = self.basis_functions(x.flatten()).reshape(*x.shape, -1)
        print(bases.shape)
        print(self.spline_parameters.shape)

        exit(1)


def main() -> None:
    device = T.device("cuda")

    k = KanLayer(2, 3)
    y = k(T.randn(55, 2))

    print(y)


if __name__ == "__main__":
    main()
