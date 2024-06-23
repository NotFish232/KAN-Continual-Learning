import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

class BSplineBasisFunctions(nn.Module):
    def __init__(self: Self, spline_order: int, num_knots: int) -> None:
        super().__init__()

        self.spline_order = spline_order
        self.num_knots = num_knots

        self.knot_vector: T.Tensor
        self.register_buffer(
            "knot_vector",
            T.linspace(-1 - 2 / num_knots, 1 + 2 / num_knots, num_knots),
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


class BSplineKanLayer(nn.Module):
    def __init__(
        self: Self,
        in_dim: int,
        out_dim: int,
        spline_order: int,
        num_knots: int,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.spline_order = spline_order
        self.num_knots = num_knots

        self.w_b = nn.Parameter(T.tensor(1, dtype=T.float32))
        self.w_s = nn.Parameter(T.tensor(1, dtype=T.float32))

        self.spline_parameters = nn.Parameter(
            T.randn((out_dim, in_dim, num_knots - spline_order))
        )

        self.basis_functions = BSplineBasisFunctions(spline_order, num_knots)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        # x.shape (batch_size, in_dim)
        bases = self.basis_functions(x.flatten()).reshape(*x.shape, -1).unsqueeze(1)
        # bases.shape (batch_size, 1, in_dim, num_knots - spline_order)
        splines = T.sum(self.spline_parameters * bases, dim=-1)
        # splines.shape (batch_size, out_dim, in_dim)
        final = T.sum(self.w_b * F.silu(x.unsqueeze(1)) + self.w_s * splines, dim=-1)
        # final.shape (batch_size, out_dim)
        return final

