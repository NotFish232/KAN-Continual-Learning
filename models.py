from typing import Callable

import torch as T
from torch import nn
from typing_extensions import Self


class BSpline(nn.Module):
    def __init__(
        self: Self,
        grid_size: int,
        spline_order: int,
        control_points: T.FloatTensor | list[float] | None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.spline_order = spline_order

        if control_points is not None:
            self.control_points = nn.Parameter(
                T.FloatTensor(control_points), requires_grad=requires_grad
            )
        else:
            self.control_points = nn.Parameter(
                T.randn(grid_size - spline_order - 1),
                requires_grad=requires_grad,
            )
        

        self.control_points = control_points
        self.knot_vector = knot_vector

    @staticmethod
    def evaluate_basis_functions(
        knot_vector: list[float],
        spline_order: int,
        t: float,
    ) -> list[float]:
        previous_basis_functions: list[float] = []
        for k in range(spline_order + 1):
            n = len(knot_vector) - k - 1

            if k == 0:
                for i in range(n):
                    value = float(knot_vector[i] <= t < knot_vector[i + 1])
                    previous_basis_functions.append(value)
            else:
                new_basis_functions = []
                for i in range(n):
                    new_basis_functions.append(
                        (t - knot_vector[i])
                        / (knot_vector[i + k] - knot_vector[i])
                        * previous_basis_functions[i]
                        + (knot_vector[i + k + 1] - t)
                        / (knot_vector[i + k + 1] - knot_vector[i + 1])
                        * previous_basis_functions[i + 1]
                    )
                previous_basis_functions = new_basis_functions

        return previous_basis_functions


class KanLayer(nn.Module):
    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        num_knots: int,
    ) -> None:
        super().__init__()

        self.w_b = nn.Parameter(T.tensor(1, dtype=T.float32))
        self.w_s = nn.Parameter(T.tensor(1, dtype=T.float32))
        pass


x = BSpline(2, [*range(4)], [*range(7)])
print(BSpline.evaluate_basis_functions([*range(7)], 1, 1.5))
