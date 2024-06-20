import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import Self


class BSpline:
    def __init__(
        self: Self,
        control_points: list[float],
        knot_vector: list[float],
        spline_order: int,
    ) -> None:
        super().__init__()

        self.control_points = control_points
        self.knot_vector = knot_vector

        self.spline_order = spline_order

    def evaluate_basis_functions(self: Self, t: float) -> list[float]:
        previous_basis_functions: list[float] = []
        for k in range(self.spline_order):
            n = len(self.knot_vector) - k - 1

            if k == 0:
                for i in range(n):
                    value = float(self.knot_vector[i] <= t < self.knot_vector[i + 1])
                    previous_basis_functions.append(value)
            else:
                new_basis_functions = []
                for i in range(n):
                    new_basis_functions.append(
                        (t - self.knot_vector[i])
                        / (self.knot_vector[i + k] - self.knot_vector[i])
                        * previous_basis_functions[i]
                        + (self.knot_vector[i + k + 1] - t)
                        / (self.knot_vector[i + k + 1] - self.knot_vector[i + 1])
                        * previous_basis_functions[i + 1]
                    )
                previous_basis_functions = new_basis_functions

        return previous_basis_functions

    def get_value_at(self: Self, t: float) -> float:
        basis_values = self.evaluate_basis_functions(t)
        value = sum(c * b for c, b in zip(self.control_points, basis_values))
        return value

    def plot(self: Self) -> None:
        num_samples = 200

        x = np.linspace(self.knot_vector[0], self.knot_vector[-1], num_samples)
        y = [self.get_value_at(t) for t in x]
        plt.plot(x, y, color="red")

        bases = [self.evaluate_basis_functions(t) for t in x]
        for i in range(len(self.control_points)):
            basis = [b[i] for b in bases]
            plt.plot(x, basis)

        c_x = x[np.argmax(bases, axis=0)]
        plt.scatter(c_x, self.control_points, color="blue")

        plt.title(self.__str__())
        plt.legend(
            [
                "B Spline Curve",
                *(f"Basis function {b + 1}" for b in range(len(self.control_points))),
                "Control Points",
            ]
        )
        plt.show()


def main() -> None:
    spline = BSpline([5, -5, 3, -5], [0, 5, 5.5, 6, 8, 9, 10], 3)
    spline.plot()


if __name__ == "__main__":
    main()
