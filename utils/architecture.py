from kan import KAN, KANLayer
from utils.models import MLP
from utils import num_parameters
from itertools import chain

# maps an (num_input, num_output, [num_layers]) => num_parameter_count => architecture
# architecture is tuple[list[int], int] in the case of a KAN
# architecture is list[int] in the case of a MLP
KAN_ARCHITECTURE: dict[
    tuple[int, int, int] | tuple[int, int], dict[int, tuple[list[int], int]]
] = {
    (1, 1, 1): {
        25: ([1, 1], 15),
        100: ([1, 1], 90),
        500: ([1, 1], 490),
        1_000: ([1, 1], 990),
    },
    (1, 1): {
        100: ([1, 2, 1], 15),
        1_000: ([1, 4, 1], 115),
        5_000: ([1, 16, 1], 148),
        10_000: ([1, 10, 10, 1], 74),
    },
    (2, 1, 1): {
        50: ([2, 1], 16),
        100: ([2, 1], 42),
        500: ([2, 1], 240),
        1_000: ([2, 1], 489),
    },
    (2, 1): {
        100: ([2, 1], 41),
        1_000: ([2, 4, 1], 74),
        5_000: ([2, 16, 1], 95),
        10_000: ([2, 12, 8, 1], 69),
    },
    (28**2, 10): {
        10_000: ([28**2, 1, 10], 4),
        100_000: ([28**2, 3, 10], 33),
        500_000: ([28**2, 10], 55),
        1_000_000: ([28**2, 16, 10], 70),
    },
}

MLP_ARCHITECTURE: dict[tuple[int, int], dict[int, list[int]]] = {
    (1, 1): {
        25: [1, 8, 1],
        100: [1, 33, 1],
        500: [1, 20, 20, 1],
        1_000: [1, 30, 30, 1],
        5_000: [1, 74, 64, 1],
        10_000: [1, 64, 76, 64, 1],
    },
    (2, 1): {
        50: [2, 12, 1],
        100: [2, 25, 1],
        500: [2, 20, 20, 1],
        1_000: [2, 29, 30, 1],
        5_000: [2, 72, 64, 1],
        10_000: [2, 64, 75, 64, 1],
    },
    (28**2, 10): {
        100_000: [28**2, 128, 10],
        500_000: [28**2, 542, 128, 10],
        1_000_000: [28**2, 1024, 192, 10],
    },
}


def main() -> None:
    kan_architectures = [x.items() for x in KAN_ARCHITECTURE.values()]
    for expected_param_count, (architecture, grid_size) in chain(*kan_architectures):
        kan = KAN(architecture, grid_size)
        param_count = num_parameters(kan)
        print(
            f"KAN: {architecture = }, {grid_size = }",
            f"expected: {expected_param_count} vs actual: {param_count}",
            f"({abs(param_count - expected_param_count) / expected_param_count:.2%} off)",
        )

    mlp_architectures = [x.items() for x in MLP_ARCHITECTURE.values()]
    for expected_param_count, architecture in chain(*mlp_architectures):
        mlp = MLP(architecture)
        param_count = num_parameters(mlp)
        print(
            f"MLP: {architecture = }",
            f"expected: {expected_param_count} vs actual: {param_count}",
            f"({abs(param_count - expected_param_count) / expected_param_count:.2%} off)",
        )


if __name__ == "__main__":
    main()
