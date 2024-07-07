from kan import KAN, KANLayer

from utils import num_parameters, suggest_KAN_architecture, suggest_MLP_architecture
from utils.io import ExperimentWriter
from utils.models import MLP

NUM_PARAMETERS = 5000

MLP_ARCHITECTURE = suggest_MLP_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=4,
    num_params=NUM_PARAMETERS,
)
KAN_ARCHITECTURE, KAN_GRID_SIZE = suggest_KAN_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=1,
    num_params=NUM_PARAMETERS,
)


def main() -> None:
    mlp = MLP(MLP_ARCHITECTURE)
    kan = KANLayer(1, 1, 5000)
    for p in kan.children():
        print(p, num_parameters(p))

    print(KAN_ARCHITECTURE, KAN_GRID_SIZE)
    print(num_parameters(mlp))
    print(num_parameters(kan))

if __name__ == "__main__":
    main()