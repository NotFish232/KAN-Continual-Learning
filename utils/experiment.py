from typing import Any

import torch as T
from kan import KAN
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import train_model_v2
from utils.data_management import ExperimentDataType, ExperimentWriter
from utils.models import MLP


def run_experiment(
    experiment_name: str,
    kan_architecture: list[int],
    mlp_architecture: list[int],
    task_datasets: list[Dataset],
    eval_datasets: dict[str, list[Dataset] | Dataset],
    pred_datasets: dict[str, list[T.Tensor] | T.Tensor],
    pred_ground_truth: dict[str, list[T.Tensor] | T.Tensor],
    experiment_dtype: ExperimentDataType,
    kan_kwargs: dict[str, Any] = {},
    mlp_kwargs: dict[str, Any] = {},
    device: T.device | None = None,
) -> None:
    device = device or T.device("cuda" if T.cuda.is_available() else "cpu")

    kan = KAN(kan_architecture, device=device, **kan_kwargs)
    mlp = MLP(mlp_architecture, **mlp_kwargs).to(device)

    models = {"kan": kan, "mlp": mlp}

    results: dict[str, Any] = {
        f"{model}_{metric}_loss": []
        for model in models
        for metric in ["train", *eval_datasets]
    } | {
        f"{model}_{metric}_predictions": []
        for model in models
        for metric in pred_datasets
    }

    for task_idx, task_dataset in enumerate(tqdm(task_datasets)):
        datasets = {"train": task_dataset} | {
            metric: d[task_idx] if isinstance(d, list) else d
            for metric, d in eval_datasets.items()
        }

        for model_name, model in models.items():
            model_results = train_model_v2(model, datasets)

            for metric, value in model_results.items():
                results[f"{model_name}_{metric}_loss"].extend(value)

            for metric, pred_dataset in pred_datasets.items():
                with T.no_grad():
                    predictions = model(
                        pred_dataset[task_idx]
                        if isinstance(pred_dataset, list)
                        else pred_dataset
                    )

                results[f"{model_name}_{metric}_predictions"].append(predictions)

    experiment_writer = ExperimentWriter(experiment_name, experiment_dtype)

    experiment_writer.log_config("mlp_architecture", mlp_architecture)
    experiment_writer.log_config("kan_architecture", kan_architecture)

    for k, v in results.items():
        if not isinstance(v[0], T.Tensor):
            v = T.tensor(v)
        experiment_writer.log_data(k, v)

    for metric, ground_truth in pred_ground_truth.items():
        experiment_writer.log_data(f"base_{metric}_predictions", ground_truth)

    experiment_writer.write()
