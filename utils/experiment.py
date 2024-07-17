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
    device: T.device | None = None,
    kan_kwargs: dict[str, Any] = {},
    mlp_kwargs: dict[str, Any] = {},
    training_kwargs: dict[str, Any] = {},
) -> None:
    """
    Runs an experiment on the models provided keeping track of various metrics,
    such as training / evaluation loss and predictions on datasets,
    saves these results to file using `ExperimentWriter` which can later be read
    with `ExperimentReader`

    Results are saved as the following:
        - task_dataset has an associated train_loss of type T.Tensor[T.float32], which is same len as eval results
        - each eval_dataset has an associated ${model}_${eval_dataset_name}_loss entry of type T.Tensor[T.float32]
        - each pred_dataset has an associated ${model}_$(pred_dataset_name)_predictions of type list[T.Tensor] and len(task_datasets)
        - each pred_ground_truth has an associated base_${metric}_{predictions} which is same as value passed in

    Parameters
    ----------
    experiment_name : str
        Name of the experiment, used for writing / reading the experiment

    kan_architecture : list[int]
        Represents the width of each layer of the kan

    mlp_architecture : list[int]
        Represents  the width of each layer of the mlp

    task_datasets : list[Dataset]
        Dataset for each individual task in continual learning

    eval_datasets : dict[str, list[Dataset] | Dataset]
        Dataset to evaluate metrics on, i.e. RMSE Loss by default
        if type list[Dataset] elements are evaluated only on corresponding tasks
        if type Dataset elements are evaluated on all tasks

    pred_datasets : dict[str, list[T.Tensor] | T.Tensor]
        Tensors to save model predictions for
        if type list[T.Tensor] predictions are evaluated only on corresponding tasks
        if type T.Tensor predictions are evaluated on all tasks

    pred_ground_truth : dict[str, list[T.Tensor] | T.Tensor]
        Ground truth labels for predictions

    experiment_dtype : ExperimentDataType
        Type for dataset, i.e., 1d functions, 2d functions, images, etc

    device : T.device | None, optional
        Device to run on, defaults to cuda if available else cpu

    kan_kwargs : dict[str, Any], optional
        Additional kwargs to pass to kan constructor, by default {}

    mlp_kwargs : dict[str, Any], optional
        Additional kwargs to pass to mlp constructor, by default {}

    training_kwargs : dict[str, Any], optional
        Additional kwargs to pass to training function, by default {}

    Returns
    ----------
    None
    """

    # use passed device or cuda if available else cpu
    device = device or T.device("cuda" if T.cuda.is_available() else "cpu")

    kan = KAN(kan_architecture, device=device, **kan_kwargs)
    mlp = MLP(mlp_architecture, **mlp_kwargs).to(device)

    models = {"kan": kan, "mlp": mlp}

    # add all metrics to resultst
    # each metric is of the form {model}_{metric}_{loss|predictions}
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
        # prepare datasets which is the task_dataset + each eval_dataset
        datasets = {"train": task_dataset} | {
            metric: d[task_idx] if isinstance(d, list) else d
            for metric, d in eval_datasets.items()
        }

        for model_name, model in models.items():
            model_results = train_model_v2(model, datasets, **training_kwargs)

            # update results with training results
            for metric, value in model_results.items():
                results[f"{model_name}_{metric}_loss"].extend(value)

            # update results with each model prediction for each pred_dataset
            for metric, pred_dataset in pred_datasets.items():
                with T.no_grad():
                    predictions = model(
                        pred_dataset[task_idx]
                        if isinstance(pred_dataset, list)
                        else pred_dataset
                    )

                results[f"{model_name}_{metric}_predictions"].append(predictions)

    experiment_writer = ExperimentWriter(experiment_name, experiment_dtype)

    # log some experimental configuration
    experiment_writer.log_config("mlp_architecture", mlp_architecture)
    experiment_writer.log_config("kan_architecture", kan_architecture)

    # values written to experiment_write should be either list[T.Tensor] or T.Tensor
    # if first element of v is not a T.Tensor assume its supposed to be a scaler tensor and convert
    for k, v in results.items():
        if not isinstance(v[0], T.Tensor):
            v = T.tensor(v)
        experiment_writer.log_data(k, v)

    # add all of the baseline metrics provided in pred_ground_truth into results
    for metric, ground_truth in pred_ground_truth.items():
        experiment_writer.log_data(f"base_{metric}_predictions", ground_truth)

    experiment_writer.write()
