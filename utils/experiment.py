from typing import Any

import torch as T
from kan import KAN
from torch import nn
from torch.utils.data import Dataset

from utils import kan_reg_term
from utils.data_management import ExperimentDataType, ExperimentWriter
from utils.models import MLP
from utils.training import TrainModelArguments, train_model


def run_experiment(
    experiment_name: str,
    kan_architectures: list[tuple[tuple[list[int], int], int]],
    mlp_architectures: list[tuple[list[int], int]],
    task_datasets: list[Dataset],
    eval_datasets: dict[str, list[Dataset] | Dataset],
    pred_datasets: dict[str, list[T.Tensor] | T.Tensor],
    pred_ground_truth: dict[str, list[T.Tensor] | T.Tensor],
    experiment_dtype: ExperimentDataType,
    device: T.device | None = None,
    kan_kwargs: dict[str, Any] = {},
    mlp_kwargs: dict[str, Any] = {},
    training_args: TrainModelArguments = TrainModelArguments(),
) -> None:
    """
    Runs an experiment on the models provided keeping track of various metrics,
    such as training / evaluation loss and predictions on datasets,
    saves these results to file using `ExperimentWriter` which can later be read
    with `ExperimentReader`

    Results are saved as the following:
    * task_dataset has an associated train_loss of type T.Tensor[T.float32], which is same len as eval results
    * each eval_dataset has an associated ${model}_${eval_dataset_name}_loss entry of type T.Tensor[T.float32]
    * each pred_dataset has an associated ${model}_$(pred_dataset_name)_predictions of type list[T.Tensor] and len(task_datasets)
    * each pred_ground_truth has an associated base_${metric}_{predictions} which is same as value passed in

    Parameters
    ----------
    experiment_name : str
        Name of the experiment, used for writing / reading the experiment

    kan_architectures : list[tuple[tuple[list[int], int], int]]
        Represents the width of each layer of the kan, and grid size, last int is param count

    mlp_architectures : list[tuple[list[int], int]]
        Represents the width of each layer of the mlp, last int is param count

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

    training_args : TrainModelArguments, optional
        Additional kwargs to pass to training function, by default TrainModelArguments()

    Returns
    ----------
    None
    """

    # use passed device or cuda if available else cpu
    device = device or T.device("cuda" if T.cuda.is_available() else "cpu")

    models: dict[str, nn.Module] = {}
    for (architecture, grid_size), param_count in kan_architectures:
        kan = KAN(architecture, grid_size, device=device, **kan_kwargs)
        models[f"kan_{param_count}"] = kan
    for architecture, param_count in mlp_architectures:
        mlp = MLP(architecture, **mlp_kwargs).to(device)
        models[f"mlp_{param_count}"] = mlp

    # all metric evaluations and predictions
    # each metric is of the form {model}_{dataset}_{{metric}|predictions}
    results: dict[str, Any] = {}

    for task_idx, task_dataset in enumerate(task_datasets):
        # prepare datasets which is the task_dataset + each eval_dataset
        datasets = {"train": task_dataset} | {
            metric: d[task_idx] if isinstance(d, list) else d
            for metric, d in eval_datasets.items()
        }

        for model_name, model in models.items():
            # add regularization term to kan training
            base_loss_fn = training_args.loss_fn or nn.MSELoss()  # TODO: FIXME
            training_args.loss_fn = None
            if model_name.startswith("kan"):
                assert isinstance(model, KAN)
                reg = kan_reg_term(model)
                loss_fn = lambda *args, **kwargs: (
                    base_loss_fn(*args, **kwargs) + reg()
                )
            else:
                loss_fn = base_loss_fn

            training_results = train_model(
                model,
                datasets,
                pbar_description=f"{model_name.upper()} Task ({task_idx + 1}/{len(task_datasets)})",
                loss_fn=loss_fn,
                **training_args.to_dict(),
            )

            # update results with training results
            for metric, value in training_results.items():
                if f"{model_name}_{metric}" not in results:
                    results[f"{model_name}_{metric}"] = []
                results[f"{model_name}_{metric}"].extend(value)

            # update results with each model prediction for each pred_dataset
            for metric, pred_dataset in pred_datasets.items():
                with T.no_grad():
                    predictions = model(
                        pred_dataset[task_idx]
                        if isinstance(pred_dataset, list)
                        else pred_dataset
                    )

                if f"{model_name}_{metric}_predictions" not in results:
                    results[f"{model_name}_{metric}_predictions"] = []
                results[f"{model_name}_{metric}_predictions"].append(predictions)

    experiment_writer = ExperimentWriter(experiment_name, experiment_dtype)

    # log some experimental configuration
    experiment_writer.log_config(
        "kan_architectures",
        [*zip((m for m in models if m.startswith("kan")), kan_architectures)],
    )
    experiment_writer.log_config(
        "mlp_architecture",
        [*zip((m for m in models if m.startswith("mlp")), mlp_architectures)],
    )
    experiment_writer.log_config("kan_kwargs", kan_kwargs)
    experiment_writer.log_config("mlp_kwargs", mlp_kwargs)

    # values written to experiment_write should be either list[T.Tensor] or T.Tensor
    # if first element of v is not a T.Tensor assume its supposed to be a scaler tensor and convert
    for k, v in results.items():
        if not isinstance(v[0], T.Tensor):
            v = T.tensor(v)
        experiment_writer.log_data(k, v)

    # add all of the baseline metrics provided in pred_ground_truth into results
    for metric, ground_truth in pred_ground_truth.items():
        experiment_writer.log_data(f"base_{metric}_predictions", ground_truth)

    # add model state dicts
    experiment_writer.log_data(f"kan_state_dict", kan.state_dict())
    experiment_writer.log_data(f"mlp_state_dict", mlp.state_dict())

    # flush changes to disk
    experiment_writer.write()
