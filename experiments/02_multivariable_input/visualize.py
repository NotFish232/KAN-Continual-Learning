from pathlib import Path

import math
import torch as T
from plotly import express as px  # type: ignore
from plotly import graph_objects as go
from plotly.subplots import make_subplots  # type: ignore

from utils import plot_on_subplot

EXPERIMENT_NAME = Path(__file__).parent.name


def create_plots(experiment_data: dict[str, T.Tensor]) -> dict[str, go.Figure]:
    X = experiment_data["X"]
    Y = experiment_data["Y"]
    X_partitioned = experiment_data["X_partitioned"]
    Y_partitioned = experiment_data["Y_partitioned"]
    kan_preds = experiment_data["kan_preds"]
    mlp_preds = experiment_data["mlp_preds"]
    kan_train_loss = experiment_data["kan_train_loss"]
    kan_test_loss = experiment_data["kan_test_loss"]
    mlp_train_loss = experiment_data["mlp_train_loss"]
    mlp_test_loss = experiment_data["mlp_test_loss"]

    NUM_PEAKS = X_partitioned.shape[0]

    function_plot = go.Figure([go.Surface(z=Y.reshape(int(math.sqrt(Y.shape[0])), -1))])

    # predictions_plot = make_subplots(rows=3, cols=NUM_PEAKS)

    # predictions_plot.update_xaxes(showticklabels=False)
    # predictions_plot.update_yaxes(showticklabels=False, range=[-0.25, 2.5])
    # for i, (kan_pred, mlp_pred) in enumerate(zip(kan_preds, mlp_preds)):
    #     plot_on_subplot(
    #         predictions_plot,
    #         (1, i + 1),
    #         px.line(x=X_partitioned[i].squeeze(), y=Y_partitioned[i].squeeze()),
    #         px.line(x=X.squeeze(), y=Y.squeeze(), range_y=[0, 2]).update_traces(
    #             opacity=0.1
    #         ),
    #     )
    #     plot_on_subplot(
    #         predictions_plot,
    #         (2, i + 1),
    #         px.line(x=X.squeeze(), y=kan_pred.squeeze()),
    #         px.line(x=X.squeeze(), y=Y.squeeze()).update_traces(opacity=0.1),
    #     )
    #     plot_on_subplot(
    #         predictions_plot,
    #         (3, i + 1),
    #         px.line(x=X.squeeze(), y=mlp_pred.squeeze()),
    #         px.line(x=X.squeeze(), y=Y.squeeze()).update_traces(opacity=0.1),
    #     )

    kan_train_plot = go.Figure(
        [
            go.Scatter(
                y=kan_train_loss,
                name="KAN Train Loss",
                showlegend=True,
                line={"color": "blue"},
            ),
            go.Scatter(
                y=kan_test_loss,
                name="KAN Test Loss",
                showlegend=True,
                line={"color": "red"},
            ),
        ]
    )
    mlp_train_plot = go.Figure(
        [
            go.Scatter(
                y=mlp_train_loss,
                name="MLP Train Loss",
                showlegend=True,
                line={"color": "blue"},
            ),
            go.Scatter(
                y=mlp_test_loss,
                name="MLP Test Loss",
                showlegend=True,
                line={"color": "red"},
            ),
        ]
    )

    return {
        "Function": function_plot,
        "KAN Plot": kan_train_plot,
        "MLP Plot": mlp_train_plot,
    }
