from plotly import express as px  # type: ignore
from plotly import graph_objects as go
from plotly.subplots import make_subplots  # type: ignore
from pathlib import Path
import torch as T
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

    function_plot = px.line(x=X.squeeze(), y=Y.squeeze(), range_y=[-0.25, 1.25])

    predictions_plot = make_subplots(rows=3, cols=NUM_PEAKS)
    predictions_plot.update_xaxes(showticklabels=False)
    predictions_plot.update_yaxes(showticklabels=False)
    for i, (kan_pred, mlp_pred) in enumerate(zip(kan_preds, mlp_preds)):
        plot_on_subplot(
            predictions_plot,
            (1, i + 1),
            px.line(x=X_partitioned[i].squeeze(), y=Y_partitioned[i].squeeze()),
            px.line(x=X.squeeze(), y=Y.squeeze()).update_traces(opacity=0.1),
        )
        plot_on_subplot(
            predictions_plot,
            (2, i + 1),
            px.line(x=X.squeeze(), y=kan_pred.squeeze()),
            px.line(x=X.squeeze(), y=Y.squeeze()).update_traces(opacity=0.1),
        )
        plot_on_subplot(
            predictions_plot,
            (3, i + 1),
            px.line(x=X.squeeze(), y=mlp_pred.squeeze()),
            px.line(x=X.squeeze(), y=Y.squeeze()).update_traces(opacity=0.1),
        )

    return {"function": function_plot, "predictions_plot": predictions_plot}
