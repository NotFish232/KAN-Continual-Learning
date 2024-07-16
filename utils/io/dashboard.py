from itertools import cycle
from typing import Generator

import streamlit as st
import torch as T
from plotly import express as px
from plotly import graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from utils import plot_on_subplot
from utils.io import ExperimentReader, get_experiment_plots, get_experiments


def plotly_colors() -> Generator[str, None, None]:
    yield from cycle(("red", "blue", "green"))


def plot_loss_graphs(data: dict[str, list[T.Tensor] | T.Tensor]) -> None:
    # maps a metric -> a dict of model -> values
    graphs: dict[str, dict[str, T.Tensor]] = {}

    for k, v in data.items():
        if not k.endswith("loss"):
            continue

        model, metric, _ = k.split("_")

        if metric not in graphs:
            graphs[metric] = {}

        assert isinstance(v, T.Tensor)
        graphs[metric][model] = v

    for metric, metric_data in graphs.items():
        traces = []

        for (model, values), color in zip(metric_data.items(), plotly_colors()):
            trace = go.Scatter(
                y=values,
                name=f"{model} {metric} loss",
                showlegend=True,
                line={"color": color},
            )
            traces.append(trace)

        plot = go.Figure(traces)
        st.plotly_chart(plot)


def plot_prediction_graphs(data: dict[str, list[T.Tensor] | T.Tensor]) -> None:
    # maps a model -> a dict of task -> values
    predictions: dict[str, dict[str, list[T.Tensor] | T.Tensor]] = {}
    num_cols = 2

    for k, v in data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.split("_")

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

        if model == "base" and isinstance(v, list):
            num_cols = len(v)

    predictions_plot = make_subplots(rows=len(predictions), cols=num_cols)
    for row_idx, ((model, task_data), color) in enumerate(
        zip(predictions.items(), plotly_colors())
    ):
        for col_idx in range(num_cols):
            traces = []

            for metric, values in task_data.items():
                if isinstance(values, list):
                    x = T.linspace(col_idx, col_idx + 1, len(values[col_idx]))
                    y = values[col_idx]
                    traces.append(
                        px.line(x=x.squeeze(), y=y.squeeze()).update_traces(
                            line_color=color
                        )
                    )
                else:
                    x = T.linspace(0, num_cols, len(values))
                    y = values
                    traces.append(
                        px.line(x=x.squeeze(), y=y.squeeze(), title="test").update_traces(
                            line_color=color, opacity=0.1
                        )
                    )

            plot_on_subplot(predictions_plot, (row_idx + 1, col_idx + 1), *traces)

    st.plotly_chart(predictions_plot)


def main():
    for experiment in ["reproduce_results"]:
        reader = ExperimentReader(experiment)
        reader.read()

        st.write(f"### {experiment}")

        st.write("## Graphs")
        plot_loss_graphs(reader.data)
        plot_prediction_graphs(reader.data)

        st.write("## Data")
        for name, data in reader.data.items():
            if isinstance(data, list):
                st.write(f"{name}: list({len(data)}) of {data[0].shape}")
            else:
                st.write(f"{name}: {data.shape}")
            with st.expander("View Data"):
                st.write(str(data))


if __name__ == "__main__":
    main()
