from itertools import cycle
from typing import Generator

import streamlit as st
import torch as T
from data_management import ExperimentReader  # type: ignore
from plotly import graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


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
    num_cols = -1
    max_length = -1
    graph_range = [0.0, 1.0]

    for k, v in data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.split("_")

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

        if model == "base":
            if isinstance(v, list):
                num_cols = len(v)
            else:
                max_length = len(v)
                graph_range = [T.min(v).item() - 0.25, T.max(v).item() + 0.25]


    predictions_plot = make_subplots(rows=len(predictions), cols=num_cols)
    predictions_plot.update_xaxes(showticklabels=False)
    predictions_plot.update_yaxes(showticklabels=False, range=graph_range)

    for metric, values in predictions["base"].items():
        if isinstance(values, T.Tensor):
            for row_idx in range(len(predictions)):
                for col_idx in range(num_cols):
                    predictions_plot.add_trace(
                        go.Scatter(
                            x=T.linspace(0, num_cols, len(values)),
                            y=values.squeeze(),
                            opacity=0.1,
                            line={"color": "lightblue"}
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    for row_idx, ((model, task_data), color) in enumerate(
        zip(predictions.items(), plotly_colors())
    ):
        for col_idx in range(num_cols):
            for metric, values in task_data.items():
                if isinstance(values, list):
                    if len(values[col_idx]) == max_length:
                        x = T.linspace(0, num_cols,  max_length)
                    else:
                        x = T.linspace(col_idx, col_idx + 1, len(values[col_idx]))
                    y = values[col_idx]
                    predictions_plot.add_trace(
                        go.Scatter(x=x.squeeze(), y=y.squeeze(), line={"color": color}),
                        row_idx + 1,
                        col_idx + 1,
                    )
         

    st.plotly_chart(predictions_plot)


def main():
    for experiment in ExperimentReader.get_experiments():
        reader = ExperimentReader(experiment)
        reader.read()

        st.write(f"### {experiment}")

        st.write("## Graphs")
        plot_loss_graphs(reader.data)
        plot_prediction_graphs(reader.data)

        st.write("## Data")
        for name, data in reader.data.items():
            if isinstance(data, list):
                st.write(f"{name}: [{data[0].shape} (x{len(data)})]")
            else:
                st.write(f"{name}: {data.shape}")
            with st.expander("View Data"):
                st.write(str(data))


if __name__ == "__main__":
    main()
