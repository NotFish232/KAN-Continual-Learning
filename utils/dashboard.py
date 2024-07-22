import math
from itertools import cycle
from typing import Callable, Generator
import json

import streamlit as st
import torch as T
from data_management import ExperimentDataType, ExperimentReader  # type: ignore
from plotly import graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def plotly_colors() -> Generator[str, None, None]:
    """
    Yields sequences of distinctive colors for plotly plots

    Yields
    ------
    Generator[str, None, None]
        Yields indefinitely a cycle of distinctive colors
    """

    yield from cycle(("red", "blue", "green"))


def plot_loss_graphs(experiment_reader: ExperimentReader) -> None:
    """
    Plots the loss graphs of an experiment
    """

    # maps a metric -> a dict of model -> values
    graphs: dict[str, dict[str, T.Tensor]] = {}

    for k, v in experiment_reader.data.items():
        # only process result items that are losses
        if not k.endswith("loss"):
            continue

        # grab the model and metric from the result key
        model, metric, _ = k.rsplit("_", 2)

        if metric not in graphs:
            graphs[metric] = {}

        assert isinstance(v, T.Tensor)
        graphs[metric][model] = v

    # Generate a graph for each metric
    # where each trace is a different model
    for metric, metric_data in graphs.items():
        traces = []

        for (model, values), color in zip(metric_data.items(), plotly_colors()):
            trace = go.Scatter(
                y=values,
                name=f"{model.capitalize()} {metric} loss",
                showlegend=True,
                line={"color": color},
            )
            traces.append(trace)

        plot = go.Figure(
            traces,
            layout=go.Layout(title=go.layout.Title(text=f"{metric.capitalize()} Loss")),
        )
        st.plotly_chart(plot)


def plot_1d_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Creates prediction graphs for 1d functions, i.e., curves
    """

    # maps a model -> a dict of task -> values
    predictions: dict[str, dict[str, list[T.Tensor] | T.Tensor]] = {}

    for k, v in experiment_reader.data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.rsplit("_", 2)

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

    assert "base" in predictions

    # get function specific data like num tasks, num points, and graph range
    # from the metric baselines
    num_tasks = None
    num_points = None
    graph_range = None

    for v in predictions["base"].values():
        if isinstance(v, list):
            # num tasks is len of list since predictions are made each task
            num_tasks = len(v)
        else:
            # num points is len of v if it is a tensor because then v represents the baseline for the whole graph
            # as such, finding the min and max of v will let you find the range of the function
            num_points = len(v)
            graph_range = [T.min(v).item() - 0.25, T.max(v).item() + 0.25]

    # make sure all of the function specific data was found from the baseline
    assert num_tasks is not None and num_points is not None and graph_range is not None

    # create subplots where each row is a model and each column is a task
    plot = make_subplots(rows=len(predictions), cols=num_tasks)
    plot.update_xaxes(showticklabels=False)
    plot.update_yaxes(showticklabels=False, range=graph_range)
    plot.update_layout({"title": {"text": "Predictions"}})

    for metric, values in predictions["base"].items():
        # plot all non task specific baselines on all subplots
        if isinstance(values, T.Tensor):
            for row_idx in range(len(predictions)):
                for col_idx in range(num_tasks):
                    plot.add_trace(
                        go.Scatter(
                            x=T.linspace(0, num_tasks, len(values)),
                            y=values.squeeze(),
                            opacity=0.1,
                            line={"color": "lightblue"},
                            name="Base Function",
                            legendgroup="base_background",
                            showlegend=row_idx + col_idx == 0,
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    for row_idx, ((model, task_data), color) in enumerate(
        zip(predictions.items(), plotly_colors())
    ):
        for col_idx in range(num_tasks):
            for metric, values in task_data.items():
                if isinstance(values, list):
                    # if prediction is the same length as the max function length baseline
                    # then plot it over the entire graph
                    # otherwise its a graph of a task and shold be plotted on a subset of the graph
                    if len(values[col_idx]) == num_points:
                        x = T.linspace(0, num_tasks, num_points)
                    else:
                        x = T.linspace(col_idx, col_idx + 1, len(values[col_idx]))

                    y = values[col_idx]
                    plot.add_trace(
                        go.Scatter(
                            x=x.squeeze(),
                            y=y.squeeze(),
                            line={"color": color},
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=col_idx == 0,
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    st.plotly_chart(plot)


def plot_2d_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Creates prediction graphs for 2d functions, i.e., surfaces
    """

    # maps a model -> a dict of task -> values
    predictions: dict[str, dict[str, list[T.Tensor] | T.Tensor]] = {}

    for k, v in experiment_reader.data.items():
        if not k.endswith("predictions"):
            continue

        model, metric, _ = k.rsplit("_", 2)

        if model not in predictions:
            predictions[model] = {}

        predictions[model][metric] = v

    assert "base" in predictions

    # get function specific data like num tasks, num points, and graph range
    # from the metric baselines
    num_tasks = None
    num_points = None
    graph_range = None

    for v in predictions["base"].values():
        if isinstance(v, list):
            # num tasks is len of list since predictions are made each task
            num_tasks = len(v)
        else:
            # num points is len of v if it is a tensor because then v represents the baseline for the whole graph
            # as such, finding the min and max of v will let you find the range of the function
            num_points = len(v)
            graph_range = [T.min(v).item() - 0.25, T.max(v).item() + 0.25]

    # make sure all of the function specific data was found from the baseline
    assert num_tasks is not None and num_points is not None and graph_range is not None

    # 2d task so num_points represents num_points on each axis
    num_points = round(math.sqrt(num_points))

    # create subplots where each row is a model and each column is a task
    plot = make_subplots(
        rows=len(predictions) - 1,
        cols=num_tasks,
        specs=[
            [{"type": "surface"} for _ in range(num_tasks)]
            for _ in range(len(predictions) - 1)
        ],
    )
    # add plot title
    plot.update_layout({"title": {"text": "Predictions"}})

    # remove ticks from all subplots
    plot.update_layout(
        {
            f"scene{i}": {
                axis: {"showticklabels": False} for axis in ("xaxis", "yaxis", "zaxis")
            }
            for i in range(1, num_tasks * len(predictions))
        }
    )

    for metric, values in predictions["base"].items():
        # plot all non task specific baselines on all subplots
        if isinstance(values, T.Tensor):
            for row_idx in range(len(predictions) - 1):
                for col_idx in range(num_tasks):
                    plot.add_trace(
                        go.Surface(
                            z=values.reshape([round(math.sqrt(num_points))] * 4)
                            .permute(0, 2, 1, 3)
                            .reshape(num_points, num_points),
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=row_idx + col_idx == 0,
                            opacity=0.1,
                            showscale=False,
                            hoverinfo="skip",
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    # don't draw other baseline stuff for 2d
    del predictions["base"]

    for row_idx, (model, task_data) in enumerate(predictions.items()):
        for col_idx in range(num_tasks):
            for metric, values in task_data.items():
                if isinstance(values, list):
                    plot.add_trace(
                        go.Surface(
                            z=values[col_idx]
                            .reshape([round(math.sqrt(num_points))] * 4)
                            .permute(0, 2, 1, 3)
                            .reshape(num_points, num_points),
                            name=f"{model.capitalize()} {metric.capitalize()}",
                            legendgroup=model,
                            showlegend=col_idx == 0,
                            showscale=False,
                            hoverinfo="skip",
                        ),
                        row_idx + 1,
                        col_idx + 1,
                    )

    st.plotly_chart(plot)


def plot_prediction_graph(experiment_reader: ExperimentReader) -> None:
    """
    Calls either `plot_1d_prediction_graphs` or `plot_2d_prediction_graphs`
    depending on `experiment_reader.experiment_dtype`
    """

    match experiment_reader.experiment_dtype:
        case ExperimentDataType.function_1d:
            plot_1d_prediction_graph(experiment_reader)
        case ExperimentDataType.function_2d:
            plot_2d_prediction_graph(experiment_reader)


def write_data(experiment_reader: ExperimentReader) -> None:
    """
    Writes the data section of the experiment_reader to streamlit
    """

    for name, obj in experiment_reader.data.items():
        if isinstance(obj, dict):
            st.write(f"{name}: { {k for k in obj.keys()} }")
        elif isinstance(obj, list):
            st.write(f"{name}: [{obj[0].shape} (x{len(obj)})]")
        else:
            st.write(f"{name}: {obj.shape}")
        with st.expander("View Data"):
            st.write(str(obj))


def write_config(experiment_reader: ExperimentReader) -> None:
    """
    Writes the config section of the experiment_reader to streamlit
    """

    for name, obj in experiment_reader.config.items():
        st.write(f"{name}: {json.dumps(obj, indent=4)}")


@st.cache_data
def fetch_experiment_reader(experiment: str) -> ExperimentReader:
    """
    Fetchs and loads the experiment_reader from an experiment
    Cached using `@st.cache_data`
    """

    reader = ExperimentReader(experiment)
    reader.read()

    return reader


def page_function(experiment: str) -> Callable:
    def _page_function() -> None:
        experiment_reader = fetch_experiment_reader(experiment)

        st.write(f"# {experiment}")
        st.write("##")

        st.write("## Graphs")
        st.write("")
        st.write("")
        plot_loss_graphs(experiment_reader)
        plot_prediction_graph(experiment_reader)

        st.write("## Data")
        st.write("")
        write_data(experiment_reader)
        st.write("##")

        st.write("## Config")
        st.write("")
        write_config(experiment_reader)

    return _page_function


def main():
    pages = [
        st.Page(page_function(e), title=e, url_path=e)
        for e in ExperimentReader.get_experiments()
    ]
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
